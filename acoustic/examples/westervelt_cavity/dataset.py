"""Full-order generation, mass POD, actual FEM labels, and lifted-state augmentation."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,time,os
import numpy as np
from scipy.linalg import eigh
from scipy.linalg.blas import dsyrk
from numba import set_num_threads
from common import read,write,sha,token,tag,cases,log,cache,complete
from physics import build,cycle_fast,native_rhs,drive,lifted_rhs

def generate_case(task,cfg,output,identity):
    amp,factor=task;folder=Path(output)/"generated"/tag(amp,factor);folder.mkdir(exist_ok=True)
    files=[folder/"basis_snapshots.npy",folder/"final_state.npz"]
    previous=cache(folder/"manifest.json",identity,files)
    if previous:return previous
    set_num_threads(cfg["fem_threads"])
    x,y,w,lap,source,radiation=build(cfg);n=len(w);spp=cfg["steps_per_period"]
    omega=cfg["omega_reference"]*factor
    dt=2*np.pi/omega/spp
    bound=float(np.max(np.asarray(abs(lap).sum(axis=1))))
    if dt*max(np.sqrt(bound),cfg["eta"]*bound)>=.45:
        raise ValueError("Explicit RK4 stability precheck failed; refine the step count.")
    p=np.zeros(n);v=np.zeros(n)
    fields=[np.empty((128,n),np.float32) for _ in range(3)]
    sample=np.arange(0,spp,spp//128)
    ends=[];extra=[];extra_times=[];last=None;streak=0;error=None;minimum=1.
    start=time.time()
    for j in range(cfg["max_cycles"]):
        minimum=min(minimum,float(cycle_fast(p,v,lap.indptr,lap.indices,lap.data,radiation,
            omega,cfg["beta"],cfg["eta"],amp,spp,j,sample,*fields,source)))
        ends.append(p.astype(np.float32))
        if j+1 in cfg["extra_snapshot_cycles"]:
            phases=np.arange(4,128,16)
            extra.append(fields[0][phases].copy())
            extra_times.extend((j+phases/128).tolist())
        if last is not None:
            numerator=denominator=0.
            for k in range(0,128,8):
                block=fields[0][k:k+8].astype(float);delta=block-last[k:k+8]
                numerator+=float(np.sum(delta*delta*w))
                denominator+=float(np.sum(block*block*w))
            error=float(np.sqrt(numerator/max(denominator,1e-300)))
        last=fields[0].copy()
        if j+1>=cfg["min_cycles"] and error is not None and error<cfg["cycle_tolerance"]:streak+=1
        else:streak=0
        if (j+1)%10==0 or j+1==cfg["max_cycles"] or streak>=cfg["consecutive_cycles"]:
            write(folder/"progress.json",dict(identity=identity,cycle=j+1,pressure_cycle_difference=error,minimum_mass_factor=minimum))
            log("generate_case",case=folder.name,cycle=j+1,pressure_cycle_difference=error)
        if cfg["converge"] and streak>=cfg["consecutive_cycles"]:break
    if cfg["converge"] and streak<cfg["consecutive_cycles"]:
        raise RuntimeError("FEM convergence criterion not met for "+folder.name)
    if len(extra)!=8 or len(ends)<16:raise ValueError("Insufficient horizon for 144 declared pressure snapshots.")
    ids=np.rint(np.linspace(0,len(ends)-1,16)).astype(int)
    snapshots=np.concatenate([np.stack([ends[k] for k in ids]),fields[0][::2],np.concatenate(extra)])
    if snapshots.shape!=(144,n) or not np.isfinite(snapshots).all():raise ValueError("Invalid snapshots")
    np.save(files[0],snapshots)
    np.savez(files[1],pressure=p,pressure_rate=v)
    times=np.r_[ids+1,j+np.arange(0,128,2)/128,extra_times]
    complete(folder/"manifest.json",identity,files,amplitude=amp,frequency_factor=factor,omega=omega,
        cycles=j+1,nodes=n,snapshots_per_case=144,snapshot_time_over_T=times.tolist(),
        pressure_cycle_difference=error,consecutive_cycles=streak,minimum_mass_factor=minimum,
        convergence_required=cfg["converge"],elapsed_seconds=time.time()-start)
    return read(folder/"manifest.json")

def generate(cfg,output,identity):
    records=[];tasks=cases(cfg,"pod")
    # Each case has its own completed manifest; interrupted campaigns reuse completed cases.
    with ProcessPoolExecutor(max_workers=cfg["fem_workers"]) as pool:
        jobs={pool.submit(generate_case,task,cfg,str(output),identity):task for task in tasks}
        for job in as_completed(jobs):
            records.append(job.result())
            write(output/"generated"/"campaign.json",dict(identity=identity,completed=len(records),total=len(tasks),cases=records))
    x,y,w,_,source,radiation=build(cfg)
    np.savez(output/"generated"/"mesh.npz",x=x,y=y,half_mass=w,mass=2*w,source=source,radiation=radiation)
    complete(output/"generated"/"manifest.json",identity,[output/"generated"/"mesh.npz"],
        cases=[dict(amplitude=a,frequency_factor=f,tag=tag(a,f)) for a,f in tasks],
        pod_case_count=len(tasks),nn_case_count=len(cases(cfg,"nn")),
        scope="All POD parameter cases are full-order data; only configured nn_cases contribute to NN fit and normalization.")

def build_pod(cfg,output,identity):
    pod=output/"pod";phi_path=pod/"phi.npy";manifest=pod/"manifest.json"
    previous=cache(manifest,identity,[phi_path])
    if previous:return previous
    root_manifest=read(output/"generated"/"manifest.json")
    if not cache(output/"generated"/"manifest.json",identity,[output/"generated"/"mesh.npz"]):raise ValueError("Complete current generation and unchanged mesh required.")
    with np.load(output/"generated"/"mesh.npz",allow_pickle=False) as mesh:mass=mesh["mass"].astype(float)
    tasks=cases(cfg,"pod");nodes=len(mass);rows=144*len(tasks)
    sw=np.sqrt(mass);constant=sw/np.sqrt(mass.sum())
    scratch=output/"cache"/"weighted_snapshots.npy";progress=pod/"weighted_progress.json"
    state=read(progress) if progress.exists() else {}
    if scratch.exists() and state.get("identity")==identity:
        weighted=np.load(scratch,mmap_mode="r+");done=int(state["completed"])
        if weighted.shape!=(rows,nodes):raise ValueError("POD scratch shape changed.")
    else:
        weighted=np.lib.format.open_memmap(scratch,mode="w+",dtype=np.float64,shape=(rows,nodes));done=0
    for i,(amp,factor) in enumerate(tasks):
        if i<done:continue
        folder=output/"generated"/tag(amp,factor)
        if not cache(folder/"manifest.json",identity,[folder/"basis_snapshots.npy",folder/"final_state.npz"]):raise ValueError("Incomplete generated case.")
        block=np.array(np.load(folder/"basis_snapshots.npy",mmap_mode="r"),dtype=float)
        block*=sw;norm=np.linalg.norm(block)
        if not norm>0:raise ValueError("Zero-energy POD case.")
        block/=norm
        means=block@constant
        block-=means[:,None]*constant[None,:]
        weighted[i*144:(i+1)*144]=block;weighted.flush()
        write(progress,dict(identity=identity,completed=i+1,total=len(tasks)))
        log("pod_weighted_snapshots",completed=i+1,total=len(tasks))
    gram_file=output/"cache"/"snapshot_gram.npy";gram_manifest=pod/"gram_manifest.json"
    if cache(gram_manifest,identity,[gram_file]):
        gram=np.array(np.load(gram_file,mmap_mode="r"),dtype=float,order="F")
    else:
        gram=np.zeros((rows,rows),dtype=float,order="F")
        for first in range(0,nodes,cfg["node_block"]):
            last=min(first+cfg["node_block"],nodes)
            slab=np.array(weighted[:,first:last],dtype=float,order="F")
            gram=dsyrk(alpha=1.,a=slab,beta=1.,c=gram,lower=1,overwrite_c=1)
            log("pod_gram",nodes_completed=last,nodes_total=nodes)
        for row in range(rows):gram[row,row+1:]=gram[row+1:,row]
        np.save(gram_file,gram);complete(gram_manifest,identity,[gram_file])
    count=min(cfg["pod_available_rank"]-1+8,rows-1)
    eigenvalues,vectors=eigh(gram,subset_by_index=[rows-count,rows-1],overwrite_a=True,check_finite=False)
    eigenvalues=eigenvalues[::-1];vectors=vectors[:,::-1]
    good=eigenvalues>eigenvalues[0]*1e-13;eigenvalues=eigenvalues[good];vectors=vectors[:,good]
    k=min(cfg["pod_available_rank"]-1,len(eigenvalues))
    if k+1<cfg["rank"]:raise ValueError("Snapshots cannot support the requested POD rank.")
    modes=np.empty((nodes,k+1));modes[:,0]=constant
    for first in range(0,nodes,cfg["node_block"]):
        last=min(first+cfg["node_block"],nodes)
        slab=np.array(weighted[:,first:last],dtype=float)
        modes[first:last,1:]=(slab.T@vectors[:,:k])/np.sqrt(eigenvalues[:k])
    modes,_=np.linalg.qr(modes,mode="reduced")
    orth=float(np.linalg.norm(modes.T@modes-np.eye(k+1)))
    if orth>=1e-8:raise ValueError("Mass POD orthogonalization failed.")
    modes/=sw[:,None]
    np.save(phi_path,modes[:,:cfg["rank"]].astype(np.float32))
    stored=np.load(phi_path).astype(float)
    stored_orth=float(np.linalg.norm(stored.T@(mass[:,None]*stored)-np.eye(cfg["rank"])))
    if not np.isfinite(stored).all() or stored.shape!=(nodes,cfg["rank"]) or stored_orth>=1e-6:
        raise ValueError("Stored float32 POD basis has invalid shape, values, or mass orthogonality.")
    complete(manifest,identity,[phi_path],rank=cfg["rank"],available_rank=k+1,cases=len(tasks),snapshots_per_case=144,
        constant_direction_explicit=True,basis_id=sha(phi_path),orthogonality_frobenius_float64=orth,stored_mass_orthogonality_frobenius=stored_orth,
        scope="Mass-weighted, case-balanced POD from all pod_cases. Fixed retained rank; no assertion that every full-field truncation error meets 1 percent.")
    return read(manifest)

def project_exact(task,cfg,output,identity,phi):
    amp,factor=task;name=tag(amp,factor);folder=output/"exact"/name;folder.mkdir(exist_ok=True)
    files=[folder/"input.npy",folder/"target.npy",folder/"sample_steps.npy",folder/"acceleration_energy.npy"]
    bid=sha(output/"pod"/"phi.npy");exact_identity=token(dict(run=identity,basis=bid,case=name))
    previous=cache(folder/"manifest.json",exact_identity,files)
    if previous:return previous
    record=read(output/"generated"/name/"manifest.json");cycles=int(record["cycles"])
    set_num_threads(cfg["fem_threads"])
    x,y,w,lap,source,radiation=build(cfg);mass=2*w;mphi=(mass[:,None]*phi).astype(np.float32)
    n=len(w);r=cfg["rank"];spp=cfg["steps_per_period"];omega=cfg["omega_reference"]*factor
    count=min(cfg["exact_samples_per_case"],cycles*128+1)
    ids=np.rint(np.linspace(0,cycles*128,count)).astype(np.int64)
    if count<cfg["base_rows"]+cfg["validation_rows"]+cfg["test_rows"] or len(np.unique(ids))!=count:raise ValueError("Insufficient disjoint exact sample rows.")
    xx=np.empty((count,2*r+1),np.float32);yy=np.empty((count,r),np.float32);energy=np.empty(count)
    p=np.zeros(n);v=np.zeros(n);fields=[np.empty((128,n),np.float32) for _ in range(3)];sample=np.arange(0,spp,spp//128)
    for j in range(cycles):
        cycle_fast(p,v,lap.indptr,lap.indices,lap.data,radiation,omega,cfg["beta"],cfg["eta"],amp,spp,j,sample,*fields,source)
        rows=np.flatnonzero((ids>=j*128)&(ids<(j+1)*128));phases=ids[rows]-j*128
        projections=[field@mphi for field in fields]
        xx[rows,:r]=projections[0][phases];xx[rows,r:2*r]=projections[1][phases]
        yy[rows]=projections[2][phases]
        # The acceleration label is captured from the ACTUAL full FEM state,
        # not from the reconstructed POD state used for nonzero perturbations.
        energy[rows]=np.sum(fields[2][phases].astype(float)**2*mass,axis=1)
        if (j+1)%10==0 or j+1==cycles:
            write(folder/"progress.json",dict(cycle=j+1,total=cycles));log("project_actual_fem",case=name,cycle=j+1,total=cycles)
    acc=native_rhs(p,v,drive(cycles,amp),cfg,lap,source,radiation)
    xx[-1,:r]=p@mphi;xx[-1,r:2*r]=v@mphi;yy[-1]=acc@mphi
    energy[-1]=float(np.sum(acc*acc*mass))
    xx[:,-1]=np.array([drive(float(step)/128,amp) for step in ids])
    with np.load(output/"generated"/name/"final_state.npz",allow_pickle=False) as native:
        difference=float(np.sqrt(np.sum(w*(p-native["pressure"])**2)/max(float(np.sum(w*native["pressure"]**2)),1e-300)))
    if not difference<1e-9:raise ValueError("FEM reintegration disagrees with generated final state.")
    if not np.all(xx[0]==0) or not np.all(yy[0]==0):raise ValueError("Physical initial state/label must be zero.")
    if not (np.isfinite(xx).all() and np.isfinite(yy).all() and np.isfinite(energy).all() and np.all(energy>=0)):
        raise ValueError("Exact FEM labels contain invalid values.")
    np.save(files[0],xx);np.save(files[1],yy);np.save(files[2],ids*(spp//128));np.save(files[3],energy)
    complete(folder/"manifest.json",exact_identity,files,amplitude=amp,frequency_factor=factor,samples=count,cycles=cycles,
        basis_id=bid,rank=r,run_identity=identity,final_reintegration_relative=difference,
        target="Mass projection of actual full-order FEM pressure acceleration; float32 periodic capture, native float64 final endpoint.")

def augment_case(task,cfg,output,identity,operators):
    amp,factor=task;name=tag(amp,factor);folder=output/"augmented"/name;folder.mkdir(exist_ok=True)
    files=[folder/(name+".npy") for name in ("train_input","train_target","val_input","val_target","test_input","test_target")]+[folder/"split_indices.npz"]
    current=token(dict(run=identity,basis=sha(output/"pod"/"phi.npy"),exact=sha(output/"exact"/name/"manifest.json")))
    if cache(folder/"manifest.json",current,files):return
    xx=np.load(output/"exact"/name/"input.npy");yy=np.load(output/"exact"/name/"target.npy")
    r=cfg["rank"];levels=np.linspace(-.2,.2,17).astype(np.float32);base_count=cfg["base_rows"];pert_count=cfg["perturbed_base_rows"]
    rng=np.random.default_rng(cfg["seed"]+round(amp*1000)*10000+round(factor*1000))
    perm=rng.permutation(len(xx));base=perm[:base_count];val=perm[base_count:base_count+cfg["validation_rows"]]
    test=perm[base_count+cfg["validation_rows"]:base_count+cfg["validation_rows"]+cfg["test_rows"]]
    if len(np.unique(np.r_[base,val,test]))!=len(base)+len(val)+len(test):raise ValueError("Split indices overlap.")
    amplitude=np.max(np.abs(xx[base,:2*r]),axis=0)
    directions=np.asarray([rng.uniform(-1,1,2*r)*amplitude for _ in base],np.float32)
    a0=xx[base];states=np.repeat(a0[:pert_count],17,axis=0)
    states[:,:2*r]+=(levels[None,:,None]*directions[:pert_count,None,:]).reshape(-1,2*r)
    states=np.concatenate([states,a0[pert_count:]])
    if len(states)!=cfg["augmentation_rows"]:raise ValueError("Augmentation count mismatch.")
    targets=np.empty((len(states),r),np.float32);minimum=1.;set_num_threads(cfg["augmentation_threads"])
    op=operators
    for first in range(0,pert_count,8):
        last=min(first+8,pert_count);q=a0[first:last,:r];v=a0[first:last,r:2*r]
        dq=directions[first:last,:r];dv=directions[first:last,r:2*r]
        rhs,local,_=lifted_rhs(q@op["pt"],dq@op["pt"],v@op["pt"],dv@op["pt"],
            (q+cfg["eta"]*v)@op["lpt"],(dq+cfg["eta"]*dv)@op["lpt"],
            np.ascontiguousarray(a0[first:last,-1]),levels,op["src"],op["rad"],np.float32(2*cfg["beta"]))
        minimum=min(minimum,float(local))
        if local<=.02 or not np.isfinite(local):raise ValueError("Invalid perturbed Westervelt mass.")
        projected=rhs@op["mp"]
        projected[np.arange(last-first)*17+8]=yy[base[first:last]]
        targets[first*17:last*17]=projected
        if first%128==0 or last==pert_count:log("augment_nonzero_levels",case=name,base_completed=last,total=pert_count)
    targets[pert_count*17:]=yy[base[pert_count:]]
    axis_count=2*r*17;axes=np.zeros((axis_count,2*r+1),np.float32)
    axes[:,:2*r]=(np.diag(amplitude)[:,None,:]*levels[None,:,None]).reshape(-1,2*r)
    anchors=[]
    for first in range(0,2*r,8):
        direction=np.diag(amplitude)[first:first+8];dq=direction[:,:r];dv=direction[:,r:]
        zero=np.zeros((len(direction),op["pt"].shape[1]),np.float32)
        rhs,local,_=lifted_rhs(zero,dq@op["pt"],zero,dv@op["pt"],zero,
            (dq+cfg["eta"]*dv)@op["lpt"],np.zeros(len(direction),np.float32),levels,op["src"],op["rad"],np.float32(2*cfg["beta"]))
        minimum=min(minimum,float(local))
        if local<=.02:raise ValueError("Invalid zero-axis state.")
        anchors.append(rhs@op["mp"])
    train_x=np.concatenate([states,axes]);train_y=np.concatenate([targets,np.concatenate(anchors)])
    for i,value in enumerate((train_x,train_y,xx[val],yy[val],xx[test],yy[test])):
        if not np.isfinite(value).all():raise ValueError("Augmentation has nonfinite values.")
        np.save(files[i],value)
    np.savez(files[6],base=base,validation=val,test=test)
    complete(folder/"manifest.json",current,files,run_identity=identity,basis_id=sha(output/"pod"/"phi.npy"),
        rows=len(train_x),axis_rows=axis_count,modal_amplitude=amplitude.tolist(),minimum_mass_factor=minimum,
        levels=levels.tolist(),forcing="Unchanged on trajectory perturbations; zero for axis anchors.",
        labels="Nonzero levels: nonlinear FEM RHS at reconstructed lifted states. Zero trajectory level: unchanged actual FEM label. Axis anchors: recomputed RHS, repeated zero rows retained.")
def pod_and_augment(cfg,output,identity):
    build_pod(cfg,output,identity)
    phi=np.load(output/"pod"/"phi.npy")
    for task in cases(cfg,"nn"):project_exact(task,cfg,output,identity,phi)
    x,y,w,lap,source,radiation=build(cfg)
    op=dict(pt=np.ascontiguousarray(phi.T,np.float32),lpt=np.ascontiguousarray((lap@phi).T,np.float32),
        mp=np.ascontiguousarray(2*w[:,None]*phi,np.float32),src=np.asarray(source,np.float32),rad=np.asarray(radiation,np.float32))
    for task in cases(cfg,"nn"):augment_case(task,cfg,output,identity,op)
    complete(output/"pod"/"pipeline_manifest.json",identity,[output/"pod"/"phi.npy"],
        generated_pod_cases=len(cases(cfg,"pod")),nn_data_cases=len(cases(cfg,"nn")),
        native_label_source="Actual full-order trajectories; not reconstructed-state labels.",
        augmentation_label_source="Recomputed coupled FEM RHS at reconstructed POD lifts, except zero-level actual labels.")
