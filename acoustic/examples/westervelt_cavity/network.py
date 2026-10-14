"""Fixed-hidden double-Tanh output fitting and autonomous CPU RK4."""
import itertools,time,platform,sys
from pathlib import Path
import numpy as np
from scipy.linalg import lstsq
import scipy,numba
from numba import njit
from threadpoolctl import threadpool_limits,threadpool_info
from common import read,write,sha,token,tag,cases,cache,complete,available_memory,GIB,log

def g(x):return np.tanh(.5*np.tanh(x))
def gp(x):return .5*(1-np.tanh(.5*np.tanh(x))**2)*(1-np.tanh(x)**2)
def hidden(x,w):
    n=x.shape[1];z=x/w["scale"];linear=g(w["eps"][0]*z)
    pair=g(w["eps"][0]*(z[:,w["ii"]]+z[:,w["jj"]])*w["factor"]+.5)
    tri=w["triples"]
    cubic=g(w["eps3"][0]*(z[:,tri[:,0]]+z[:,tri[:,1]]+z[:,tri[:,2]])*w["cfactor"])
    return np.c_[linear,pair,cubic]
def weights(scale,cfg):
    r=cfg["rank"];n=2*r+1
    pr=min(r,cfg["pair_rank"])
    coordinates=np.r_[np.arange(pr),np.arange(r,r+pr),[2*r]]
    a,b=np.triu_indices(len(coordinates));ii,jj=coordinates[a],coordinates[b]
    factor=np.where(ii==jj,.5,1/np.sqrt(2))
    cr=min(r,cfg["cubic_rank"]);selected=list(range(cr))+list(range(r,r+cr))+[2*r]
    triples=np.array(list(itertools.combinations_with_replacement(selected,3)),int)
    cfactor=np.array([1/np.sqrt(sum(list(t).count(k)**2 for k in set(t))) for t in triples])
    return dict(scale=scale,ii=ii,jj=jj,factor=factor,eps=np.array([cfg["epsilon"]]),triples=triples,
        cfactor=cfactor,eps3=np.array([cfg["cubic_epsilon"]]))
def design(x,w):
    raw=hidden(x,w);n=x.shape[1];nq=len(w["ii"]);l=raw[:,:n]
    eps=w["eps"][0];eps3=w["eps3"][0];ratio=gp(.5)/gp(0.)
    return np.c_[np.ones(len(x)),l/(eps*gp(0.)),
        (raw[:,n:n+nq]-g(.5)-ratio*(l[:,w["ii"]]+l[:,w["jj"]])*w["factor"])/(eps*eps),
        (raw[:,n+nq:]-(eps3/eps)*(l[:,w["triples"][:,0]]+l[:,w["triples"][:,1]]+l[:,w["triples"][:,2]])*w["cfactor"])/(eps3**3)]
def fold(coef,w,n):
    eps=w["eps"][0];eps3=w["eps3"][0];ii=w["ii"];jj=w["jj"];tri=w["triples"];nq=len(ii)
    ratio=gp(.5)/gp(0.)
    paircoef=coef[1+n:1+n+nq]/eps**2
    cubiccoef=coef[1+n+nq:]/eps3**3
    linearcoef=coef[1:1+n]/(eps*gp(0.))
    for k in range(nq):
        linearcoef[ii[k]]-=ratio*w["factor"][k]*paircoef[k]
        linearcoef[jj[k]]-=ratio*w["factor"][k]*paircoef[k]
    for k in range(len(tri)):
        for j in tri[k]:linearcoef[j]-=(eps3/eps)*w["cfactor"][k]*cubiccoef[k]
    w["Wout"]=np.r_[linearcoef,paircoef,cubiccoef].T.copy()
    w["bout"]=coef[0]-g(.5)*paircoef.sum(0)
    return w
def selected_indices(cfg):
    r=cfg["rank"];nonzero=np.flatnonzero(np.arange(cfg["perturbed_base_rows"]*17)%17!=8)
    rng=np.random.default_rng(cfg["training_selection_seed"]);wanted=set(cases(cfg,"nn"));result={}
    # Consume random choices in all 44 sorted POD-case positions, reproducing
    # the original parent algorithm without needing its trained NN or indices.
    for task in sorted(cases(cfg,"pod")):
        chosen=rng.choice(nonzero,cfg["selected_perturbation_rows"],replace=False)
        if task in wanted:
            result[tag(*task)]=np.r_[chosen,np.arange(cfg["augmentation_rows"],cfg["augmentation_rows"]+2*r*17)]
    if len(result)!=len(wanted):raise ValueError("Every NN case must be included in the configured POD cases.")
    return result
def evaluate(w,split,cfg,output,gram):
    numerator=denominator=full_num=full_den=0.;records=[]
    for task in cases(cfg,"nn"):
        name=tag(*task);folder=output/"augmented"/name
        xx=np.load(folder/(split+"_input.npy"),mmap_mode="r");yy=np.load(folder/(split+"_target.npy"),mmap_mode="r")
        with np.load(folder/"split_indices.npz") as ids:selected=ids["validation" if split=="val" else "test"]
        energy=np.load(output/"exact"/name/"acceleration_energy.npy")[selected]
        cn=cd=cfn=0.
        for first in range(0,len(xx),256):
            x=np.asarray(xx[first:first+256],float);y=np.asarray(yy[first:first+256],float)
            pred=hidden(x,w)@w["Wout"].T+w["bout"];delta=pred-y
            cn+=float(np.sum(delta*delta));cd+=float(np.sum(y*y))
            residual=energy[first:first+len(x)]-2*np.sum(pred*y,axis=1)+np.einsum("ni,ij,nj->n",pred,gram,pred)
            cfn+=float(np.maximum(residual,0).sum())
        cfd=float(energy.sum())
        records.append(dict(case=name,relative_L2_percent=float(100*np.sqrt(cn/max(cd,1e-300))),
            full_field_relative_L2_percent=float(100*np.sqrt(cfn/max(cfd,1e-300)))))
        numerator+=cn;denominator+=cd;full_num+=cfn;full_den+=cfd
    return dict(global_relative_L2_percent=float(100*np.sqrt(numerator/max(denominator,1e-300))),
        full_field_global_relative_L2_percent=float(100*np.sqrt(full_num/max(full_den,1e-300))),
        full_field_includes_POD_truncation=True,per_case=records)
def train(cfg,output,identity):
    folder=output/"training";model_file=folder/"model.json"
    bid=sha(output/"pod"/"phi.npy")
    source_manifest=read(output/"pod"/"pipeline_manifest.json")
    if source_manifest.get("status")!="complete" or source_manifest.get("identity")!=identity:raise ValueError("Complete current POD/augmentation stage required.")
    data_identity={}
    for task in cases(cfg,"nn"):
        name=tag(*task);data=output/"augmented"/name
        meta=read(data/"manifest.json")
        arrays=[data/(part+".npy") for part in ("train_input","train_target","val_input","val_target","test_input","test_target")]+[data/"split_indices.npz"]
        if meta.get("run_identity")!=identity or meta.get("basis_id")!=bid or not cache(data/"manifest.json",meta.get("identity"),arrays):
            raise ValueError("Augmentation identity or actual file hashes disagree: "+name)
        exact=output/"exact"/name
        native=read(exact/"manifest.json")
        native_files=[exact/(part+".npy") for part in ("input","target","sample_steps","acceleration_energy")]
        if native.get("run_identity")!=identity or native.get("basis_id")!=bid or not cache(exact/"manifest.json",native.get("identity"),native_files):
            raise ValueError("Actual FEM-label files or identity disagree: "+name)
        data_identity[name]=sha(data/"manifest.json")
    fit_identity=token(dict(run=identity,basis=bid,data=data_identity))
    previous=cache(folder/"manifest.json",fit_identity,[model_file,folder/"weights.npz"])
    if previous:return previous
    indices=selected_indices(cfg);xs=[];ys=[]
    for task in cases(cfg,"nn"):
        name=tag(*task);ids=indices[name];data=output/"augmented"/name
        x=np.load(data/"train_input.npy",mmap_mode="r");y=np.load(data/"train_target.npy",mmap_mode="r")
        expected=cfg["augmentation_rows"]+34*cfg["rank"]
        if x.shape!=(expected,2*cfg["rank"]+1) or y.shape!=(expected,cfg["rank"]):raise ValueError("Training array shape mismatch.")
        xs.append(np.array(x[ids],float));ys.append(np.array(y[ids],float))
    x=np.concatenate(xs);y=np.concatenate(ys);del xs,ys
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError("Nonfinite training data.")
    scale=np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-12);yscl=np.maximum(np.sqrt(np.mean(y*y,axis=0)),1e-12)
    w=weights(scale,cfg);n=x.shape[1];width=n+len(w["ii"])+len(w["triples"]);columns=width+1
    design_bytes=len(x)*columns*8
    required=2*design_bytes+2*y.nbytes+4*GIB
    available=available_memory()
    write(folder/"preflight.json",dict(design_bytes=design_bytes,estimated_required_memory_bytes=required,
        available_memory_bytes=available,solver="scipy.linalg.lstsq, LAPACK gelsd",condition_cutoff=cfg["least_squares_cond"]))
    if available is not None and available<required:raise MemoryError("Insufficient measured RAM for faithful dense gelsd fit; use a machine with more available memory.")
    h=np.lib.format.open_memmap(output/"tmp"/"design.npy",mode="w+",dtype=float,fortran_order=True,shape=(len(x),columns))
    start=time.time()
    with threadpool_limits(limits=cfg["blas_threads"],user_api="blas"):
        for first in range(0,len(x),256):
            h[first:first+256]=design(x[first:first+256],w)
            if first%8192==0:log("build_design",rows_completed=first,total=len(x))
        h.flush()
        solver_start=time.time()
        coef,_,rank,singular=lstsq(h,np.asfortranarray(y/yscl),cond=cfg["least_squares_cond"],
            overwrite_a=True,check_finite=False,lapack_driver="gelsd")
        solver_seconds=time.time()-solver_start
        del h
        coef*=yscl
        reference=design(x[:min(100,len(x))],w)@coef
        w=fold(coef,w,n)
        ordinary=hidden(x[:min(100,len(x))],w)@w["Wout"].T+w["bout"]
        parity=float(np.linalg.norm(ordinary-reference)/max(np.linalg.norm(reference),1e-300))
        if not np.isfinite(parity) or parity>=1e-7:raise ValueError("Output folding equivalence failed.")
        if not all(np.isfinite(value).all() for value in w.values()):raise ValueError("Nonfinite fitted weights.")
        phi=np.load(output/"pod"/"phi.npy").astype(float)
        with np.load(output/"generated"/"mesh.npz") as mesh:mass=mesh["mass"]
        gram=phi.T@(mass[:,None]*phi)
        validation=evaluate(w,"val",cfg,output,gram);test=evaluate(w,"test",cfg,output,gram)
        pools=threadpool_info()
    np.savez(folder/"weights.npz",**w);np.savez(folder/"normalization.npz",scale=scale,output_scale=yscl)
    np.savez(folder/"coefficient_singular_values.npz",coefficient=coef,singular_values=singular)
    write(folder/"training_indices.json",{name:ids.tolist() for name,ids in indices.items()})
    model={key:value.tolist() for key,value in w.items()}
    model.update(rank=cfg["rank"],basis_id=bid,run_identity=identity,architecture=[n,width,width,cfg["rank"]],
        precision="float64",trained_parameters="Output weights and bias only; hidden parameters fixed.")
    write(model_file,model)
    summary=dict(architecture=[n,width,width,cfg["rank"]],training_rows=len(x),rows_per_case=len(next(iter(indices.values()))),
        fit_case_count=len(indices),shared_POD_case_count=len(cases(cfg,"pod")),least_squares_rank=int(rank),
        solver_seconds=solver_seconds,fit_and_evaluation_seconds=time.time()-start,folding_relative=parity,
        normalization="RMS of the selected NN training rows only; all other POD cases excluded.",
        validation=validation,test=test,actual_threadpools=pools,
        versions=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__),
        scope="The omitted POD frequencies are excluded from NN fit/normalization but contributed to the POD basis; they are not unseen full-pipeline cases.",
        numerical_protocol="Double-precision dense gelsd with condition cutoff; no normal equations or streamed QR replacement.")
    write(folder/"summary.json",summary)
    complete(folder/"manifest.json",fit_identity,[model_file,folder/"weights.npz"],run_identity=identity,basis_id=bid,**summary)
    log("train_complete",training_rows=len(x),validation_percent=validation["global_relative_L2_percent"],test_percent=test["global_relative_L2_percent"])

@njit(cache=True)
def nn_acc(z,u,pars):
    Wout,bout,scale,ii,jj,factor,eps,triples,cfactor,eps3=pars
    x=np.empty(len(z)+1);x[:-1]=z;x[-1]=u;xn=x/scale;n=len(x)
    h=np.empty(n+len(ii)+len(triples))
    for k in range(n):h[k]=np.tanh(.5*np.tanh(eps[0]*xn[k]))
    for k in range(len(ii)):
        h[n+k]=np.tanh(.5*np.tanh(eps[0]*(xn[int(ii[k])]+xn[int(jj[k])])*factor[k]+.5))
    for k in range(len(triples)):
        hh=xn[int(triples[k,0])]+xn[int(triples[k,1])]+xn[int(triples[k,2])]
        h[n+len(ii)+k]=np.tanh(.5*np.tanh(eps3[0]*hh*cfactor[k]))
    return Wout@h+bout

@njit(cache=True)
def drive(c,amp):return amp*.5*(1-np.cos(np.pi*min(c/30.,1.)))*np.sin(2*np.pi*c)

@njit(cache=True)
def integrate_cycle(z,cidx,amp,omega,spp,pars,qguard,vguard):
    r=len(z)//2;dt=2*np.pi/omega/spp
    saved=np.empty((129,2*r));saved[:]=np.nan;cursor=0
    for k in range(spp):
        c=cidx+k/spp
        if k%(spp//128)==0:saved[cursor]=z;cursor+=1
        k1=np.concatenate((z[r:],nn_acc(z,drive(c,amp),pars)))
        z2=z+.5*dt*k1;k2=np.concatenate((z2[r:],nn_acc(z2,drive(c+.5/spp,amp),pars)))
        z3=z+.5*dt*k2;k3=np.concatenate((z3[r:],nn_acc(z3,drive(c+.5/spp,amp),pars)))
        z4=z+dt*k3;k4=np.concatenate((z4[r:],nn_acc(z4,drive(c+1/spp,amp),pars)))
        z=z+dt*(k1+2*k2+2*k3+k4)/6
        if not np.all(np.isfinite(z)):return z,saved[:cursor],k+1,1
        if np.linalg.norm(z[:r])>qguard or np.linalg.norm(z[r:])>vguard:return z,saved[:cursor],k+1,2
    saved[cursor]=z
    return z,saved[:cursor+1],spp,0


def predict(cfg,output,identity):
    model_path=output/"training"/"model.json"
    training_manifest=output/"training"/"manifest.json"
    fitted=read(training_manifest)
    if fitted.get("run_identity")!=identity or not cache(training_manifest,fitted.get("identity"),[model_path,output/"training"/"weights.npz"]):
        raise ValueError("Fitted model identity or actual model/weights file hashes disagree.")
    model=read(model_path)
    if model.get("run_identity")!=identity:raise ValueError("Prediction model belongs to another configuration/source.")
    r=cfg["rank"]
    if model.get("rank")!=r or model.get("basis_id")!=sha(output/"pod"/"phi.npy"):raise ValueError("POD/model identity mismatch.")
    keys=("Wout","bout","scale","ii","jj","factor","eps","triples","cfactor","eps3")
    pars=tuple(np.ascontiguousarray(model[k],dtype=float) for k in keys)
    results=[]
    with threadpool_limits(limits=1,user_api="blas"):
        for amp,factor in cfg["prediction_cases"]:
            name=tag(amp,factor);folder=output/"prediction"/name;folder.mkdir(exist_ok=True)
            trajectory=folder/"trajectory.npz";rid=token(dict(run=identity,model=sha(model_path),amp=amp,factor=factor,cycles=cfg["prediction_cycles"]))
            previous=cache(folder/"manifest.json",rid,[trajectory])
            if previous:results.append(previous);continue
            z=np.zeros(2*r);blocks=[];times=[];start=time.time();stop=0;steps=0
            omega=cfg["omega_reference"]*factor
            for j in range(cfg["prediction_cycles"]):
                # The learned RHS alone advances q and qdot; no physical RHS,
                # mass-balance correction, reference trajectory, or interpolation.
                z,block,finished,stop=integrate_cycle(z,j,amp,omega,cfg["steps_per_period"],pars,
                    cfg["prediction_state_norm_guard"],cfg["prediction_state_norm_guard"])
                count=min(len(block),128);blocks.append(block[:count]);times.append(j+np.arange(count)/128)
                steps=j*cfg["steps_per_period"]+finished
                if stop:break
                if (j+1)%10==0:log("predict_case",case=name,cycles_completed=j+1)
            blocks.append(z[None,:]);times.append(np.array([steps/cfg["steps_per_period"]]))
            values=np.concatenate(blocks);sample_times=np.concatenate(times)
            if not np.isfinite(values).all():
                write(folder/"failure.json",dict(stop_code=int(stop),completed_steps=steps))
                raise ValueError("Nonfinite autonomous NN trajectory; failure preserved.")
            np.savez(trajectory,time_over_T=sample_times,q=values[:,:r],v=values[:,r:],last_state=z)
            record=dict(amplitude=amp,frequency_factor=factor,requested_cycles=cfg["prediction_cycles"],
                completed_steps=steps,completed_requested_horizon=stop==0,stop_code=int(stop),
                maximum_saved_q_norm=float(np.max(np.linalg.norm(values[:,:r],axis=1))),
                maximum_saved_qdot_norm=float(np.max(np.linalg.norm(values[:,r:],axis=1))),
                elapsed_seconds=time.time()-start,run_identity=identity,model_sha256=sha(model_path),
                inference="Autonomous zero-start RK4 of the NN acceleration only.",guard="Separate q and qdot Euclidean norm guards; no accuracy claim.")
            complete(folder/"manifest.json",rid,[trajectory],**record);results.append(read(folder/"manifest.json"))
            if stop:raise RuntimeError("NN state guard stopped integration; see preserved prediction manifest.")
    write(output/"prediction"/"summary.json",dict(status="complete",run_identity=identity,cases=results,
        profile=cfg["profile"],smoke_is_accuracy_reproduction=False if cfg["profile"]=="smoke" else None))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure,axes=plt.subplots(figsize=(7,4))
        for amp,factor in cfg["prediction_cases"]:
            with np.load(output/"prediction"/tag(amp,factor)/"trajectory.npz") as data:
                axes.plot(data["time_over_T"],np.linalg.norm(data["q"],axis=1),label=f"A={amp:g}, f/f0={factor:g}")
        axes.set_xlabel("Time / forcing period");axes.set_ylabel("POD pressure coefficient norm");axes.legend()
        figure.tight_layout();figure.savefig(output/"prediction"/"modal_pressure_norm.png",dpi=160);plt.close(figure)
    except ImportError:
        log("optional_plot_skipped",reason="Matplotlib not installed; numerical trajectory artifacts are complete.")
