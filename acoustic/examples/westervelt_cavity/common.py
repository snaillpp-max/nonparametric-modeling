"""Portable files, current-run identities, and resource checks."""
from pathlib import Path
import ctypes,hashlib,json,os,shutil,time

GIB=1024**3
def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024**2),b""):h.update(block)
    return h.hexdigest()
def token(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()
def tag(amp,factor):
    return f"u{amp:.3f}_f{factor:.4f}"
def cases(cfg,which):
    return [(float(a),float(f)) for a in cfg[which+"_amplitudes"] for f in cfg[which+"_frequency_factors"]]
def log(stage,**extra):
    print(json.dumps(dict(stage=stage,**extra)),flush=True)
def available_memory():
    if os.name=="nt":
        class Memory(ctypes.Structure):
            _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ("total","available","page_total","page_available","virtual_total","virtual_available","extended")]
        state=Memory();state.length=ctypes.sizeof(state)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):return int(state.available)
    if hasattr(os,"sysconf"):
        try:return int(os.sysconf("SC_AVPHYS_PAGES")*os.sysconf("SC_PAGE_SIZE"))
        except (ValueError,OSError):pass
    return None
def cache(path,identity,files):
    path=Path(path)
    if not path.exists():return None
    old=read(path)
    if old.get("status")!="complete" or old.get("identity")!=identity:return None
    if not all(Path(p).is_file() and old.get("files",{}).get(Path(p).name)==sha(p) for p in files):return None
    return old
def complete(path,identity,files,**values):
    write(path,dict(status="complete",identity=identity,files={Path(p).name:sha(p) for p in files},**values))
def initialize(output,cfg):
    if cfg["ramp_cycles"]!=30:
        raise ValueError("This original-kernel reproduction requires ramp_cycles=30.")
    if cfg["perturbation_levels"]!=17 or cfg["perturbation_min"]!=-.2 or cfg["perturbation_max"]!=.2:
        raise ValueError("This original augmentation protocol requires 17 levels in [-0.2, 0.2].")
    if cfg["steps_per_period"]<128 or cfg["steps_per_period"]%128:
        raise ValueError("steps_per_period must be at least 128 and divisible by 128.")
    if cfg["nx"]%4 or cfg["ny"]%16:
        raise ValueError("Geometry requires nx divisible by 4 and ny divisible by 16.")
    if cfg["augmentation_rows"]!=cfg["perturbed_base_rows"]*17+cfg["base_rows"]-cfg["perturbed_base_rows"]:
        raise ValueError("augmentation_rows is inconsistent with perturbation/base counts.")
    if not set(cases(cfg,"nn")).issubset(set(cases(cfg,"pod"))):
        raise ValueError("NN cases must be a subset of the POD parameter cases.")
    output=Path(output).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    for name in ("cache","tmp","generated","pod","exact","augmented","training","prediction"):
        (output/name).mkdir(exist_ok=True)
    source_files=sorted(Path(__file__).parent.glob("*.py"))
    source_hashes={p.name:sha(p) for p in source_files}
    identity=token(dict(configuration=cfg,source_sha256=source_hashes))
    manifest=output/"run_config.json"
    if manifest.exists() and read(manifest)["identity"]!=identity:
        raise ValueError("Output directory has another configuration; choose a new output directory.")
    write(manifest,dict(identity=identity,configuration=cfg,source_sha256=source_hashes))
    for name in ("TMP","TEMP"):os.environ[name]=str(output/"tmp")
    os.environ["NUMBA_CACHE_DIR"]=str(output/"cache"/"numba")
    os.environ["MPLCONFIGDIR"]=str(output/"cache"/"matplotlib")
    os.environ["OPENBLAS_NUM_THREADS"]=str(cfg["blas_threads"])
    os.environ["OMP_NUM_THREADS"]=str(cfg["fem_threads"])
    return output,identity
def resource_report(cfg,output):
    nodes=(cfg["nx"]+1)*(cfg["ny"]//2+1)-(cfg["nx"]//4)*round(.35*cfg["ny"]/.8)
    ncases=len(cases(cfg,"pod"));r=cfg["rank"];n=2*r+1
    pair_coordinates=2*min(r,cfg["pair_rank"])+1
    pairs=pair_coordinates*(pair_coordinates+1)//2
    cubic_coordinates=2*min(r,cfg["cubic_rank"])+1
    triples=cubic_coordinates*(cubic_coordinates+1)*(cubic_coordinates+2)//6
    width=n+pairs+triples
    rows=len(cases(cfg,"nn"))*(cfg["selected_perturbation_rows"]+2*r*17)
    design=rows*(width+1)*8
    snapshot=ncases*144*nodes*4
    weighted=ncases*144*nodes*8
    gram=(ncases*144)**2*8
    training_ram=2*design+2*rows*r*8+4*GIB
    disk_estimate=snapshot+weighted+gram+design+ncases*nodes*16+len(cases(cfg,"nn"))*(cfg["augmentation_rows"]+34*r)*(n+r)*4
    report=dict(estimate=True,estimated_nodes=nodes,design_shape=[rows,width+1],design_bytes=design,
        training_available_memory_bytes=available_memory(),training_required_memory_estimate_bytes=training_ram,
        retained_storage_estimate_bytes=disk_estimate,disk_free_bytes=shutil.disk_usage(output).free,
        duration_estimate=None,duration_note="No runtime is predicted without measuring this exact configuration.",
        storage_note="Estimate includes pressure snapshots, retained weighted POD scratch, Gram matrix, and fitting scratch; filesystem overhead and interrupted-run copies need additional space.")
    write(Path(output)/"resource_preflight.json",report)
    log("resource_preflight",**report)
    if report["disk_free_bytes"]<disk_estimate*1.2:raise MemoryError("Insufficient disk space for estimated retained outputs and 20% reserve.")
    return report
