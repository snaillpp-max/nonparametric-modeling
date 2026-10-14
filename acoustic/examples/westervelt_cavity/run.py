"""Four independent stages: generate -> POD/augment -> train -> predict."""
import argparse,os,sys
from pathlib import Path
from common import read,initialize,resource_report,log
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",choices=("smoke","paper"),default="paper")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=("all","generate","pod","train","predict"),default="all")
    parser.add_argument("--config",type=Path,default=Path(__file__).with_name("config.json"))
    args=parser.parse_args()
    cfg=read(args.config)[args.profile]
    if os.environ.get("ACOUSTIC_MAX_THREADS"):
        cap=int(os.environ["ACOUSTIC_MAX_THREADS"])
        if cap<1:raise ValueError("ACOUSTIC_MAX_THREADS must be a positive integer.")
        cfg=dict(cfg)
        for key in ("blas_threads","fem_threads","augmentation_threads"):cfg[key]=min(cfg[key],cap)
    output,identity=initialize(args.output,cfg)
    resource_report(cfg,output)
    if cfg["profile"]=="smoke":log("smoke_notice",meaning="Coarse, short workflow validation only; does not reproduce paper accuracy or timings.")
    if args.stage in ("all","generate"):
        from dataset import generate
        generate(cfg,output,identity)
    if args.stage in ("all","pod"):
        from dataset import pod_and_augment
        pod_and_augment(cfg,output,identity)
    if args.stage in ("all","train"):
        from network import train
        train(cfg,output,identity)
    if args.stage in ("all","predict"):
        from network import predict
        predict(cfg,output,identity)
    log("requested_stages_complete",profile=cfg["profile"],requested_stage=args.stage,output=str(output))
if __name__=="__main__":main()
