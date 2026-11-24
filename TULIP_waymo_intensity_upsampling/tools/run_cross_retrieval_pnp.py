#!/usr/bin/env python3
"""Run reference LoFTR/PnP read-only, with all Top-5 candidates retained."""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=['A','B','C'],required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--scope',choices=['pilot10','all507'],default='pilot10')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args()
    out=args.run_root/f'{args.variant}_{args.scope}_pnp'
    out.mkdir(exist_ok=False)
    ref=Path('/home/wakamatsu/ITS/handover_shimizu/src/LoFTR')
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(ref/'src'))
    cmd=['/home/wakamatsu/ITS/.venv_tulip/bin/python','-u',str(ref/'tools/eval_cross_match_to_pose_zalign_multi_visall_v2.py'),
         '--manifest',str(args.run_root/f'{args.variant}_top5_{args.scope}.csv'),
         '--ckpt','/mnt/w/32line/experiment/loftr/train/run/2026_01_28/coarse_fine_bacbone/latest_model.pth',
         '--out-dir',str(out),'--device',args.device,'--amp-dtype','fp32',
         '--gt-type','all','--z-align','none','--icp-csv',str(args.run_root/'icp_db_to_q.csv'),
         '--icp-mat-cols',','.join(f'T{i}{j}' for i in range(4) for j in range(4))]
    import json
    (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    with (out/'execution.log').open('w') as log:
        subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(f'Completed {args.variant}: {out}')


if __name__=='__main__': main()
