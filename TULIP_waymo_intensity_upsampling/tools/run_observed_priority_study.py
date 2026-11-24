"""Explicit staged evaluation of observed-priority diagnostic rasters."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['retrieval','compare','dense-prepare','dense-pilot','dense-all'])
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args(); root=args.root.resolve()
    project=Path(__file__).resolve().parents[1]
    data=root/'database'; out=root/'retrieval'
    tools=project/'tools'
    def run(cmd):
        print(' '.join(map(str,cmd)),flush=True)
        subprocess.run(list(map(str,cmd)),check=True)
    assert (data/'completed.json').is_file(),'Build full database first'
    if args.action=='retrieval':
        if out.exists(): raise FileExistsError(f'Preserving existing {out}; use compare if complete')
        out.mkdir()
        cmd=[sys.executable,project.parent/'handover_shimizu/src/retrieval/evaluate.py',
            '--checkpoint',project.parent/'waymo_outputs/old32_retrieval/train_wdrive_paper80_b16/checkpoints/best.ckpt',
            '--pairs',data/'pairs.parquet','--db',data/'db.parquet',
            '--icp_csv','/mnt/w/32line/datasets/splits/cross/all/icp_startend_staticmaps.csv',
            '--invert_icp','false','--out_dir',out,'--name','observed_priority',
            '--backbone','resnet34','--clusters','64','--embed_dim','256',
            '--image_h','512','--image_w','864','--modality_stats',project/'configs/paper80_effective_modality_stats.json',
            '--device','cuda','--batch_size','16','--num_workers','2','--amp','true',
            '--write_failures','false','--write_failure_gallery','false']
        (out/'command.json').write_text(json.dumps(list(map(str,cmd)),indent=2))
        run(cmd)
    elif args.action=='compare':
        import pandas as pd
        rows=[]
        paths={v:project/f'outputs/paperlike_retrieval_{v}_507q/paperlike_{v}_507q_metrics.json' for v in 'ABC'}
        paths['observed_priority']=out/'observed_priority_metrics.json'
        for name,path in paths.items():
            d=json.loads(path.read_text()); assert d['strict']['Q']==507 and d['strict']['DB']==754
            rows.append(dict(condition=name,strict_R10=d['strict']['R@']['10'],
                soft2_R10=d['soft']['R_soft']['2.0']['R@']['10'],
                soft5_R10=d['soft']['R_soft']['5.0']['R@']['10'],median_rank=d['strict']['MedianRank']))
        print(pd.DataFrame(rows).to_string(index=False))
        # Check IDs/camera paths, not just equal dataset sizes.
        old=pd.read_parquet(project/'outputs/shimizu_protocol_507q_754db_abc/retrieval_manifests/B_pairs.parquet')
        new=pd.read_parquet(data/'pairs.parquet')
        for col in ['cam_png','pair_id','db_seg','q_seg','anchor_id']:
            assert old[col].astype(str).equals(new[col].astype(str)),col
        print('Query identity and order checked. Exploratory evaluation, not independent validation.')
    else:
        dense=root/'dense'
        if args.action=='dense-prepare':
            run([sys.executable,tools/'evaluate_retrieval_topk_pipeline.py','prepare','--root',dense,
                 '--topk',out/'observed_priority_topk_top50.jsonl','--k','10'])
        else:
            if args.action=='dense-all':
                assert (dense/'pilot10_pnp/integrated_summary.json').exists(),'Complete and inspect pilot10 first'
            run([sys.executable,tools/'evaluate_retrieval_topk_pipeline.py','run','--root',dense,
                 '--scope','pilot10' if args.action=='dense-pilot' else 'all507'])


if __name__=='__main__': main()
