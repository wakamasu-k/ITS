"""Repair position sidecars and reselect saved weights using validation only.

Original checkpoints, training logs, selected_retrieval.json and old test
results are preserved. Re-evaluations use FP32, like the inherited trainer's
validation, and a new output directory. No optimizer or backward pass is run.
"""
import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
REF = PROJECT.parent / 'handover_shimizu/src/retrieval'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def module():
    spec = importlib.util.spec_from_file_location('observed8_training_reference', REF/'train.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def repair(root):
    import numpy as np
    import pandas as pd
    jobs = [j for j in load(root/'plan.json')['jobs'] if j['role'] in ('val','test')]
    pending=[]; report=[]
    for j in jobs:
        source=Path(j['template']).parent
        target=root/'data'/j['role']/'anchors'/j['subset']/j['segment']
        if sha(source/'submaps.json') != sha(target/'submaps.json'):
            raise ValueError(f'Anchor layout differs: {target}')
        table=pd.read_csv(source/'anchors.csv')
        if not {'anchor_id','x','y'}.issubset(table): raise ValueError('Invalid anchors.csv')
        if table.anchor_id.duplicated().any(): raise ValueError('Duplicate anchor IDs')
        db=pd.read_parquet(root/f'manifests/{j["role"]}_db.parquet')
        db=db[db.db_seg==j['segment']]
        lookup=table.set_index('anchor_id')
        for row in db.itertuples():
            aid=int(row.anchor_id)
            a=load(source/str(aid)/'anchor.json')
            b=load(target/str(aid)/'anchor.json')
            for name in ('anchor_pose_T_wv','world_to_cam','intrinsic'):
                if not np.allclose(a[name],b[name],atol=1e-8,rtol=0):
                    raise ValueError(f'Pose/calibration mismatch: {target}/{aid}, {name}')
            pose=np.asarray(b['anchor_pose_T_wv']).reshape(4,4)
            if not np.allclose(pose[:2,3],lookup.loc[aid,['x','y']].to_numpy(dtype=float),atol=1e-6,rtol=0):
                raise ValueError(f'Position table mismatch: {target}/{aid}')
        src=source/'anchors.csv'; dst=target/'anchors.csv'
        if dst.exists() and dst.read_bytes()!=src.read_bytes():
            raise FileExistsError(f'Refusing to overwrite different position table: {dst}')
        pending.append((src,dst))
        report.append(dict(role=j['role'],segment=j['segment'],anchors_checked=len(db),
                           source=str(src),target=str(dst),sha256=sha(src)))
    # All poses are checked before adding any sidecar files.
    for src,dst in pending:
        if not dst.exists():
            with dst.open('xb') as f: f.write(src.read_bytes())
    print(json.dumps(dict(position_tables=len(report),anchors_checked=sum(x['anchors_checked'] for x in report),
                         note='Copied original positions; no images or poses changed',details=report),
                     ensure_ascii=False,indent=2))


def check(root, mod):
    import numpy as np
    import pandas as pd
    pairs=pd.read_parquet(root/'manifests/val_pairs.parquet')
    db=pd.read_parquet(root/'manifests/val_db.parquet')
    test=pd.read_parquet(root/'manifests/test_pairs.parquet')
    assert (len(pairs),len(db),len(test))==(374,443,507)
    assert not set(pairs.cam_png)&set(test.cam_png)
    assert not (set(pairs.q_seg)|set(pairs.db_seg)) & (set(test.q_seg)|set(test.db_seg))
    args=load(root/'retrieval_train/args.json')
    assert Path(args['val_pairs']).resolve()==(root/'manifests/val_pairs.parquet').resolve()
    assert Path(args['val_db']).resolve()==(root/'manifests/val_db.parquet').resolve()
    assert args['val_invert_icp'] is False
    for seg,group in db.groupby('db_seg'):
        found=mod._find_anchors_csv_from_anchor_png(str(group.iloc[0].anchor_png))
        if found is None: raise FileNotFoundError(f'anchors.csv is missing for {seg}; run repair')
        expected=Path(group.iloc[0].anchor_png).parent.parent/'anchors.csv'
        if Path(found).resolve()!=expected.resolve():
            raise ValueError(f'Unexpected position table: {found}')
    missing=[]
    for p in pairs.cam_png:
        xy=mod._load_cam_xy_from_sidecar(p)
        if xy is None or not np.isfinite(xy).all(): missing.append(p)
    if missing: raise ValueError(f'Query poses missing ({len(missing)}); first: {missing[0]}')
    gt=mod._build_soft_gt_sets(pairs,db,icp_csv=args['val_icp_csv'],invert_icp=False,
                               thresholds_m=(1.,2.,5.,10.))
    if gt is None: raise ValueError('No ICP transforms / Soft GT')
    gallery={str(p):i for i,p in enumerate(db.anchor_png)}
    report=dict(validation_queries=374,validation_db=443,query_positions_readable=374,
                validation_test_segment_overlap=0,thresholds={})
    for threshold,sets in gt.items():
        counts=[len(s) for s in sets]
        for i,row in pairs.iterrows():
            if not sets[i] or gallery[str(row.anchor_png)] not in sets[i]:
                raise ValueError(f'Empty/misaligned Soft GT at query {i}')
        report['thresholds'][str(threshold)]=dict(empty_gt=sum(n==0 for n in counts),
              candidate_count_min=min(counts),candidate_count_median=float(np.median(counts)),
              candidate_count_max=max(counts))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return pairs,db,args,report


def fingerprint(root):
    paths=[root/'manifests/val_pairs.parquet',root/'manifests/val_db.parquet',
           root/'modality_stats.json',root/'retrieval_train/args.json',REF/'train.py']
    args=load(root/'retrieval_train/args.json');paths.append(Path(args['val_icp_csv']))
    for j in load(root/'plan.json')['jobs']:
        if j['role']=='val':
            paths.append(root/'data/val/anchors'/j['subset']/j['segment']/'anchors.csv')
    return {str(p):sha(p) for p in paths}


def protocol(root,out,batch):
    state=dict(input_hashes=fingerprint(root),precision='FP32',batch_size=batch,
               selection_metric='Soft_t5m_R@10',selection_queries=374,selection_db=443)
    out.mkdir(parents=True,exist_ok=True)
    path=out/'protocol.json'
    if path.exists():
        if load(path)!=state: raise ValueError('Recheck settings changed; use a new --output-dir')
    else: save(path,state)


def epochs_requested(text):
    if text=='all': return list(range(1,81))
    nums=[int(x) for x in text.split(',')]
    if not nums or any(n<1 or n>80 for n in nums): raise ValueError('--epochs: all or e.g. 1,80')
    return sorted(set(nums))


def validate(root,out,epochs,batch):
    import torch
    mod=module()
    pairs,db,args,report=check(root,mod)
    protocol(root,out,batch)
    if not (out/'position_check.json').exists(): save(out/'position_check.json',report)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; run in the training WSL environment')
    print('GPU:',torch.cuda.get_device_name(0),flush=True)
    stats=load(root/'modality_stats.json')
    values=[float(stats[k]) for k in ('cam_mean','cam_std','anc_mean','anc_std')]
    if not all(math.isfinite(v) for v in values) or values[1]<=0 or values[3]<=0:
        raise ValueError('Invalid training normalization')
    # Same architecture and FP32 validation as training. ImageNet initialization
    # is overwritten completely by strict checkpoint loading before inference.
    model=None
    for epoch in epochs_requested(epochs):
        checkpoint=root/f'retrieval_train/checkpoints/epoch_{epoch:03d}.ckpt'
        output=out/f'epoch_{epoch:03d}.json'
        if output.exists():
            old=load(output);st=checkpoint.stat()
            if old['checkpoint_size']!=st.st_size or old['checkpoint_mtime_ns']!=st.st_mtime_ns:
                raise ValueError(f'Checkpoint changed after evaluation: {checkpoint}')
            print('REUSE',output,flush=True);continue
        if model is None:
            model=mod.DualEncoder(backbone=args['backbone'],out_dim=args['embed_dim'],
                                  clusters=args['clusters'],freeze_stages=args['freeze_stages']).cuda()
        ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if int(ckpt['epoch'])!=epoch: raise ValueError('Checkpoint epoch mismatch')
        for key in ('backbone','clusters','embed_dim','image_h','image_w'):
            if ckpt['args'][key]!=args[key]: raise ValueError(f'Checkpoint configuration differs: {key}')
        model.load_state_dict(ckpt['state_dict'],strict=True)
        del ckpt;gc.collect()
        print(f'EVALUATING epoch {epoch}/80: 374 validation queries, 443 DB (FP32)',flush=True)
        metrics=mod.eval_retrieval_metrics(model,pairs,db,torch.device('cuda'),
            image_size=(args['image_h'],args['image_w']),batch_size=batch,
            icp_csv=args['val_icp_csv'],invert_icp=False,
            cam_mean=values[0],cam_std=values[1],anc_mean=values[2],anc_std=values[3])
        for t in (1,2,5,10):
            if metrics[f'Soft_t{t}m_R@10']+1e-12<metrics['Strict_R@10']:
                raise ValueError('Soft recall below Strict despite GT inclusion; inspect evaluation')
        if not all(math.isfinite(float(v)) for v in metrics.values()): raise ValueError('Nonfinite metrics')
        st=checkpoint.stat()
        save(output,dict(epoch=epoch,checkpoint=str(checkpoint),checkpoint_size=st.st_size,
             checkpoint_mtime_ns=st.st_mtime_ns,metrics=metrics))
        print(json.dumps(dict(epoch=epoch,strict_R10=metrics['Strict_R@10'],
              soft5_R10=metrics['Soft_t5m_R@10'])),flush=True)


def select(root,out):
    p=load(out/'protocol.json')
    if p['input_hashes']!=fingerprint(root): raise ValueError('Validation input changed')
    rows=[load(out/f'epoch_{e:03d}.json') for e in range(1,81)]
    if {r['epoch'] for r in rows}!=set(range(1,81)): raise ValueError('Expected 80 distinct epochs')
    for r in rows:
        expected=root/f'retrieval_train/checkpoints/epoch_{r["epoch"]:03d}.ckpt'
        st=expected.stat()
        if (Path(r['checkpoint']).resolve()!=expected.resolve()
                or r['checkpoint_size']!=st.st_size or r['checkpoint_mtime_ns']!=st.st_mtime_ns):
            raise ValueError(f'Checkpoint differs from evaluated file: {expected}')
    if max(r['metrics']['Soft_t5m_R@10'] for r in rows)<=0:
        raise ValueError('All Soft recalls are zero; refusing checkpoint selection')
    best=min(rows,key=lambda r:(-r['metrics']['Soft_t5m_R@10'],r['epoch']))
    choice=dict(epoch=best['epoch'],checkpoint=best['checkpoint'],metric='Soft_t5m_R@10',
                value=best['metrics']['Soft_t5m_R@10'],selection_split='374 validation queries; 443 DB',
                validation_metrics=best['metrics'])
    dest=out/'selected_retrieval.json'
    if dest.exists():
        if load(dest)!=choice: raise ValueError('Different selection already exists')
    else: save(dest,choice)
    table=out/'validation_metrics.csv'
    if not table.exists():
        with table.open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['epoch',*best['metrics']]);w.writeheader()
            w.writerows(dict(epoch=r['epoch'],**r['metrics']) for r in rows)
    print(json.dumps(choice,ensure_ascii=False,indent=2))


def test(root,out):
    choice=load(out/'selected_retrieval.json')
    if load(out/'protocol.json')['input_hashes']!=fingerprint(root): raise ValueError('Inputs changed')
    dest=out/'test507'
    if (dest/'observed8_reselected_metrics.json').is_file():
        print('Saved test result:',dest/'observed8_reselected_metrics.json');return
    dest.mkdir(exist_ok=False)
    a=load(root/'retrieval_train/args.json')
    cmd=list(map(str,[sys.executable,REF/'evaluate.py','--checkpoint',choice['checkpoint'],
        '--pairs',root/'manifests/test_pairs.parquet','--db',root/'manifests/test_db.parquet',
        '--icp_csv',a['val_icp_csv'],'--invert_icp','false','--out_dir',dest,'--name','observed8_reselected',
        '--backbone',a['backbone'],'--clusters',a['clusters'],'--embed_dim',a['embed_dim'],
        '--freeze_stages',a['freeze_stages'],'--image_h',a['image_h'],'--image_w',a['image_w'],
        '--modality_stats',root/'modality_stats.json','--device','cuda','--batch_size','16',
        '--num_workers','2','--amp','true','--write_failures','false','--write_failure_gallery','false']))
    save(dest/'command.json',cmd)
    subprocess.run(cmd,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['repair','check','validate','select','test'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--epochs',default='all',help='all or comma-separated epochs, e.g. 1,80')
    parser.add_argument('--batch-size',type=int,default=32)
    args=parser.parse_args();root=args.root.resolve()
    out=(args.output_dir or root/'validation_recheck_20261029').resolve()
    if args.batch_size<1: parser.error('--batch-size must be positive')
    if args.action=='repair': repair(root)
    elif args.action=='check': check(root,module())
    elif args.action=='validate': validate(root,out,args.epochs,args.batch_size)
    elif args.action=='select': select(root,out)
    else: test(root,out)


if __name__=='__main__': main()
