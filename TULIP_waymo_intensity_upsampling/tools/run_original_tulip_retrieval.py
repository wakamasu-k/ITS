"""Retrain Retrieval on ORIGINAL TULIP B before observed-priority experiments.

TULIP weights and the 16->32 fusion rule remain fixed. Source datasets and
inherited code are not edited. Train/validation images are new B renders;
the exact original 507-query B test images are reused.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'tools'))
import run_observed8_localization as shared

CHECKPOINT=PROJECT/'outputs/waymo_16_to_32_baseline_to_moderate_s0_e20/checkpoint.pth'
TULIP_PY=PROJECT.parent/'.venv_tulip/bin/python'
TULIP_ROOT=PROJECT.parent/'TULIP_backup/tulip'
DEFAULT_LOFTR=PROJECT.parent/'waymo_outputs/old32_dense/loftr_retrain_stage1_paper_wdrive/stage1_final_gstep715958.pth'
BASE=PROJECT/'outputs/shimizu_protocol_507q_754db_abc'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def plan(root):
    d=shared.read(root/'plan.json')
    if d.get('experiment')!='original_tulip_B_retrieval': raise ValueError('Wrong experiment root')
    return d


def prepare(root):
    import pandas as pd
    if root.exists():
        d=plan(root)
        if digest(CHECKPOINT)!=d['tulip_sha256']: raise ValueError('TULIP checkpoint changed')
        print('Existing plan preserved:',root/'plan.json');return
    ts=shared.tables()
    # Keep the original B test paths and exactly the same query/anchor identities.
    bp=pd.read_parquet(BASE/'retrieval_manifests/B_pairs.parquet')
    bd=pd.read_parquet(BASE/'retrieval_manifests/B_db.parquet')
    for col in ('cam_png','pair_id','db_seg','q_seg','anchor_id'):
        if not ts['test_pairs'][col].astype(str).equals(bp[col].astype(str)):
            raise ValueError(f'Original B test differs: {col}')
    for col in ('pair_id','db_seg','anchor_id'):
        if not ts['test_db'][col].astype(str).equals(bd[col].astype(str)):
            raise ValueError(f'Original B DB order differs: {col}')
    ts['test_pairs']=bp;ts['test_db']=bd
    jobs=[]
    for role in ('train','val','test'):
        segments=sorted(set(ts['train'].segment_id)) if role=='train' else sorted(set(ts[role+'_db'].db_seg))
        for seg in segments:
            if role=='train':
                template=shared.ORIGINAL/'submaps/training'/seg/'submaps.json';sub='training'
            else:
                group=ts[role+'_db'];row=group[group.db_seg==seg].iloc[0]
                sub=str(row.db_subset)
                # C source is used only for the shared anchor layout, not pixels.
                template=shared.ORIGINAL/'submaps/cross/original_val_test_all/db'/sub/seg/'submaps.json'
            candidates=[p for s in ('training','validation','testing')
                        if (p:=shared.RAW/s/(seg+'.tfrecord')).is_file()]
            if len(candidates)!=1: raise FileNotFoundError(f'TFRecord missing/ambiguous: {seg}')
            if not template.is_file(): raise FileNotFoundError(template)
            if role!='train' and not template.with_name('anchors.csv').is_file():
                raise FileNotFoundError(template.with_name('anchors.csv'))
            jobs.append(dict(role=role,segment=seg,subset=sub,tfrecord=str(candidates[0]),
                 template=str(template),template_sha256=digest(template),reuse_existing=(role=='test')))
    checkpoint_hash=digest(CHECKPOINT)
    for seg in sorted(set(bd.db_seg)):
        folder=BASE/'maps/B_tulip32/training'/seg
        profile=shared.read(folder/'render_profile.json')
        original=Path(profile['checkpoint'])
        if not original.is_absolute(): original=PROJECT/original
        if original.resolve()!=CHECKPOINT.resolve(): raise ValueError('Original B used another TULIP checkpoint')
        if profile['range_scale_m']!=75 or profile['intensity_transform']!='log1p':
            raise ValueError('Original B preprocessing differs')
    with (PROJECT/'outputs/training_pairs_16_to_32_full/index.csv').open() as f:
        tulip_rows=list(csv.DictReader(f))
    tulip_train={r['q_seg'] for r in tulip_rows if r['dataset_role']=='train'}
    heldout=set(ts['val_pairs'].db_seg)|set(ts['val_pairs'].q_seg)|set(bp.db_seg)|set(bp.q_seg)
    overlap=sorted(tulip_train&heldout)
    if overlap: raise ValueError(f'TULIP training overlaps localization validation/test: {overlap}')
    root.mkdir(parents=True)
    shared.save(root/'plan.json',dict(experiment='original_tulip_B_retrieval',jobs=jobs,
        tulip_checkpoint=str(CHECKPOINT),tulip_sha256=checkpoint_hash,range_scale_m=75.,intensity_transform='log1p',
        observed_priority=False,selection='374 validation queries only; Soft(5m) R@10; earliest epoch on ties',
        test='507 queries / 754 original B anchors',tulip_train_localization_heldout_overlap=overlap,
        note='Retrieval adapts to original B. No TULIP retraining and no observed-priority composition.'))
    for name,table in ts.items(): table.to_parquet(root/(name+'_source.parquet'),index=False)
    print(json.dumps(dict(roles={r:sum(j['role']==r for j in jobs) for r in ('train','val','test')},
        new_maps=199,reuse_test_maps=5,train_pairs=len(ts['train']),validation_queries=374,test_queries=507,
        tulip_train_localization_heldout_overlap=0,output=str(root)),indent=2))


def stage_map(base,j,d):
    folder=base/'maps/B_tulip32'/j['subset']/j['segment']
    marker=folder/'original_B_completed.json'
    if marker.is_file():
        expected=dict(checkpoint_sha256=d['tulip_sha256'],source_tfrecord=j['tfrecord'],full_segment=True)
        if shared.read(marker)!=expected: raise ValueError(f'Map provenance differs: {folder}')
        stats=shared.read(folder/'stats.json')
        if stats['failed_frames'] or stats['processed_frames']<=0 or not (folder/'map_static.npz').is_file():
            raise ValueError(f'Incomplete map: {folder}')
        print('REUSE map:',folder,flush=True);return
    if (folder/'map_static.npz').exists():
        raise FileExistsError(f'Map exists without completion provenance; inspect before resuming: {folder}')
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TF_FORCE_GPU_ALLOW_GROWTH='true')
    shared.run([TULIP_PY,PROJECT/'tools/build_original_tulip_B_map.py',
       '--tfrecord',j['tfrecord'],'--subset',j['subset'],'--tulip-root',TULIP_ROOT,
       '--checkpoint',CHECKPOINT,'--output-root',base/'maps','--range-scale-m','75',
       '--intensity-transform','log1p','--map-variants','B','--device','cuda'],env=env)
    stats=shared.read(folder/'stats.json')
    if stats['failed_frames'] or stats['processed_frames']<=0: raise ValueError('Map frame failures')
    shared.save(marker,dict(checkpoint_sha256=d['tulip_sha256'],source_tfrecord=j['tfrecord'],full_segment=True))


def data(root,role,limit):
    d=plan(root)
    if digest(CHECKPOINT)!=d['tulip_sha256']: raise ValueError('TULIP checkpoint changed')
    jobs=[j for j in d['jobs'] if role=='all' or j['role']==role]
    if limit: jobs=jobs[:limit]
    cpu=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for j in jobs:
        if j['reuse_existing']:
            print('REUSE original B test images:',j['segment'],flush=True);continue
        base=root/'data'/j['role'];sub=j['subset'];seg=j['segment']
        template=Path(j['template']);target=base/'anchors'/sub/seg/'submaps.json'
        if digest(template)!=j['template_sha256']: raise ValueError('Anchor layout changed')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if target.read_bytes()!=template.read_bytes(): raise ValueError(f'Layout mismatch: {target}')
        else: shutil.copyfile(template,target)
        if j['role']=='val':
            src=template.with_name('anchors.csv');dest=target.with_name('anchors.csv')
            if dest.exists():
                if src.read_bytes()!=dest.read_bytes(): raise ValueError('Position table differs')
            else: shutil.copyfile(src,dest)
        stage_map(base,j,d)
        shared.run([sys.executable,shared.REF/'data_make/valid_test/render_anchor_intensity_txt_se3_fix.py',
            '--tfrecord',j['tfrecord'],'--subset',sub,'--maps-root',base/'maps/B_tulip32',
            '--submaps-root',base/'anchors','--crop-mode','none','--int-mode','global_map',
            '--point-size','2','--write-index','true','--overwrite','false'],env=cpu)
        if j['role']=='train':
            shared.run([sys.executable,shared.REF/'data_make/train/make_retrieval_shifted_ri_txt.py',
                '--tfrecord',j['tfrecord'],'--subset',sub,'--maps-root',base/'maps/B_tulip32',
                '--submaps-root',base/'anchors','--cam-gray-root',shared.ORIGINAL/'cam_gray',
                '--out-root',base/'retrieval_pairs','--num-samples-per-anchor','8','--seed','0',
                '--zero-ri-mode','reuse','--overwrite','false'],env=cpu)
        print('DONE:',j['role'],seg,flush=True)


def manifests(root):
    import pandas as pd
    d=plan(root);out=root/'manifests'
    if out.exists(): raise FileExistsError(out)
    for j in d['jobs']:
        if j['reuse_existing']: continue
        marker=root/'data'/j['role']/'maps/B_tulip32'/j['subset']/j['segment']/'original_B_completed.json'
        if not marker.is_file() or shared.read(marker)['checkpoint_sha256']!=d['tulip_sha256']:
            raise ValueError(f'Map not complete: {marker}')
    prepared={}
    for name in ('train','val_pairs','val_db','test_pairs','test_db'):
        table=pd.read_parquet(root/(name+'_source.parquet')).copy()
        if name=='train':
            table['anchor_png']=[str(root/'data/train/retrieval_pairs/training'/str(r.segment_id)/
               str(r.submap_id)/'ri_000.png') for r in table.itertuples()]
        elif name.startswith('val_'):
            table['anchor_png']=[str(root/'data/val/anchors'/str(r.db_subset)/str(r.db_seg)/
               str(r.anchor_id)/'anchor.png') for r in table.itertuples()]
        for path in set(table.anchor_png):
            p=Path(path)
            if not p.is_file(): raise FileNotFoundError(p)
            if name!='train':
                for f in ('anchor.json','anchor_index.npz'):
                    if not (p.parent/f).is_file(): raise FileNotFoundError(p.parent/f)
        if 'cam_png' in table:
            for path in set(table.cam_png):
                if not Path(path).is_file(): raise FileNotFoundError(path)
        prepared[name]=table
    out.mkdir()
    for name,table in prepared.items(): table.to_parquet(out/(name+'.parquet'),index=False)
    counts={name:len(t) for name,t in prepared.items()}
    shared.save(out/'completed.json',counts)
    print(json.dumps(counts,indent=2))


def check(root):
    """Check geometric validation before starting GPU training."""
    import numpy as np
    import pandas as pd
    import recheck_observed8_retrieval as checker
    mod=checker.module()
    pairs=pd.read_parquet(root/'manifests/val_pairs.parquet')
    db=pd.read_parquet(root/'manifests/val_db.parquet')
    if (len(pairs),len(db))!=(374,443): raise ValueError('Validation size differs')
    for path in pairs.cam_png:
        xy=mod._load_cam_xy_from_sidecar(path)
        if xy is None or not np.isfinite(xy).all(): raise ValueError(f'Query position missing: {path}')
    for seg,group in db.groupby('db_seg'):
        expected=Path(group.iloc[0].anchor_png).parent.parent/'anchors.csv'
        actual=mod._find_anchors_csv_from_anchor_png(str(group.iloc[0].anchor_png))
        if actual is None or Path(actual).resolve()!=expected.resolve():
            raise ValueError(f'anchors.csv missing/mislocated: {seg}')
    gt=mod._build_soft_gt_sets(pairs,db,icp_csv=str(shared.ICP),invert_icp=False,
                              thresholds_m=(1.,2.,5.,10.))
    if gt is None or any(not s for groups in gt.values() for s in groups):
        raise ValueError('Empty validation Soft GT')
    print(json.dumps(dict(validation_queries=374,validation_db=443,
        empty_gt={str(t):sum(not s for s in groups) for t,groups in gt.items()},
        soft5_candidate_count_median=float(np.median([len(s) for s in gt[5.]]))),indent=2))


def evaluate(root):
    choice=shared.read(root/'selected_retrieval.json')
    out=root/'retrieval_test';out.mkdir(exist_ok=False)
    shared.run([sys.executable,shared.REF/'retrieval/evaluate.py','--checkpoint',choice['checkpoint'],
        '--pairs',root/'manifests/test_pairs.parquet','--db',root/'manifests/test_db.parquet',
        '--icp_csv',shared.ICP,'--invert_icp','false','--out_dir',out,'--name','original_B_retrained',
        '--backbone','resnet34','--clusters','64','--embed_dim','256','--image_h','512','--image_w','864',
        '--modality_stats',root/'modality_stats.json','--device','cuda','--batch_size','16',
        '--num_workers','2','--amp','true','--write_failures','false','--write_failure_gallery','false'],out/'console.log')


def dense(root,action,checkpoint):
    spec=importlib.util.spec_from_file_location('original_B_topk',PROJECT/'tools/evaluate_retrieval_topk_pipeline.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    target=root/'dense'
    if action=='dense-prepare':
        if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
        mod.CKPT=str(checkpoint.resolve())
        topk=root/'retrieval_test/original_B_retrained_topk_top50.jsonl'
        mod.prepare(SimpleNamespace(root=target,topk=topk,k=10))
    else:
        mod.CKPT=shared.read(target/'protocol.json')['checkpoint']
        if action=='dense-all' and not (target/'pilot10_pnp/integrated_summary.json').is_file():
            raise RuntimeError('Run and inspect dense-pilot first')
        sys.argv=['evaluate_retrieval_topk_pipeline.py','run','--root',str(target),
                  '--scope','pilot10' if action=='dense-pilot' else 'all507']
        mod.main()


def status(root):
    d=plan(root)
    for role in ('train','val'):
        jobs=[j for j in d['jobs'] if j['role']==role]
        n=sum((root/'data'/role/'maps/B_tulip32'/j['subset']/j['segment']/'original_B_completed.json').is_file() for j in jobs)
        print(f'{role} B maps: {n}/{len(jobs)}')
    for name in ('manifests/completed.json','modality_stats.json','retrieval_train/logs/train_metrics.csv',
                 'selected_retrieval.json','retrieval_test/original_B_retrained_metrics.json',
                 'dense/all507_pnp/integrated_summary.json'):
        print(name,':','あり' if (root/name).is_file() else 'なし')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','data','manifests','stats','check','train','select','evaluate',
                                   'dense-prepare','dense-pilot','dense-all','status'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--role',choices=['train','val','test','all'],default='all')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--loftr-checkpoint',type=Path,default=DEFAULT_LOFTR)
    a=p.parse_args();root=a.root.resolve()
    if a.limit<0: p.error('--limit must be nonnegative')
    if a.action=='prepare': prepare(root)
    elif a.action=='data': data(root,a.role,a.limit)
    elif a.action=='stats': shared.stats(root)
    elif a.action=='train':
        check(root);shared.train(root)
    elif a.action=='select': shared.select(root)
    elif a.action.startswith('dense-'): dense(root,a.action,a.loftr_checkpoint)
    else: globals()[a.action](root)


if __name__=='__main__': main()
