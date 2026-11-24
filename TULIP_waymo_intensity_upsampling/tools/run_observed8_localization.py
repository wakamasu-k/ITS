"""Staged, isolated observed-8-line localization data and Retrieval experiment.

Uses the existing training cameras and anchor identities. Validation excludes
the five 507-query evaluation pairs. The test set never selects checkpoints.
PnP is geometric; it has no trainable weights. LoFTR training requires new
8-line geometric correspondence labels, not the old 32-line manifest.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
ITS = PROJECT.parent
REF = ITS / 'handover_shimizu/src'
RAW = Path('/mnt/w/waymo_tf/perception_v1.4.3/individual_files')
ORIGINAL = Path('/mnt/w/32line/datasets')
OLD = ITS / 'waymo_outputs/old32_retrieval'
ICP = ORIGINAL / 'splits/cross/all/icp_startend_staticmaps.csv'


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write('\n')


def run(cmd, log=None, env=None):
    cmd = list(map(str, cmd))
    print('RUN:', ' '.join(cmd), flush=True)
    if log is None:
        subprocess.run(cmd, check=True, env=env)
    else:
        with log.open('x') as f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env=env)
            for line in proc.stdout:
                print(line, end='', flush=True)
                f.write(line)
            if proc.wait():
                raise RuntimeError(f'Command failed; preserved log: {log}')


def tables():
    import pandas as pd
    train = pd.read_parquet(OLD / 'training_wdrive.parquet')
    combined = pd.read_parquet(OLD / 'val4_test4_all_pairs_wdrive.parquet')
    combined_db = pd.read_parquet(OLD / 'val4_test4_all_db_wdrive.parquet')
    base = PROJECT / 'outputs/shimizu_protocol_507q_754db_abc/retrieval_manifests'
    test = pd.read_parquet(base / 'C_pairs.parquet')
    test_db = pd.read_parquet(base / 'C_db.parquet')
    val = combined[~combined.pair_id.isin(set(test.pair_id))].copy()
    val_db = combined_db[combined_db.pair_id.isin(set(val.pair_id))].copy()
    assert (len(train), len(val), len(val_db), len(test), len(test_db)) == (28772,374,443,507,754)
    sets = [set(train.segment_id)] + [set(t.db_seg) | set(t.q_seg) for t in (val,test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), 'Segment overlap'
    return dict(train=train, val_pairs=val.reset_index(drop=True), val_db=val_db.reset_index(drop=True),
                test_pairs=test.reset_index(drop=True), test_db=test_db.reset_index(drop=True))


def prepare(root, lines=8):
    ts = tables()
    jobs = []
    for role in ('train','val','test'):
        if role == 'train':
            segments = sorted(set(ts['train'].segment_id))
        else:
            segments = sorted(set(ts[role+'_db'].db_seg))
        for seg in segments:
            if role == 'train':
                template = ORIGINAL / 'submaps/training' / seg / 'submaps.json'
                subset = 'training'
            else:
                rows = ts[role+'_db']; row = rows[rows.db_seg == seg].iloc[0]
                template = Path(row.anchor_png).parent.parent / 'submaps.json'
                subset = str(row.db_subset)
            # Source subsets and roles are different concepts.
            candidates = [p for sub in ('training','validation','testing')
                          if (p := RAW / sub / (seg+'.tfrecord')).is_file()]
            if len(candidates) != 1:
                raise FileNotFoundError(f'Expected one TFRecord for {seg}, found {candidates}')
            if not template.is_file():
                raise FileNotFoundError(template)
            jobs.append(dict(role=role, segment=seg, subset=subset, tfrecord=str(candidates[0]),
                             template=str(template),
                             template_sha256=hashlib.sha256(template.read_bytes()).hexdigest()))
    root.mkdir(parents=True, exist_ok=False)
    save(root/'plan.json', dict(native_rings=list(range(0,64,64//lines)), jobs=jobs,
        validation_queries=374, validation_db=443, test_queries=507, test_db=754,
        selection='validation-only Soft(5m) R@10; earliest epoch on ties',
        note=f'Observed {lines}-line experiment using the inherited method'))
    for name, table in ts.items():
        table.to_parquet(root/(name+'_source.parquet'), index=False)
    missing = REF/'LoFTR/tools/make_loftr_matches_gt_step2a_paper_final_fixed_cam_v6.py'
    save(root/'dependencies.json', dict(loftr_gt_v6=str(missing), loftr_gt_v6_exists=missing.is_file(),
         explanation='v7 imports v6. New 8-line GT generation needs this dependency or a validated replacement.'))
    print(json.dumps(dict(roles={r:sum(j['role']==r for j in jobs) for r in ('train','val','test')},
          train_rows=len(ts['train']), val_queries=374, test_queries=507,
          loftr_gt_dependency_present=missing.is_file(), output=str(root)),indent=2))


def data(root, role, limit, lines=8):
    jobs = [j for j in read(root/'plan.json')['jobs'] if role == 'all' or j['role'] == role]
    if limit: jobs = jobs[:limit]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='4')
    for j in jobs:
        base = root/'data'/j['role']; seg=j['segment']; sub=j['subset']
        target = base/'anchors'/sub/seg/'submaps.json'
        target.parent.mkdir(parents=True,exist_ok=True)
        template = Path(j['template'])
        if hashlib.sha256(template.read_bytes()).hexdigest() != j['template_sha256']:
            raise RuntimeError(f'Source template changed: {template}')
        if target.exists():
            if target.read_bytes() != template.read_bytes(): raise RuntimeError(f'Template mismatch: {target}')
        else: shutil.copyfile(template,target)
        # The inherited trainer discovers XY positions via anchors.csv beside
        # submaps.json; it does not read the DB parquet's position columns.
        if j['role'] in ('val','test'):
            source_positions=template.with_name('anchors.csv')
            target_positions=target.with_name('anchors.csv')
            if not source_positions.is_file(): raise FileNotFoundError(source_positions)
            if target_positions.exists():
                if target_positions.read_bytes()!=source_positions.read_bytes():
                    raise ValueError(f'Position table differs: {target_positions}')
            else: shutil.copyfile(source_positions,target_positions)
        builder = (PROJECT/'tools/make_observed8_static_map.py' if lines == 8 else
                   REF/'data_make/valid_test/make_static_map_txt.py')
        ring_args = [] if lines == 8 else ['--lidar-lines', str(lines), '--ring-keep-even', 'true']
        run([sys.executable, builder,
             '--tfrecord',j['tfrecord'],'--subset',sub,'--out-root',base/'maps',
             '--bbox-vehicle-y-scale','2.5','--overwrite','false',*ring_args],env=env)
        run([sys.executable,REF/'data_make/valid_test/render_anchor_intensity_txt_se3_fix.py',
             '--tfrecord',j['tfrecord'],'--subset',sub,'--maps-root',base/'maps',
             '--submaps-root',base/'anchors','--crop-mode','none','--int-mode','global_map',
             '--point-size','2','--write-index','true','--overwrite','false'],env=env)
        if j['role'] == 'train':
            run([sys.executable,REF/'data_make/train/make_retrieval_shifted_ri_txt.py',
                 '--tfrecord',j['tfrecord'],'--subset',sub,'--maps-root',base/'maps',
                 '--submaps-root',base/'anchors','--cam-gray-root',ORIGINAL/'cam_gray',
                 '--out-root',base/'retrieval_pairs','--num-samples-per-anchor','8',
                 '--seed','0','--overwrite','false'],env=env)
        print('DONE segment:',j['role'],seg,flush=True)


def manifests(root, test_only=False, lines=8):
    import pandas as pd
    out=root/('test_manifests' if test_only else 'manifests')
    if out.exists(): raise FileExistsError(out)
    # Require successful 8-line provenance for every segment before training.
    for j in read(root/'plan.json')['jobs']:
        if test_only and j['role']!='test': continue
        folder=root/'data'/j['role']/'maps'/j['subset']/j['segment']
        if lines == 8:
            p=folder/'observed8_provenance.json'
            complete=(p.is_file() and read(p)['native_ring_ids'] == list(range(0,64,8))
                      and read(p).get('max_frames') is None)
        else:
            p=folder/'render_profile.json'
            profile=read(p) if p.is_file() else {}
            complete=(profile.get('lidar_lines') == lines and profile.get('ring_keep_even') is True
                      and (folder/'map_static.npz').is_file() and (folder/'stats.json').is_file()
                      and read(folder/'stats.json')['frames'] > 0)
        if not complete:
            raise RuntimeError(f'{lines}-line map not complete: {p}')
    report={}; prepared={}
    names=('test_pairs','test_db') if test_only else ('train','val_pairs','val_db','test_pairs','test_db')
    for name in names:
        df=pd.read_parquet(root/(name+'_source.parquet'))
        mapped=[]
        for r in df.itertuples():
            if name=='train':
                p=root/'data/train/retrieval_pairs/training'/str(r.segment_id)/str(r.submap_id)/'ri_000.png'
            else:
                role=name.split('_')[0]
                p=root/'data'/role/'anchors'/str(r.db_subset)/str(r.db_seg)/str(r.anchor_id)/'anchor.png'
                for f in ('anchor.png','anchor.json','anchor_index.npz'):
                    if not (p.parent/f).is_file(): raise FileNotFoundError(p.parent/f)
            if not p.is_file(): raise FileNotFoundError(p)
            mapped.append(str(p))
        df['anchor_png']=mapped
        if 'cam_png' in df:
            absent=[p for p in set(df.cam_png) if not Path(p).is_file()]
            if absent: raise FileNotFoundError(f'Missing cameras: {absent[:3]}')
        prepared[name]=df
        report[name]=len(df)
    out.mkdir()
    for name,df in prepared.items(): df.to_parquet(out/(name+'.parquet'),index=False)
    save(out/'completed.json',report)
    print(json.dumps(report,indent=2))


def stats(root):
    if not (root/'manifests/completed.json').is_file(): raise RuntimeError('Build manifests first')
    if (root/'modality_stats.json').exists(): raise FileExistsError(root/'modality_stats.json')
    run([sys.executable,REF/'retrieval/compute_modality_stats_from_pairs.py',
         '--pairs',root/'manifests/train.parquet','--out-json',root/'modality_stats.json',
         '--only-label1','--anchor-mode','all_ri','--ri-count','8','--dedup-anchors',
         '--resize-h','512','--resize-w','864'])


def train(root):
    if not (root/'manifests/completed.json').is_file(): raise RuntimeError('Build manifests first')
    d=read(root/'modality_stats.json')
    # These flat keys are written by the inherited statistics tool and accepted
    # by the trainer. Require all four, instead of its silent fallback behavior.
    values=[float(d[k]) for k in ('cam_mean','cam_std','anc_mean','anc_std')]
    if not all(math.isfinite(x) for x in values) or values[1]<=0 or values[3]<=0:
        raise ValueError('Invalid modality statistics')
    if d['cam_missing'] or d['anc_missing']: raise ValueError('Missing training images in statistics')
    dest=root/'retrieval_train'; dest.mkdir(exist_ok=False)
    cmd=[sys.executable,REF/'retrieval/train.py','--pairs',root/'manifests/train.parquet',
         '--out-dir',dest,'--epochs','80','--batch-size','16','--lr','0.0003','--wd','0.0001',
         '--backbone','resnet34','--clusters','64','--embed-dim','256','--freeze-stages','2',
         '--image-h','512','--image-w','864','--seed','0','--device','cuda','--temperature','0.12',
         '--val-pairs',root/'manifests/val_pairs.parquet','--val-db',root/'manifests/val_db.parquet',
         '--val-icp-csv',ICP,'--val-invert-icp','false','--val1-name','validation_only',
         '--best-on','val1','--best-by','R@10','--num-workers','4','--amp','--scheduler','cosine',
         '--grad-clip','1.0','--patience','0','--jitter','0.2','--train-ri-mode','epoch_cycle',
         '--train-ri-count','8','--train-ri-seed','0','--modality-stats',root/'modality_stats.json',
         '--save-every-epochs','1']
    save(dest/'command.json',list(map(str,cmd)))
    run(cmd,dest/'console.log')


def select(root):
    import pandas as pd
    df=pd.read_csv(root/'retrieval_train/logs/train_metrics.csv')
    if set(df.epoch.astype(int)) != set(range(1,81)):
        raise RuntimeError('Expected all 80 epochs; inspect interrupted training')
    metric='val1/Soft_t5m_R@10'
    vals=pd.to_numeric(df[metric],errors='coerce')
    if vals.isna().any(): raise ValueError('Missing validation metric')
    if (vals==0).all():
        raise ValueError('All Soft validation values are zero. Use recheck_observed8_retrieval.py '
                         'to repair position sidecars and re-evaluate saved checkpoints.')
    row=df.assign(selection_value=vals).sort_values(['selection_value','epoch'],ascending=[False,True]).iloc[0]
    ckpt=root/f'retrieval_train/checkpoints/epoch_{int(row.epoch):03d}.ckpt'
    if not ckpt.is_file(): raise FileNotFoundError(ckpt)
    save(root/'selected_retrieval.json',dict(checkpoint=str(ckpt),epoch=int(row.epoch),
          metric=metric,value=float(row.selection_value),selection_split='374 validation queries only'))
    print(read(root/'selected_retrieval.json'))


def evaluate(root, lines=8):
    ckpt=read(root/'selected_retrieval.json')['checkpoint']
    out=root/'retrieval_test';out.mkdir(exist_ok=False)
    run([sys.executable,REF/'retrieval/evaluate.py','--checkpoint',ckpt,
         '--pairs',root/'manifests/test_pairs.parquet','--db',root/'manifests/test_db.parquet',
         '--icp_csv',ICP,'--invert_icp','false','--out_dir',out,'--name',f'observed{lines}',
         '--backbone','resnet34','--clusters','64','--embed_dim','256','--image_h','512','--image_w','864',
         '--modality_stats',root/'modality_stats.json','--device','cuda','--batch_size','16',
         '--num_workers','2','--amp','true','--write_failures','false','--write_failure_gallery','false'],out/'console.log')


def evaluate_fixed(root):
    """Measure the effect of 8-line input under the previous fixed weights."""
    data=root/'test_manifests'
    if not (data/'completed.json').is_file(): raise RuntimeError('Run test-manifests first')
    out=root/'retrieval_fixed';out.mkdir(exist_ok=False)
    run([sys.executable,REF/'retrieval/evaluate.py',
         '--checkpoint',OLD/'train_wdrive_paper80_b16/checkpoints/best.ckpt',
         '--pairs',data/'test_pairs.parquet','--db',data/'test_db.parquet',
         '--icp_csv',ICP,'--invert_icp','false','--out_dir',out,'--name','observed8_fixed',
         '--backbone','resnet34','--clusters','64','--embed_dim','256','--image_h','512','--image_w','864',
         '--modality_stats',PROJECT/'configs/paper80_effective_modality_stats.json',
         '--device','cuda','--batch_size','16','--num_workers','2','--amp','true',
         '--write_failures','false','--write_failure_gallery','false'],out/'console.log')


def dense(root, action, checkpoint, retrieval_source):
    """Reuse the existing Top-K pipeline with an explicit LoFTR checkpoint."""
    import importlib.util
    from types import SimpleNamespace
    spec=importlib.util.spec_from_file_location('observed8_topk',PROJECT/'tools/evaluate_retrieval_topk_pipeline.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    target=root/('dense_fixed' if retrieval_source=='fixed' else 'dense')
    if action=='dense-prepare':
        if checkpoint is None or not checkpoint.is_file():
            raise ValueError('Supply --loftr-checkpoint: an 8-line trained weight or an explicitly labeled baseline')
        topk=root/('retrieval_fixed/observed8_fixed_topk_top50.jsonl' if retrieval_source=='fixed'
                   else 'retrieval_test/observed8_topk_top50.jsonl')
        if not topk.is_file(): raise FileNotFoundError(topk)
        mod.CKPT=str(checkpoint.resolve())
        mod.prepare(SimpleNamespace(root=target,topk=topk,k=10))
    else:
        protocol=read(target/'protocol.json')
        mod.CKPT=protocol['checkpoint']
        if action=='dense-all' and not (target/'pilot10_pnp/integrated_summary.json').is_file():
            raise RuntimeError('Run and inspect dense-pilot first')
        sys.argv=[str(PROJECT/'tools/evaluate_retrieval_topk_pipeline.py'),'run','--root',str(target),
                  '--scope','pilot10' if action=='dense-pilot' else 'all507']
        mod.main()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','data','manifests','test-manifests','evaluate-fixed',
                                   'stats','train','select','evaluate','dense-prepare','dense-pilot','dense-all'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--role',choices=['train','val','test','all'],default='all')
    p.add_argument('--limit',type=int,default=0,help='Only for data: max segments, 0=all')
    p.add_argument('--loftr-checkpoint',type=Path)
    p.add_argument('--retrieval-source',choices=['fixed','trained8'],default='trained8')
    a=p.parse_args();root=a.root.resolve()
    if a.limit<0: p.error('--limit must be nonnegative')
    if a.action=='data': data(root,a.role,a.limit)
    elif a.action=='test-manifests': manifests(root,test_only=True)
    elif a.action=='evaluate-fixed': evaluate_fixed(root)
    elif a.action.startswith('dense-'): dense(root,a.action,a.loftr_checkpoint,a.retrieval_source)
    else: globals()[a.action](root)


if __name__=='__main__': main()
