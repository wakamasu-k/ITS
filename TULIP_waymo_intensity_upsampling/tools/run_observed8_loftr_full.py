"""Build full observed8 training labels on W: and train Stage-1."""
import argparse
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import run_observed8_localization as shared
from run_observed8_loftr_smoke import anchor_rows

ROOT = Path('/mnt/w/its_outputs/observed8_loftr/full_stage1')
SOURCE = shared.PROJECT / 'outputs/observed8_localization_20261024'


def build(root, jobs, limit):
    from check_observed8_loftr_labels import generate

    for index, job in enumerate(jobs[:limit] if limit else jobs):
        segment = job['segment']
        marker = root/'completed'/f'{segment}.json'
        if marker.is_file():
            print(f'REUSE: {segment}', flush=True)
            continue
        depth = root/'depth'/job['subset']/segment
        depth_ready = root/'rendered'/f'{segment}.json'
        if not depth_ready.is_file():
            subprocess.run([sys.executable, shared.PROJECT/'tools/prepare_observed8_loftr_depth.py',
                            '--train-segment-index', str(index), '--max-anchors', '-1',
                            '--out-root', root/'depth'], check=True)
            shared.save(depth_ready, {'segment': segment})
        anchors = sorted(marker.parent for marker in depth.glob('*/_done.json'))
        layout = shared.read(SOURCE/'data/train/anchors'/job['subset']/segment/'submaps.json')
        if len(anchors) != len(layout['submaps']):
            raise ValueError(f'Incomplete depth data: {depth}')
        segment_labels = root/'labels'/segment
        segment_labels.mkdir(parents=True, exist_ok=True)
        pairs = 0
        for anchor in anchors:
            labels = segment_labels/anchor.name
            if not labels.exists():
                with tempfile.TemporaryDirectory(prefix='.partial_', dir=segment_labels) as scratch:
                    staged = Path(scratch)/'labels'
                    generate(anchor, staged, True, write_visuals=False, allow_empty=True)
                    staged.rename(labels)
            pairs += sum(1 for _ in anchor_rows(anchor/'cam.json', labels, expected_pairs=None))
        shared.save(marker, dict(segment=segment, anchors=len(anchors), pairs=pairs))
        print(f'DONE {index+1}/{len(jobs)}: {segment}, {len(anchors)} anchors, {pairs} pairs',
              flush=True)


def make_manifest(root, jobs):
    rows = []
    dropped = 0
    for job in jobs:
        segment = job['segment']
        marker = shared.read(root/'completed'/f'{segment}.json')
        anchors = sorted((root/'depth'/job['subset']/segment).glob('*/cam.json'))
        if len(anchors) != marker['anchors']:
            raise ValueError(f'Anchor count changed: {segment}')
        pairs = 0
        for camera in anchors:
            labels = root/'labels'/segment/camera.parent.name
            for row in anchor_rows(camera, labels, expected_pairs=None):
                pairs += 1
                if row['n_matches'] >= 8:
                    rows.append(row)
                else:
                    dropped += 1
        if pairs != marker['pairs']:
            raise ValueError(f'Pair count changed: {segment}')
    if not rows:
        raise ValueError('No usable training pairs')
    with tempfile.NamedTemporaryFile(mode='w', newline='', dir=root, delete=False) as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path(target.name).replace(root/'manifest.csv')
    print(f'Manifest: {len(rows)} pairs, {dropped} below eight matches', flush=True)


def train(root):
    manifest = root/'manifest.csv'
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    python = shared.ITS/'.venv_tulip/bin/python'
    checkpoint = shared.ITS/'waymo_outputs/old32_dense/loftr_retrain_stage1_paper_wdrive/stage1_final_gstep715958.pth'
    weights = root/'weights'
    command = [python, '-u', shared.PROJECT/'tools/run_loftr_stage1_compat.py',
               '--manifest', manifest, '--out-dir', weights, '--init-ckpt', checkpoint,
               '--epochs', '1', '--max-steps', '-1', '--num-workers', '2', '--amp-dtype', 'none']
    if (weights/'epoch_001.pth').is_file():
        raise RuntimeError(f'Epoch checkpoint already exists: {weights}')
    if (weights/'latest_model.pth').is_file():
        command.append('--resume')
    env = dict(os.environ, PYTHONPATH=str(shared.REF/'LoFTR/src'),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    subprocess.run([python, '-c', 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'],
                   env=env, check=True)
    with tempfile.NamedTemporaryFile(prefix='train_', suffix='.log', dir=root, delete=False) as log:
        log_path = Path(log.name)
    log_path.unlink()
    shared.run(command, log_path, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['data', 'manifest', 'train', 'status'])
    parser.add_argument('--limit-segments', type=int, default=0, help='data only, 0 means all')
    args = parser.parse_args()
    if args.limit_segments < 0 or (args.limit_segments and args.action != 'data'):
        parser.error('--limit-segments must be nonnegative and used only with data')
    if not os.path.ismount('/mnt/w'):
        raise RuntimeError('Mount W: first')
    jobs = [job for job in shared.read(SOURCE/'plan.json')['jobs'] if job['role'] == 'train']
    if args.action == 'status':
        done = sum((ROOT/'completed'/f'{job["segment"]}.json').is_file() for job in jobs)
        print(f'Completed: {done}/{len(jobs)} segments')
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'data':
            for folder in ('completed', 'rendered'):
                (ROOT/folder).mkdir(exist_ok=True)
            build(ROOT, jobs, args.limit_segments)
        elif args.action == 'manifest':
            if (ROOT/'weights').exists():
                raise RuntimeError('The training manifest is fixed once training starts')
            make_manifest(ROOT, jobs)
        else:
            train(ROOT)


if __name__ == '__main__':
    main()
