"""Train a trial Stage-1 checkpoint from observed8 pilot or benchmark labels."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import tempfile

import run_observed8_localization as shared


def anchor_rows(camera_path, labels, expected_pairs=8):
    camera = shared.read(camera_path)
    cam = (shared.ORIGINAL / 'cam_gray' / camera['subset'] / camera['segment_id']
           / camera['camera_name'] / Path(camera['cam_image_path']).name)
    with (labels / 'summary.csv').open() as source:
        rows = list(csv.DictReader(source))
    if (not rows or len({row['image'] for row in rows}) != len(rows)
            or (expected_pairs is not None and len(rows) != expected_pairs)):
        raise ValueError(f'Unexpected label rows in {labels}')
    for row in rows:
        stem = row['image']
        anc = camera_path.parent / f'{stem}.png'
        gt = labels / f'{stem}_gt.npz'
        for path in (cam, anc, gt):
            if not path.is_file():
                raise FileNotFoundError(path)
        yield dict(anc_png=anc, cam_png=cam, gt_npz=gt, n_matches=int(row['matches']),
                   n_fine=int(row['fine']),
                   pair_name=f'{camera["segment_id"]}_{camera_path.parent.name}_{stem}', error='')


def manifest_rows(pilot):
    depth_root = Path('/mnt/w/its_outputs/observed8_loftr/depth_pilot')
    for index, root in enumerate((depth_root, pilot / 'depth')):
        cameras = list(root.glob('*/*/*/cam.json'))
        if len(cameras) != 1:
            raise ValueError(f'Expected one camera record under {root}')
        yield from anchor_rows(cameras[0], pilot / f'segment_{index}')


def benchmark_rows(benchmark):
    report = shared.read(benchmark / 'benchmark_summary.json')
    if len(report) != 3 or len({row['segment'] for row in report}) != 3:
        raise ValueError('Expected three distinct benchmark training segments')
    plan = shared.read(shared.PROJECT / 'outputs/observed8_localization_20261024/plan.json')
    train = {job['segment']: job['subset'] for job in plan['jobs'] if job['role'] == 'train'}
    for row in report:
        segment = row['segment']
        cameras = sorted((benchmark / 'depth' / train[segment] / segment).glob('*/cam.json'))
        if len(cameras) != 10 or (row['anchors'], row['pairs']) != (10, 80):
            raise ValueError(f'Incomplete benchmark segment: {segment}')
        for camera in cameras:
            yield from anchor_rows(camera, benchmark / 'labels' / segment / camera.parent.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--pilot', type=Path)
    source.add_argument('--benchmark', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path('/mnt/w/its_outputs/observed8_loftr'))
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    rows = list(benchmark_rows(args.benchmark.resolve()) if args.benchmark
                else manifest_rows(args.pilot.resolve()))
    args.output_root.mkdir(parents=True, exist_ok=True)
    prefix = 'stage1_trial_' if args.benchmark else 'stage1_smoke_'
    run = Path(tempfile.mkdtemp(prefix=prefix, dir=args.output_root.resolve()))
    manifest = run / 'manifest.csv'
    with manifest.open('x') as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Manifest: {manifest} ({len(rows)} pairs)', flush=True)
    if args.prepare_only:
        return
    loftr = shared.REF / 'LoFTR'
    python = shared.ITS / '.venv_tulip/bin/python'
    checkpoint = shared.ITS / 'waymo_outputs/old32_dense/loftr_retrain_stage1_paper_wdrive/stage1_final_gstep715958.pth'
    trainer = shared.PROJECT / 'tools/run_loftr_stage1_compat.py'
    env = dict(os.environ, PYTHONPATH=str(loftr/'src'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    subprocess.run([python, '-c', 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'],
                   env=env, check=True)
    command = [python, '-u', trainer, '--manifest', manifest, '--out-dir', run/'weights',
               '--init-ckpt', checkpoint, '--epochs', '1', '--max-steps', str(len(rows)),
               '--num-workers', '0', '--amp-dtype', 'none']
    shared.run(command, run/'console.log', env=env)


if __name__ == '__main__':
    main()
