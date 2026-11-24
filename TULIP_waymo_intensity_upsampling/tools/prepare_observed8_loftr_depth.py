"""Render a small observed-8-line depth sample for validating LoFTR labels."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import run_observed8_localization as shared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root', type=Path,
                        default=Path('/mnt/w/its_outputs/observed8_loftr/depth_pilot'))
    parser.add_argument('--train-segment-index', type=int, default=0)
    parser.add_argument('--max-anchors', type=int, default=1)
    args = parser.parse_args()
    if args.max_anchors != -1 and args.max_anchors < 1:
        parser.error('--max-anchors must be positive or -1 for the full segment')
    source = shared.PROJECT / 'outputs/observed8_localization_20261024'
    plan = shared.read(source / 'plan.json')
    jobs = [job for job in plan['jobs'] if job['role'] == 'train']
    if not 0 <= args.train_segment_index < len(jobs):
        parser.error('--train-segment-index is out of range')
    job = jobs[args.train_segment_index]
    train = source / 'data/train'
    provenance = train / 'maps' / job['subset'] / job['segment'] / 'observed8_provenance.json'
    metadata = shared.read(provenance)
    if metadata['native_ring_ids'] != list(range(0, 64, 8)) or metadata['max_frames'] is not None:
        raise ValueError('Expected a complete observed-8-line map')
    out = args.out_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    renderer = shared.REF / 'data_make/train/make_retrieval_shifted_ri_txt.py'
    command = [sys.executable, str(renderer), '--tfrecord', job['tfrecord'],
               '--subset', job['subset'], '--maps-root', str(train / 'maps'),
               '--submaps-root', str(train / 'anchors'),
               '--cam-gray-root', str(shared.ORIGINAL / 'cam_gray'),
               '--out-root', str(out), '--num-samples-per-anchor', '8',
               '--seed', '0', '--max-anchors', str(args.max_anchors), '--zero-ri-mode', 'render',
               '--write-index', 'true', '--overwrite', 'false']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='4')
    subprocess.run(command, env=env, check=True)
    segment = out / job['subset'] / job['segment']
    markers = list(segment.glob('*/_done.json'))
    layout = shared.read(train / 'anchors' / job['subset'] / job['segment'] / 'submaps.json')
    total = len(layout['submaps'] if isinstance(layout, dict) else layout)
    expected = total if args.max_anchors == -1 else min(total, args.max_anchors)
    if len(markers) != expected:
        raise RuntimeError(f'Expected {expected} completed anchors in {segment}')
    for marker in sorted(markers):
        anchor = marker.parent
        images = sorted(anchor.glob('ri_*.png'))
        if not images or any(not image.with_name(image.stem + '_index_depth.npz').is_file()
                             or not image.with_suffix('.json').is_file() for image in images):
            raise RuntimeError(f'Missing image, depth, or pose data in {anchor}')
        print(json.dumps({'anchor': str(anchor), 'image_depth_pairs': len(images)}))


if __name__ == '__main__':
    main()
