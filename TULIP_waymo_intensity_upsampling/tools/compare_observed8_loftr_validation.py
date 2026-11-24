"""Compare baseline and trial LoFTR on the same 374 validation queries."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import evaluate_retrieval_topk_pipeline as dense
import run_observed8_localization as shared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    args = parser.parse_args()
    trial = args.trial.resolve(strict=True)
    baseline = Path(dense.CKPT).resolve(strict=True)
    output = Path('/mnt/w/its_outputs/observed8_loftr')
    if not os.path.ismount('/mnt/w'):
        raise RuntimeError('Mount W: before running validation')
    source = shared.PROJECT / 'outputs/observed8_localization_20261024'
    original = source / 'validation_recheck_20261029/test507/command.json'
    command = shared.read(original)
    run = Path(tempfile.mkdtemp(prefix='validation374_', dir=output))
    print(f'Output: {run}', flush=True)
    retrieval = run / 'retrieval'
    retrieval.mkdir()
    for flag, value in {
        '--pairs': source/'manifests/val_pairs.parquet',
        '--db': source/'manifests/val_db.parquet',
        '--out_dir': retrieval,
        '--name': 'observed8_validation',
    }.items():
        command[command.index(flag)+1] = str(value)
    shared.save(run/'retrieval_command.json', command)
    shared.run(command, run/'retrieval.log')
    topk = retrieval/'observed8_validation_topk_top50.jsonl'
    raw, _, rows = dense.candidate_manifest(topk, 10, 374)
    scope = 'validation374'
    manifest = run/f'{scope}.csv'
    dense.write_csv(manifest, rows)
    shared.save(run/'protocol.json', dict(
        queries=374, candidates=len(rows), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        retrieval_sha256=hashlib.sha256(raw).hexdigest(), baseline=str(baseline), trial=str(trial)))
    loftr = shared.REF/'LoFTR'
    env = dict(os.environ, PYTHONPATH=str(loftr/'src'), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    report = {}
    for name, checkpoint in [('baseline', baseline), ('trial', trial)]:
        arm = run/name
        arm.mkdir()
        shutil.copyfile(manifest, arm/manifest.name)
        out = arm/f'{scope}_pnp'
        out.mkdir()
        command = [dense.PYTHON, '-u', loftr/'tools/eval_cross_match_to_pose_zalign_multi_visall_v2.py',
                   '--manifest', manifest, '--ckpt', checkpoint, '--out-dir', out, '--device', 'cuda',
                   '--amp-dtype', 'fp16', '--gt-type', 'all', '--z-align', 'none',
                   '--icp-csv', dense.ICP, '--no-write-per-radius']
        shared.save(arm/'command.json', list(map(str, command)))
        shared.run(command, arm/'console.log', env=env)
        dense.summarize(arm, scope)
        report[name] = shared.read(out/'integrated_summary.json')
    shared.save(run/'comparison.json', report)
    print(f'Done: {run/"comparison.json"}', flush=True)


if __name__ == '__main__':
    main()
