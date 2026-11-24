"""Measure depth rendering and label generation for 3 training segments."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from check_observed8_loftr_labels import generate
import run_observed8_localization as shared


def main():
    root = Path('/mnt/w/its_outputs/observed8_loftr')
    if not os.path.ismount('/mnt/w'):
        raise RuntimeError('Mount W: before running the benchmark')
    root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='data_benchmark_', dir=root))
    print(f'Output: {run}', flush=True)
    jobs = [j for j in shared.read(shared.PROJECT /
            'outputs/observed8_localization_20261024/plan.json')['jobs'] if j['role'] == 'train']
    report = []
    for index in (0, 1, 2):
        job = jobs[index]
        start = time.monotonic()
        subprocess.run([sys.executable, str(shared.PROJECT/'tools/prepare_observed8_loftr_depth.py'),
                        '--train-segment-index', str(index), '--max-anchors', '10',
                        '--out-root', str(run/'depth')], check=True)
        render_seconds = time.monotonic() - start
        source = run/'depth'/job['subset']/job['segment']
        anchors = sorted(marker.parent for marker in source.glob('*/_done.json'))
        start = time.monotonic()
        pairs = []
        for anchor in anchors:
            result = generate(anchor, run/'labels'/job['segment']/anchor.name, True)
            pairs.extend(result['pairs'])
            print(f'{job["segment"]}/{anchor.name}: {len(result["pairs"])} pairs', flush=True)
        row = dict(segment=job['segment'], anchors=len(anchors), pairs=len(pairs),
                   min_matches=min(p['matches'] for p in pairs),
                   max_matches=max(p['matches'] for p in pairs),
                   excluded=sum(p['excluded'] for p in pairs),
                   render_seconds=round(render_seconds, 2),
                   label_seconds=round(time.monotonic()-start, 2))
        report.append(row)
        print(json.dumps(row), flush=True)
    (run/'benchmark_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Done: {run}', flush=True)


if __name__ == '__main__':
    main()
