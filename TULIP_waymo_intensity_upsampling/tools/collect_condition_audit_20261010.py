#!/usr/bin/env python3
"""Collect a read-only research audit; only a new report is written.

Uses Python standard library. Does not run experiments or modify input files.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path,
                        default=Path('/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling'))
    args = parser.parse_args()
    root = args.project_root.resolve()
    if not root.is_dir():
        parser.error('Project directory does not exist: ' + str(root))
    reference = root.parent / 'handover_shimizu'
    sections = []

    def add(title, body):
        sections.append('\n## ' + title + '\n\n' + body + '\n')

    def read_file(path):
        path = Path(path)
        if not path.is_file():
            add(str(path), 'MISSING')
            return
        try:
            content = path.read_text(encoding='utf-8', errors='replace')
            add(str(path), '```text\n' + content + '\n```')
        except OSError as exc:
            add(str(path), 'READ ERROR: ' + str(exc))

    add('Audit scope', 'Project: ' + str(root) + '\nReference: ' + str(reference)
        + '\nInputs are read-only. No model inference, rendering, or filtering is run.'
        + '\nThis report may include local paths and source code.')
    try:
        result = subprocess.run(
            ['git', '--no-optional-locks', '-C', str(root), 'status', '--short', '--branch'],
            capture_output=True, text=True, timeout=60, check=False)
        add('Git status', 'Exit: ' + str(result.returncode) + '\n```text\n'
            + result.stdout + result.stderr + '\n```')
    except (OSError, subprocess.TimeoutExpired) as exc:
        add('Git status', str(exc))

    for path in [root.parent / 'AGENTS.md', root / 'AGENTS.md',
                 root / 'tools/AGENTS.md', root / 'scripts/AGENTS.md',
                 reference / 'AGENTS.md']:
        if path.is_file():
            read_file(path)

    project_files = [
        'tools/build_tulip_16_to_32_static_map.py',
        'tools/prepare_shimizu_retrieval_protocol.py',
        'tools/make_shimizu_abc_retrieval_manifests.py',
        'tools/compare_anchor_reproduction_sample.py',
        'scripts/run_shimizu_507q_754db_abc_maps.sh',
        'scripts/run_shimizu_754db_render_ab.sh',
        'configs/shimizu_retrieval_modality_stats.json',
    ]
    for name in project_files:
        read_file(root / name)
    for name in [
        'src/data_make/train/make_static_map_txt.py',
        'src/data_make/train/make_submaps_distance_txt.py',
        'src/data_make/train/render_anchor_intensity_txt.py',
        'src/data_make/valid_test/render_anchor_intensity_txt_se3_fix.py',
    ]:
        read_file(reference / name)

    experiment = root / 'outputs/shimizu_protocol_507q_754db_abc'
    for name in [
        'retrieval_A_retry1/shimizu_507q_754db_A_metrics.json',
        'retrieval_B/shimizu_507q_754db_B_metrics.json',
        'A_vs_reference_sample/summary.json',
        'B_vs_reference_sample/summary.json',
        'C_reproduction_comparison_sample/summary.json',
    ]:
        read_file(experiment / name)
    read_file(root / 'outputs/shimizu_retrieval_reproduction_epoch16_retry1/shimizu_test4_reproduced_metrics.json')

    for variant in ['A', 'B']:
        folder = experiment / ('anchors_' + variant)
        if not folder.is_dir():
            add('Anchor inventory ' + variant, 'MISSING: ' + str(folder))
            continue
        metadata = sorted(folder.rglob('anchor.json'))
        counts = {name: sum(1 for _ in folder.rglob(name))
                  for name in ['anchor.png', 'anchor_index.npz']}
        counts['anchor.json'] = len(metadata)
        add('Anchor inventory ' + variant, json.dumps(counts, indent=2))
        # One metadata sample per segment, plus any root-level submap definition.
        seen_segments = set()
        for path in metadata:
            segment = path.parent.parent
            if segment not in seen_segments:
                seen_segments.add(segment)
                read_file(path)

    maps = experiment / 'maps'
    if maps.is_dir():
        for path in sorted(maps.rglob('map_static.npz')):
            try:
                # Read ZIP member directory only, not large point arrays.
                with zipfile.ZipFile(path) as archive:
                    members = [{'name': item.filename, 'bytes_uncompressed': item.file_size,
                                'bytes_compressed': item.compress_size}
                               for item in archive.infolist()]
                add('Map array inventory: ' + str(path), json.dumps(members, indent=2))
            except (OSError, zipfile.BadZipFile) as exc:
                add(str(path), str(exc))
    else:
        add('Map inventory', 'MISSING: ' + str(maps))

    add('Next analysis; not yet executed',
        '1. Compare source conditions, including coordinate transforms and masking.\n'
        '2. Establish whether 75 m means normalization scale, clipping, or a validity threshold.\n'
        '3. Identify whether per-frame range, ring, column, and generation labels are preserved.\n'
        '4. Design intensity-only ablation preserving geometry and measured-ring values.\n'
        '5. Verify metrics definitions and sample scope before interpreting differences.\n'
        '\nDo not calculate sensor range as the norm of world-coordinate XYZ. '
        'World coordinates require the corresponding sensor pose; sensor-frame range is preferable.\n'
        'The earlier handover suggestion to threshold the world-coordinate norm at 75 m is incorrect.\n'
        'A normalization scale of 75 m alone does not establish that values above it are outliers.')

    report_dir = root / 'outputs/condition_audit'
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = report_dir / ('condition_audit_' + stamp + '.md')
    with output.open('x', encoding='utf-8') as handle:
        handle.write('# Shimizu / TULIP condition audit\n' + ''.join(sections))
    print('REPORT: ' + str(output))
    print('Attach this report to the chat. No experiment or input modification was performed.')


if __name__ == '__main__':
    main()
