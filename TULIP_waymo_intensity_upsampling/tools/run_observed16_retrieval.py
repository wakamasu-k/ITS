"""Train and evaluate Retrieval on native observed 16-line Waymo data."""
import argparse
from pathlib import Path

import run_observed8_localization as shared
from run_original_tulip_retrieval import check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'data', 'manifests', 'stats',
                                          'check', 'train', 'select', 'evaluate'])
    parser.add_argument('--root', type=Path, default=Path('/mnt/w/its_outputs/observed16_localization'))
    parser.add_argument('--role', choices=['train', 'val', 'test', 'all'], default='all')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    if args.action == 'prepare':
        shared.prepare(root, lines=16)
        return
    if shared.read(root/'plan.json')['native_rings'] != list(range(0, 64, 4)):
        parser.error('--root must refer to a prepared observed16 experiment')
    if args.action == 'data':
        shared.data(root, args.role, args.limit, lines=16)
    elif args.action in ('manifests', 'evaluate'):
        getattr(shared, args.action)(root, lines=16)
    elif args.action == 'check':
        check(root)
    else:
        if args.action == 'train':
            check(root)
        getattr(shared, args.action)(root)


if __name__ == '__main__':
    main()
