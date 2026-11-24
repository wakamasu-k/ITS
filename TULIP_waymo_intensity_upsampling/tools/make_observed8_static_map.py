"""Run the inherited static-map builder with native TOP rings 0,8,...,56.

The inherited file is read and patched in memory; it is never edited.
All its filtering, calibration and pose handling are retained. There is no
image-row thinning and no random point subsampling.
"""
import hashlib
import json
from pathlib import Path
import sys


SOURCE = Path('/home/wakamatsu/ITS/handover_shimizu/src/data_make/valid_test/make_static_map_txt.py')


def adapted_source(source):
    edits = {
        'not in (64, 32, 16)': 'not in (64, 32, 16, 8)',
        'choices=[64, 32, 16]': 'choices=[64, 32, 16, 8]',
        '        elif lidar_lines != 64:':
            '        elif lidar_lines == 8:\n'
            '            keep = (ring_id % 8 == (0 if ring_keep_even else 1))\n'
            '            xyz, inten = xyz[keep], inten[keep]\n'
            '        elif lidar_lines != 64:',
        '        H, W, C = [d if isinstance(d, int) else d.size for d in ri_top.shape.dims]':
            '        H, W, C = [d if isinstance(d, int) else d.size for d in ri_top.shape.dims]\n'
            '        if H != 64:\n'
            '            raise ValueError(f"Expected native TOP 64 rows, found {H}")',
    }
    for old, new in edits.items():
        if source.count(old) != 1:
            raise RuntimeError(f'Inherited code changed; review adaptation: {old!r}')
        source = source.replace(old, new)
    return source


def main():
    argv = sys.argv[1:]
    max_frames = None
    if '--max-frames' in argv:
        pos = argv.index('--max-frames')
        max_frames = int(argv[pos+1])
        if max_frames < 1: raise ValueError('--max-frames must be positive')
        del argv[pos:pos+2]
    if '--lidar-lines' in argv or '--ring-keep-even' in argv:
        raise SystemExit('This wrapper fixes --lidar-lines 8 --ring-keep-even true')
    source = SOURCE.read_text()
    code = adapted_source(source)
    if max_frames:
        old='    ds = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")'
        if code.count(old)!=1: raise RuntimeError('Review frame-limit adaptation')
        code=code.replace(old,old+f'.take({max_frames})')
    print('[observed8] native TOP rings: 0,8,16,24,32,40,48,56', flush=True)
    print('[observed8] inherited source SHA256: ' + hashlib.sha256(source.encode()).hexdigest(), flush=True)
    sys.argv = [str(SOURCE), *argv, '--lidar-lines', '8', '--ring-keep-even', 'true']
    # TensorFlow preprocessing stays on CPU; reserve CUDA for inference/training.
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    namespace = {'__name__': '__main__', '__file__': str(SOURCE)}
    exec(compile(code, str(SOURCE), 'exec'), namespace)
    if '--out-root' in argv and '--tfrecord' in argv and '--subset' in argv:
        tfrecord = Path(argv[argv.index('--tfrecord') + 1])
        if tfrecord.suffix == '.tfrecord':
            out = Path(argv[argv.index('--out-root') + 1]) / argv[argv.index('--subset') + 1] / tfrecord.stem
            if not (out / 'map_static.npz').is_file():
                raise RuntimeError('Static map was not created')
            (out / 'observed8_provenance.json').write_text(json.dumps({
                'native_ring_ids': list(range(0, 64, 8)),
                'inherited_source': str(SOURCE),
                'inherited_sha256': hashlib.sha256(source.encode()).hexdigest(),
                'source_tfrecord': str(tfrecord.resolve()),
                'max_frames': max_frames,
            }, indent=2) + '\n')


if __name__ == '__main__':
    main()
