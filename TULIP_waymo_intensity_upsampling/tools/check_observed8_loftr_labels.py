"""Generate pilot matches using same-map depth at the real camera pose."""
import argparse
import csv
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

import run_observed8_localization as shared

sys.path.insert(0, str(shared.REF / 'LoFTR/tools'))
import make_loftr_matches_gt_cross_eval_v3 as geometry


@lru_cache(maxsize=1)
def segment_camera_boxes(tfrecord, camera_name):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2

    camera_id = dataset_pb2.CameraName.Name.Value(camera_name)
    records = []
    for record in tf.compat.v1.io.tf_record_iterator(tfrecord):
        frame = dataset_pb2.Frame.FromString(record)
        calib = next(c for c in frame.context.camera_calibrations if c.name == camera_id)
        labels = next((group.labels for group in frame.camera_labels
                       if group.name == camera_id), None)
        boxes = []
        for label in labels or ():
            if label.type not in (label.TYPE_VEHICLE, label.TYPE_PEDESTRIAN, label.TYPE_CYCLIST):
                continue
            box = label.box
            boxes.append((box.center_x-box.length/2, box.center_y-box.width/2,
                          box.center_x+box.length/2, box.center_y+box.width/2))
        records.append((frame.timestamp_micros, (calib.height, calib.width),
                        boxes if labels is not None else None))
    return records


def camera_boxes(camera, shape):
    plan = shared.read(shared.PROJECT / 'outputs/observed8_localization_20261024/plan.json')
    job = next(job for job in plan['jobs'] if job['role'] == 'train'
               and job['segment'] == camera['segment_id'])
    records = segment_camera_boxes(job['tfrecord'], camera['camera_name'])
    index = int(camera['anchor_frame_index'])
    if not 0 <= index < len(records):
        raise ValueError('Camera frame is missing from TFRecord')
    timestamp, raw_shape, boxes = records[index]
    if timestamp != int(camera['cam_meta']['timestamp_us']) or raw_shape != shape:
        raise ValueError('Camera image and TFRecord annotation disagree')
    if boxes is None:
        raise ValueError('Camera annotations are absent for this frame')
    return boxes


def generate(anchor, out, mask_camera_objects, write_visuals=True, allow_empty=False):
    start = time.monotonic()
    anchor = anchor.resolve()
    out = out.resolve()
    camera = shared.read(anchor / 'cam.json')
    cam_meta = camera['cam_meta']
    pose = geometry._parse_world_to_cam(cam_meta)
    k = tuple(map(float, camera['intrinsic']))
    if not np.allclose(k, geometry._parse_intrinsic4(cam_meta)):
        raise ValueError('Camera intrinsics disagree')
    cam_path = (shared.ORIGINAL / 'cam_gray' / camera['subset'] / camera['segment_id']
                / camera['camera_name'] / Path(camera['cam_image_path']).name)
    cam_img = cv2.imread(str(cam_path), cv2.IMREAD_GRAYSCALE)
    if cam_img is None:
        raise FileNotFoundError(cam_path)
    h, w = cam_img.shape
    boxes = camera_boxes(camera, (h, w)) if mask_camera_objects else []
    grid_args = dict(stride=8, cell_point='topleft', margin=8, min_depth=.5, max_depth=200.)

    def raster(stem):
        pi, depth = geometry.load_index_depth(anchor / (stem + '_index_depth.npz'))
        if pi.shape != (h, w) or depth.shape != (h, w):
            raise ValueError(f'Depth shape differs from camera image: {stem}')
        return pi, depth, geometry.build_rep_grid(pi, depth, **grid_args)

    # The renderer's zero-offset vehicle pose can differ from the camera pose.
    # Build the visibility raster at the pose recorded for the camera image.
    map_path = (shared.PROJECT / 'outputs/observed8_localization_20261024/data/train/maps'
                / camera['subset'] / camera['segment_id'] / 'map_static.npz')
    with np.load(map_path, allow_pickle=False) as data:
        xyz = data['xyz'] @ pose[:3, :3].T + pose[:3, 3]
    xyz = xyz[np.isfinite(xyz).all(axis=1) & (xyz[:, 0] >= .5) & (xyz[:, 0] <= 200)]
    u, v, depth = geometry.project_xforward(xyz, *k)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    d1 = np.full((h, w), np.inf, dtype=np.float32)
    np.minimum.at(d1, (v[inside].astype(int), u[inside].astype(int)), depth[inside])
    # The geometry helpers use this array as a coverage mask.
    pi1 = np.where(np.isfinite(d1), 0, -1).astype(np.int32)
    dst = geometry.build_rep_grid(pi1, d1, **grid_args)
    intrinsics = dict(zip(('fx0', 'fy0', 'cx0', 'cy0', 'fx1', 'fy1', 'cx1', 'cy1'), k + k))
    world = np.eye(4)

    def match(a, b, pa, pb, pi, depth):
        return geometry.compute_nn_map_to_grid_cross(
            a, b, pa, pb, world, **intrinsics, point_index_dst=pi, depth_dst=depth,
            reproj_th=4., vis_mode='depth', depth_abs_th=1., depth_rel_th=.01)

    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for image in sorted(anchor.glob('ri_*.png')):
        t = time.monotonic()
        stem = image.stem
        pi0, d0, src = raster(stem)
        source_pose = geometry._parse_world_to_cam(shared.read(image.with_suffix('.json')))
        i, j = geometry.mutual_matches_from_maps(
            match(src, dst, source_pose, pose, pi1, d1),
            match(dst, src, pose, source_pose, pi0, d0))
        xy0 = np.column_stack((src.rep_x[i], src.rep_y[i])).astype(np.float32)
        xy1 = np.column_stack((dst.rep_x[j], dst.rep_y[j])).astype(np.float32)
        u, v, depth = geometry.compute_mkpts1_f_cross(
            src, i, source_pose, pose, world, **intrinsics)
        fine = np.column_stack((u, v)).astype(np.float32)
        before = len(i)
        excluded = np.zeros(before, dtype=bool)
        for x0, y0, x1, y1 in boxes:
            for xy in (xy1, fine):
                excluded |= ((xy[:, 0] >= x0) & (xy[:, 0] <= x1)
                             & (xy[:, 1] >= y0) & (xy[:, 1] <= y1))
        keep = ~excluded
        i, j, xy0, xy1, fine, depth = (value[keep] for value in (i, j, xy0, xy1, fine, depth))
        offset = fine - xy1
        ok = np.isfinite(fine).all(axis=1) & (depth > 0) & (np.abs(offset) <= 4).all(axis=1)
        if (not allow_empty and len(i) == 0) or not np.isfinite(fine).all():
            raise ValueError(f'Projection sanity check failed: {stem}')
        np.savez_compressed(out / (stem + '_gt.npz'), mkpts0=xy0, mkpts1=xy1,
                            mkpts1_f=fine, offset1=offset, fine_ok=ok.astype(np.uint8),
                            i_ids=i, j_ids=j, hw0=[h, w], hw1=[h, w], stride=8,
                            meta_json=json.dumps({'visibility': 'same-map camera-pose depth proxy',
                                                  'camera_object_mask': mask_camera_objects}))
        ri = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if ri is None or ri.shape != (h, w):
            raise ValueError(f'Unexpected RI shape: {image}')
        rows.append(dict(image=stem, matches_before=before, excluded=int(excluded.sum()),
                         matches=len(i), fine=int(ok.sum()), seconds=round(time.monotonic()-t, 3)))
        if not write_visuals:
            continue
        canvas = cv2.cvtColor(cv2.hconcat([ri, cam_img]), cv2.COLOR_GRAY2BGR)
        for x0, y0, x1, y1 in boxes:
            cv2.rectangle(canvas, (w+int(x0), int(y0)), (w+int(x1), int(y1)), (0, 100, 255), 2)
        ids = np.flatnonzero(ok)
        for n in ids[np.linspace(0, len(ids)-1, min(len(ids), 36), dtype=int)]:
            a = tuple(np.rint(xy0[n]).astype(int))
            b = tuple(np.rint(fine[n] + [w, 0]).astype(int))
            cv2.line(canvas, a, b, (70, 210, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, a, 4, (255, 210, 0), -1)
            cv2.circle(canvas, b, 4, (255, 210, 0), -1)
        cv2.imwrite(str(out / (stem + '_matches.jpg')), canvas)
    with (out / 'summary.csv').open('x') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {'camera_boxes': len(boxes), 'pairs': rows,
            'elapsed_seconds': round(time.monotonic()-start, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchor', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mask-camera-objects', action='store_true')
    args = parser.parse_args()
    print(json.dumps(generate(args.anchor, args.out, args.mask_camera_objects), indent=2))


if __name__ == '__main__':
    main()
