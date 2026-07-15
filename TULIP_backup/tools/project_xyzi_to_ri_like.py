import argparse
from pathlib import Path

import cv2
import numpy as np


def load_points(path):
    path = Path(path)

    if path.suffix == ".bin":
        pts = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
        return pts

    if path.suffix == ".ply":
        with open(path, "r") as f:
            lines = f.readlines()

        end_idx = None
        for i, line in enumerate(lines):
            if line.strip() == "end_header":
                end_idx = i + 1
                break

        if end_idx is None:
            raise ValueError(f"Cannot find end_header in {path}")

        pts = np.loadtxt(lines[end_idx:], dtype=np.float32)

        if pts.ndim == 1:
            pts = pts[None, :]

        if pts.shape[1] == 3:
            intensity = np.ones((pts.shape[0], 1), dtype=np.float32)
            pts = np.hstack([pts, intensity])

        return pts[:, :4]

    raise ValueError(f"Unsupported file type: {path}")


def read_kitti_calib(calib_path, camera):
    data = {}

    with open(calib_path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            vals = np.array([float(x) for x in value.strip().split()], dtype=np.float64)
            data[key] = vals

    if camera not in data:
        raise KeyError(f"{camera} not found in calib file")

    P = data[camera].reshape(3, 4)

    if "Tr" in data:
        Tr = data["Tr"].reshape(3, 4)
    elif "Tr_velo_to_cam" in data:
        Tr = data["Tr_velo_to_cam"].reshape(3, 4)
    else:
        raise KeyError("Tr or Tr_velo_to_cam not found in calib file")

    Tr4 = np.eye(4, dtype=np.float64)
    Tr4[:3, :4] = Tr

    return P, Tr4


def project_to_ri_like(points, calib_path, image_path, camera, point_size):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(image_path)

    h, w = gray.shape[:2]

    P, Tr4 = read_kitti_calib(calib_path, camera)

    xyz = points[:, :3].astype(np.float64)
    intensity = points[:, 3].astype(np.float32)

    pts_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)

    # KITTI: LiDAR座標 -> Camera座標
    cam = (Tr4 @ pts_h.T).T
    Xc = cam[:, 0]
    Yc = cam[:, 1]
    Zc = cam[:, 2]  # KITTIではこれがカメラ前方方向の深度

    cam_h = np.concatenate([cam[:, :3], np.ones((cam.shape[0], 1), dtype=np.float64)], axis=1)
    proj = (P @ cam_h.T).T

    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]

    u_i = u.astype(np.int32)
    v_i = v.astype(np.int32)

    valid = (
        (Zc > 0.1)
        & (u_i >= 0)
        & (u_i < w)
        & (v_i >= 0)
        & (v_i < h)
        & np.isfinite(intensity)
    )

    intensity_img = np.zeros((h, w), dtype=np.float32)
    depth_img = np.full((h, w), np.inf, dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)

    radius = int(point_size)

    valid_indices = np.where(valid)[0]

    for idx in valid_indices:
        cx = int(u_i[idx])
        cy = int(v_i[idx])
        depth = float(Zc[idx])
        inten = float(intensity[idx])

        x0 = max(0, cx - radius)
        x1 = min(w - 1, cx + radius)
        y0 = max(0, cy - radius)
        y1 = min(h - 1, cy + radius)

        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                if (xx - cx) ** 2 + (yy - cy) ** 2 > radius ** 2:
                    continue

                # z-buffer：同じpixelでは手前の点を採用
                if depth < depth_img[yy, xx]:
                    depth_img[yy, xx] = depth
                    intensity_img[yy, xx] = inten
                    mask[yy, xx] = 255

    return intensity_img, depth_img, mask, gray


def postprocess_intensity(
    intensity_img,
    mask,
    upper_percentile=99.5,
    global_up_val=None,
    hist_eq=True,
    clahe=True,
    gamma=1.0,
    clahe_clip_limit=2.0,
    clahe_tile_grid=8,
):
    covered = mask > 0
    out = np.zeros_like(intensity_img, dtype=np.float32)

    if np.count_nonzero(covered) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    vals = intensity_img[covered].astype(np.float32)

    if global_up_val is not None and global_up_val > 0:
        upper = float(global_up_val)
    else:
        upper = float(np.percentile(vals, upper_percentile))

    upper = max(upper, 1e-6)

    out[covered] = np.clip(intensity_img[covered] / upper, 0.0, 1.0)

    if gamma != 1.0:
        out[covered] = np.power(out[covered], gamma)

    img_u8 = np.zeros_like(mask, dtype=np.uint8)
    img_u8[covered] = (out[covered] * 255.0).astype(np.uint8)

    if hist_eq:
        img_u8 = cv2.equalizeHist(img_u8)
        img_u8[~covered] = 0

    if clahe:
        clahe_obj = cv2.createCLAHE(
            clipLimit=float(clahe_clip_limit),
            tileGridSize=(int(clahe_tile_grid), int(clahe_tile_grid)),
        )
        img_u8 = clahe_obj.apply(img_u8)
        img_u8[~covered] = 0

    return img_u8


def save_depth_color(depth_img, mask, out_path):
    covered = (mask > 0) & np.isfinite(depth_img)

    if np.count_nonzero(covered) == 0:
        cv2.imwrite(str(out_path), np.zeros((*mask.shape, 3), dtype=np.uint8))
        return

    # KITTI/TULIPでは maximum_range=80m なので，0〜80mで固定正規化する
    d_min = 0.0
    d_max = 80.0

    norm = np.zeros_like(depth_img, dtype=np.float32)
    norm[covered] = np.clip(
        (depth_img[covered] - d_min) / (d_max - d_min),
        0.0,
        1.0,
    )

    # 近距離に点が集中するので，少し見やすくするためgamma補正
    # gamma < 1 にすると遠距離側の差が見やすくなる
    gamma = 0.6
    norm[covered] = np.power(norm[covered], gamma)

    depth_u8 = np.zeros_like(mask, dtype=np.uint8)

    # 今回は「近い=赤，遠い=青」にするため反転
    depth_u8[covered] = ((1.0 - norm[covered]) * 255.0).astype(np.uint8)

    # JETよりTURBOの方が階調が分かりやすい
    color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)

    color[~covered] = 0

    cv2.imwrite(str(out_path), color)
    
def save_overlay(gray, ri_img, mask, out_path):
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ri_color = cv2.applyColorMap(ri_img, cv2.COLORMAP_JET)

    covered = mask > 0
    overlay = gray_bgr.copy()
    overlay[covered] = cv2.addWeighted(gray_bgr[covered], 0.4, ri_color[covered], 0.6, 0)

    cv2.imwrite(str(out_path), overlay)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--points", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--camera", default="P0", choices=["P0", "P1", "P2", "P3"])

    parser.add_argument("--out", required=True)
    parser.add_argument("--mask_out", default=None)
    parser.add_argument("--overlay_out", default=None)
    parser.add_argument("--depth_color_out", default=None)

    parser.add_argument("--point_size", type=int, default=2)
    parser.add_argument("--upper_percentile", type=float, default=99.5)
    parser.add_argument("--global_up_val", type=float, default=None)

    parser.add_argument("--hist_eq", action="store_true")
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--clahe_clip_limit", type=float, default=2.0)
    parser.add_argument("--clahe_tile_grid", type=int, default=8)

    args = parser.parse_args()

    points = load_points(args.points)

    intensity_img, depth_img, mask, gray = project_to_ri_like(
        points=points,
        calib_path=args.calib,
        image_path=args.image,
        camera=args.camera,
        point_size=args.point_size,
    )

    ri_img = postprocess_intensity(
        intensity_img=intensity_img,
        mask=mask,
        upper_percentile=args.upper_percentile,
        global_up_val=args.global_up_val,
        hist_eq=args.hist_eq,
        clahe=args.clahe,
        gamma=args.gamma,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid=args.clahe_tile_grid,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), ri_img)

    if args.mask_out is not None:
        cv2.imwrite(str(args.mask_out), mask)

    if args.overlay_out is not None:
        save_overlay(gray, ri_img, mask, args.overlay_out)

    if args.depth_color_out is not None:
        save_depth_color(depth_img, mask, args.depth_color_out)

    print("saved ri-like intensity:", args.out)
    if args.mask_out:
        print("saved mask:", args.mask_out)
    if args.overlay_out:
        print("saved overlay:", args.overlay_out)
    if args.depth_color_out:
        print("saved camera depth color:", args.depth_color_out)

    print("valid pixels:", int(np.count_nonzero(mask)))
    print("intensity raw min/max:", float(np.min(points[:, 3])), float(np.max(points[:, 3])))
    finite_depth = depth_img[np.isfinite(depth_img)]
    if finite_depth.size > 0:
        print("KITTI camera depth Z min/max:", float(finite_depth.min()), float(finite_depth.max()))


if __name__ == "__main__":
    main()