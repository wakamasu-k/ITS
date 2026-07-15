import numpy as np
from pathlib import Path
import argparse


def save_xyzi_ply(points, out_path):
    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"points must be [N,4], but got {points.shape}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float intensity\n")
        f.write("end_header\n")
        np.savetxt(f, points, fmt="%.6f %.6f %.6f %.6f")


def range_intensity_to_xyzi(img):
    """
    img: [64, 1024, 2]
      img[..., 0] = range [m]
      img[..., 1] = intensity
    """

    if img.ndim != 3 or img.shape[-1] < 2:
        raise ValueError(f"Expected [H,W,2], but got {img.shape}")

    image_rows, image_cols, _ = img.shape

    range_img = img[..., 0].astype(np.float32)
    intensity_img = img[..., 1].astype(np.float32)

    # sample_kitti_dataset.py と同じ角度設定
    ang_start_y = 24.8
    ang_res_y = 26.8 / (image_rows - 1)
    ang_res_x = 360.0 / image_cols

    rows, cols = np.indices((image_rows, image_cols))

    # create_range_map の逆変換
    vertical_angle_deg = rows * ang_res_y - ang_start_y
    horizontal_angle_deg = 90.0 - (cols - image_cols / 2.0) * ang_res_x

    vertical_angle = np.deg2rad(vertical_angle_deg)
    horizontal_angle = np.deg2rad(horizontal_angle_deg)

    r = range_img

    # 無効点を除去
    valid = np.isfinite(r) & (r > 0.0)

    # 元コードは arctan2(x, y) で水平角を作っているので，
    # x = r_xy * sin(h), y = r_xy * cos(h)
    r_xy = r * np.cos(vertical_angle)
    x = r_xy * np.sin(horizontal_angle)
    y = r_xy * np.cos(horizontal_angle)
    z = r * np.sin(vertical_angle)

    points = np.column_stack([
        x[valid],
        y[valid],
        z[valid],
        intensity_img[valid],
    ])

    return points.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="./kitti_utils/kitti_train/00000000.npy",
        help="Input .npy file [64,1024,2]",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./debug_outputs/gt_xyzi_000000.ply",
        help="Output .ply path",
    )
    args = parser.parse_args()

    img = np.load(args.input)

    print("input:", args.input)
    print("shape:", img.shape)
    print("dtype:", img.dtype)
    print("range min/max:", img[..., 0].min(), img[..., 0].max())
    print("intensity min/max:", img[..., 1].min(), img[..., 1].max())
    print("nonzero range:", np.count_nonzero(img[..., 0]))
    print("nonzero intensity:", np.count_nonzero(img[..., 1]))

    points = range_intensity_to_xyzi(img)

    print("points shape:", points.shape)
    print("x min/max:", points[:, 0].min(), points[:, 0].max())
    print("y min/max:", points[:, 1].min(), points[:, 1].max())
    print("z min/max:", points[:, 2].min(), points[:, 2].max())
    print("intensity min/max:", points[:, 3].min(), points[:, 3].max())

    save_xyzi_ply(points, args.output)
    print("saved:", args.output)


if __name__ == "__main__":
    main()
