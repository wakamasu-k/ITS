import argparse
from pathlib import Path

import cv2
import numpy as np


def load_points(path):
    """
    .bin または ASCII .ply から x,y,z,intensity を読む。

    KITTI .bin:
        float32 の [x, y, z, intensity]

    PLY:
        header の後に
        x y z intensity
        の4列がある想定
    """
    path = Path(path)

    if path.suffix == ".bin":
        points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
        return points[:, :4]

    if path.suffix == ".ply":
        with open(path, "r") as f:
            lines = f.readlines()

        end_header = None
        for i, line in enumerate(lines):
            if line.strip() == "end_header":
                end_header = i + 1
                break

        if end_header is None:
            raise ValueError(f"end_header not found: {path}")

        points = np.loadtxt(lines[end_header:], dtype=np.float32)

        if points.ndim == 1:
            points = points[None, :]

        if points.shape[1] < 4:
            raise ValueError(f"Expected [N,4] points, but got {points.shape}")

        return points[:, :4]

    raise ValueError(f"Unsupported file type: {path}")


def read_kitti_odometry_calib(calib_path, camera="P0"):
    """
    KITTI Odometry形式の calib.txt を読む。

    gray画像:
        image_0 -> P0
        image_1 -> P1

    color画像:
        image_2 -> P2
        image_3 -> P3
    """
    calib_path = Path(calib_path)
    calib = {}

    with open(calib_path, "r") as f:
        for line in f:
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            value = value.strip()

            if len(value) == 0:
                continue

            calib[key] = np.array(
                [float(x) for x in value.split()],
                dtype=np.float32
            )

    if camera not in calib:
        raise KeyError(f"{camera} not found in {calib_path}")

    if "Tr" not in calib:
        raise KeyError(f"Tr not found in {calib_path}")

    # P0, P1, P2, P3 は 3x4 の投影行列
    P = calib[camera].reshape(3, 4)

    # Tr は Velodyne LiDAR座標 -> Camera座標 の 3x4 行列
    Tr = calib["Tr"].reshape(3, 4)

    # 4x4 に拡張
    Tr_4x4 = np.eye(4, dtype=np.float32)
    Tr_4x4[:3, :] = Tr

    return P, Tr_4x4


def project_to_image(points_xyzi, calib_path, image_path, camera="P0"):
    """
    x,y,z,intensity 点群をカメラ画像平面へ投影して，
    カメラ画像形式の反射強度画像を作る。
    """
    image_path = Path(image_path)

    # 投影先画像を読む
    gray_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if gray_img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    height, width = gray_img.shape[:2]

    # calib読み込み
    P, Tr_4x4 = read_kitti_odometry_calib(
        calib_path=calib_path,
        camera=camera
    )

    xyz = points_xyzi[:, :3].astype(np.float32)
    intensity = points_xyzi[:, 3].astype(np.float32)

    # intensity は 0〜1 に丸める
    intensity = np.clip(intensity, 0.0, 1.0)

    # 同次座標にする
    ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
    xyz_h = np.hstack([xyz, ones])  # [N,4]

    # LiDAR座標 -> Camera座標
    cam_h = (Tr_4x4 @ xyz_h.T).T  # [N,4]

    # カメラの前方にある点だけ使う
    depth = cam_h[:, 2]
    front_mask = depth > 0.1

    cam_h = cam_h[front_mask]
    depth = depth[front_mask]
    intensity = intensity[front_mask]

    # Camera座標 -> 画像座標
    img_h = (P @ cam_h.T).T  # [N,3]

    u = img_h[:, 0] / img_h[:, 2]
    v = img_h[:, 1] / img_h[:, 2]

    # pixel座標に丸める
    u = np.round(u).astype(np.int32)
    v = np.round(v).astype(np.int32)

    # 画像範囲内の点だけ残す
    inside_mask = (
        (u >= 0) & (u < width) &
        (v >= 0) & (v < height)
    )

    u = u[inside_mask]
    v = v[inside_mask]
    depth = depth[inside_mask]
    intensity = intensity[inside_mask]

    # 出力画像
    intensity_img = np.zeros((height, width), dtype=np.float32)
    depth_img = np.full((height, width), np.inf, dtype=np.float32)
    mask_img = np.zeros((height, width), dtype=np.uint8)

    # Z-buffer:
    # 同じpixelに複数点が入った場合，カメラに近い点を採用する
    for px, py, dep, inten in zip(u, v, depth, intensity):
        if dep < depth_img[py, px]:
            depth_img[py, px] = dep
            intensity_img[py, px] = inten
            mask_img[py, px] = 255

    return intensity_img, mask_img, gray_img


def save_outputs(
    intensity_img,
    mask_img,
    gray_img,
    out_path,
    mask_out=None,
    overlay_out=None,
):
    """
    反射強度画像，mask画像，overlay画像を保存する。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # intensity画像を0〜255にして保存
    intensity_u8 = (np.clip(intensity_img, 0.0, 1.0) * 255).astype(np.uint8)
    cv2.imwrite(str(out_path), intensity_u8)

    # mask保存
    if mask_out is not None:
        mask_out = Path(mask_out)
        mask_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_out), mask_img)

    # overlay保存
    if overlay_out is not None:
        overlay_out = Path(overlay_out)
        overlay_out.parent.mkdir(parents=True, exist_ok=True)

        gray_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        color_intensity = cv2.applyColorMap(
            intensity_u8,
            cv2.COLORMAP_JET
        )

        overlay = gray_bgr.copy()

        valid = mask_img > 0
        overlay[valid] = color_intensity[valid]

        cv2.imwrite(str(overlay_out), overlay)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--points",
        required=True,
        help="input point cloud path: KITTI .bin or xyzi .ply"
    )
    parser.add_argument(
        "--calib",
        required=True,
        help="KITTI odometry calib.txt"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="target camera image"
    )
    parser.add_argument(
        "--camera",
        default="P0",
        choices=["P0", "P1", "P2", "P3"],
        help="projection matrix. image_0=P0, image_1=P1, image_2=P2, image_3=P3"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output intensity image path"
    )
    parser.add_argument(
        "--mask_out",
        default=None,
        help="output valid mask path"
    )
    parser.add_argument(
        "--overlay_out",
        default=None,
        help="output overlay image path"
    )

    args = parser.parse_args()

    points = load_points(args.points)

    print("points:", points.shape)
    print("xyz min:", points[:, :3].min(axis=0))
    print("xyz max:", points[:, :3].max(axis=0))
    print("intensity min/max:", points[:, 3].min(), points[:, 3].max())

    intensity_img, mask_img, gray_img = project_to_image(
        points_xyzi=points,
        calib_path=args.calib,
        image_path=args.image,
        camera=args.camera,
    )

    print("image shape:", gray_img.shape)
    print("projected pixels:", np.count_nonzero(mask_img))
    print("intensity image min/max:", intensity_img.min(), intensity_img.max())

    save_outputs(
        intensity_img=intensity_img,
        mask_img=mask_img,
        gray_img=gray_img,
        out_path=args.out,
        mask_out=args.mask_out,
        overlay_out=args.overlay_out,
    )

    print("saved intensity:", args.out)

    if args.mask_out is not None:
        print("saved mask:", args.mask_out)

    if args.overlay_out is not None:
        print("saved overlay:", args.overlay_out)


if __name__ == "__main__":
    main()
