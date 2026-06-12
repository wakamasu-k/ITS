#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_pairs_txt(path: Path):
    lines = [s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(lines) % 3 != 0:
        raise ValueError("pairs_txt は 3行1組: lane_tag, db_seg, q_seg である必要があります")
    out = []
    for i in range(0, len(lines), 3):
        out.append((lines[i], lines[i + 1], lines[i + 2]))
    return out


def seg_short(seg: str) -> str:
    m = re.match(r"segment-(\d{6})", seg)
    return m.group(1) if m else seg[:6]


def pair_id(db_seg: str, q_seg: str) -> str:
    return f"segpair_{seg_short(db_seg)}__{seg_short(q_seg)}"


def load_map_xy(maps_root: Path, subset: str, seg: str, sample_n: int, seed: int):
    p = maps_root / subset / seg / "map_static.npz"
    if not p.exists():
        raise FileNotFoundError(p)

    data = np.load(p)
    xyz = np.asarray(data["xyz"], dtype=np.float64)

    xyz = xyz[np.isfinite(xyz).all(axis=1)]

    # 極端な上下を少し落とす
    z = xyz[:, 2]
    lo, hi = np.percentile(z, [2, 98])
    xyz = xyz[(z >= lo) & (z <= hi)]

    if sample_n > 0 and len(xyz) > sample_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(xyz), size=sample_n, replace=False)
        xyz = xyz[idx]

    # SE(2) ICP用に z=0 にする
    pts = np.zeros((len(xyz), 3), dtype=np.float64)
    pts[:, 0] = xyz[:, 0]
    pts[:, 1] = xyz[:, 1]
    pts[:, 2] = 0.0
    return pts


def to_pcd(points: np.ndarray, voxel: float):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    return pcd


def rotz(deg: float):
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    T = np.eye(4)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    return T


def estimate_icp(db_pts, q_pts, voxel, threshold, yaw_step_deg, max_iter):
    src = to_pcd(db_pts, voxel)
    tgt = to_pcd(q_pts, voxel)

    src_np = np.asarray(src.points)
    tgt_np = np.asarray(tgt.points)

    if len(src_np) == 0 or len(tgt_np) == 0:
        raise RuntimeError("empty point cloud after downsampling")

    cs = src_np.mean(axis=0)
    ct = tgt_np.mean(axis=0)

    best = None

    yaws = np.arange(-180, 180, yaw_step_deg, dtype=float)

    for yaw0 in yaws:
        T0 = rotz(float(yaw0))
        R0 = T0[:3, :3]
        T0[:3, 3] = ct - R0 @ cs

        reg = o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            threshold,
            T0,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iter)),
        )

        score = (float(reg.fitness), -float(reg.inlier_rmse))
        if best is None or score > best[0]:
            best = (score, reg)

    reg = best[1]
    T = np.asarray(reg.transformation, dtype=np.float64)

    yaw = math.degrees(math.atan2(T[1, 0], T[0, 0]))
    tx = float(T[0, 3])
    ty = float(T[1, 3])
    tz = float(T[2, 3])

    return yaw, tx, ty, tz, float(reg.fitness), float(reg.inlier_rmse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_txt", required=True)
    ap.add_argument("--maps_root", required=True)
    ap.add_argument("--subset", default="training")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--sample_n", type=int, default=30000)
    ap.add_argument("--voxel", type=float, default=1.0)
    ap.add_argument("--threshold_m", type=float, default=3.0)
    ap.add_argument("--yaw_step_deg", type=float, default=15.0)
    ap.add_argument("--max_iter", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = parse_pairs_txt(Path(args.pairs_txt))
    maps_root = Path(args.maps_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for lane_tag, db_seg, q_seg in pairs:
        pid = pair_id(db_seg, q_seg)
        print("=" * 80)
        print("[PAIR]", pid)
        print("DB:", db_seg)
        print("Q :", q_seg)

        db_pts = load_map_xy(maps_root, args.subset, db_seg, args.sample_n, args.seed)
        q_pts = load_map_xy(maps_root, args.subset, q_seg, args.sample_n, args.seed + 1)

        yaw, tx, ty, tz, fitness, rmse = estimate_icp(
            db_pts,
            q_pts,
            voxel=args.voxel,
            threshold=args.threshold_m,
            yaw_step_deg=args.yaw_step_deg,
            max_iter=args.max_iter,
        )

        print(f"[ICP] yaw={yaw:.6f}, tx={tx:.6f}, ty={ty:.6f}, tz={tz:.6f}, fitness={fitness:.6f}, rmse={rmse:.6f}")

        rows.append({
            "pair_id": pid,
            "lane_tag": lane_tag,
            "db_seg": db_seg,
            "q_seg": q_seg,
            "final_yaw_deg": yaw,
            "final_tx": tx,
            "final_ty": ty,
            "final_tz": tz,
            "fitness": fitness,
            "inlier_rmse": rmse,
        })

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("[OK] wrote:", out_csv)


if __name__ == "__main__":
    main()
