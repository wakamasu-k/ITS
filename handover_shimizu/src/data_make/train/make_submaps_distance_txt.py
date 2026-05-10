# 使い方:
#   python make_submaps_distance_txt.py --help
#   必要な入力パスと出力先を引数で指定して実行する。

"""
距離ベースで Waymo セグメントから submap / anchor のメタ情報を生成するスクリプト。

TFRecord またはセグメント一覧テキストを入力に取り、各フレーム姿勢と
submap 割り当てを CSV / JSON として保存する。
"""


import sys
import json
import math
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset


DEFAULT_OUT_ROOT = "/mnt/e/waymo/data/submaps"
DEFAULT_TFRECORD_ROOT = "/mnt/e/waymo_tf/perception_v1.4.3/individual_files"


def ensure_dir(p: Path) -> None:
    """出力先ディレクトリを作成して存在を保証する。"""
    p.mkdir(parents=True, exist_ok=True)


def _read_nonempty_lines(txt_path: Path) -> List[str]:
    """txtを1行ずつ読み、空行・コメント行（#で開始）を除いて返す。"""
    lines: List[str] = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    continue
                lines.append(s)
    except UnicodeDecodeError:
        # 万一BOM付きなどだった場合の保険
        with open(txt_path, "r", encoding="utf-8-sig") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    continue
                lines.append(s)
    # 重複は安易に残すと同じ出力を何度も踏むので、順序を保って除去
    uniq: List[str] = []
    seen = set()
    for s in lines:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _looks_like_tfrecord_path(s: str) -> bool:
    """文字列が TFRecord パス表記かどうかを判定する。"""
    ls = s.lower()
    return ls.endswith(".tfrecord") or ls.endswith(".tfrecord.gz")


def resolve_tfrecord_path(
    item: str,
    subset: str,
    tfrecord_root: Path,
    txt_dir: Optional[Path] = None,
) -> Path:
    """
    txt内の1行（item）を tfrecord の実パスに解決する。
    解決順:
      1) item が絶対パスで存在
      2) txt_dir/item が存在（相対パス対応）
      3) item が相対パスで存在（カレントディレクトリ基準）
      4) stem とみなして DEFAULT_TFRECORD_ROOT/subset/<stem>.tfrecord を探索
      5) item が .tfrecord 名（存在しない）なら DEFAULT_TFRECORD_ROOT/subset/<name> を探索
    """
    s = item.strip()
    if not s:
        raise FileNotFoundError("Empty line")

    p = Path(s).expanduser()

    # 1) 絶対パスで存在
    if p.is_absolute() and p.exists():
        return p

    # 2) txt_dir基準の相対パス
    if txt_dir is not None:
        cand = (txt_dir / p).expanduser()
        if cand.exists():
            return cand.resolve()

    # 3) カレントディレクトリ基準の相対パス
    if p.exists():
        return p.resolve()

    # 4) stem から構築（.tfrecord 付与）
    tfroot = Path(tfrecord_root)
    if _looks_like_tfrecord_path(s):
        name = Path(s).name  # ファイル名だけ抽出
    else:
        name = f"{s}.tfrecord"

    cand2 = tfroot / subset / name
    if cand2.exists():
        return cand2.resolve()

    raise FileNotFoundError(
        "tfrecord が見つかりません。\n"
        f"  item: {item}\n"
        f"  tried: {p}\n"
        f"  tried: {cand2}\n"
        "  ヒント: txtにフルパスを書くか、--tfrecord-root が正しいか確認してください。"
    )


def mat4_list_to_np(m) -> np.ndarray:
    """行優先の 16 要素配列を 4x4 の NumPy 行列に変換する。"""
    return np.array(m, dtype=np.float64).reshape(4, 4)


def yaw_from_T_wv(T_wv: np.ndarray) -> float:
    """
    world<-vehicle の回転から水平ヨー角[deg]を抽出
    yaw = atan2(R[1,0], R[0,0]) を [-180,180] に正規化
    """
    R = T_wv[:3, :3]
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    while yaw > 180.0:
        yaw -= 360.0
    while yaw < -180.0:
        yaw += 360.0
    return float(yaw)


def yaw_diff_deg(a: float, b: float) -> float:
    """角度差の絶対値（-180..180 wrap込み）"""
    d = a - b
    while d > 180.0:
        d -= 360.0
    while d < -180.0:
        d += 360.0
    return abs(d)


def load_frames_poses(tfrecord_path: Path) -> pd.DataFrame:
    """
    セグメントの全フレームから (frame_index, timestamp_us, x, y, yaw_deg) を取得
    """
    ds = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")
    xs, ys, yaws, ts, idxs = [], [], [], [], []

    for i, rec in enumerate(ds):
        fr = open_dataset.Frame()
        fr.ParseFromString(rec.numpy())
        T_wv = mat4_list_to_np(fr.pose.transform)  # world<-vehicle
        x, y = float(T_wv[0, 3]), float(T_wv[1, 3])
        yaw = yaw_from_T_wv(T_wv)
        xs.append(x)
        ys.append(y)
        yaws.append(yaw)
        ts.append(int(fr.timestamp_micros))
        idxs.append(i)

    df = pd.DataFrame(
        {
            "frame_index": idxs,
            "timestamp_us": ts,
            "x": xs,
            "y": ys,
            "yaw_deg": yaws,
        }
    )
    return df


def cumulative_distance(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    連続フレーム間の2D距離を累積した配列（先頭は0）
    """
    if len(xs) == 0:
        return np.zeros((0,), dtype=np.float64)
    dx = xs[1:] - xs[:-1]
    dy = ys[1:] - ys[:-1]
    step = np.sqrt(dx * dx + dy * dy)
    cum = np.zeros((len(xs),), dtype=np.float64)
    cum[1:] = np.cumsum(step)
    return cum


def _maybe_add_yaw_jump_anchors(
    df_pose: pd.DataFrame,
    anchors: List[int],
    yaw_jump_deg: float,
    min_anchor_sep_m: float = 2.0,
) -> List[int]:
    """
    フレーム間yaw差が閾値を超える点を補助アンカー候補として追加。
    ただし既存アンカーから近すぎる候補は追加しない。
    """
    if yaw_jump_deg <= 0.0:
        return anchors

    idxs = df_pose["frame_index"].to_numpy()
    xs = df_pose["x"].to_numpy()
    ys = df_pose["y"].to_numpy()
    yaws = df_pose["yaw_deg"].to_numpy()

    # 急変フレーム候補（i=1..N-1 の i を候補にする）
    yaw_d = np.zeros_like(yaws, dtype=np.float64)
    for i in range(1, len(yaws)):
        yaw_d[i] = float(yaw_diff_deg(float(yaws[i]), float(yaws[i - 1])))

    candidates = np.where(yaw_d >= float(yaw_jump_deg))[0].tolist()

    if len(candidates) == 0:
        return anchors

    # 既存アンカー位置を座標で集める
    a_points = []
    for a in anchors:
        k = int(np.searchsorted(idxs, a))
        k = min(max(k, 0), len(xs) - 1)
        a_points.append((float(xs[k]), float(ys[k])))
    a_points = np.array(a_points, dtype=np.float64) if len(a_points) > 0 else None

    for ci in candidates:
        cx, cy = float(xs[ci]), float(ys[ci])

        if a_points is None or len(a_points) == 0:
            anchors.append(int(idxs[ci]))
            a_points = np.array([(cx, cy)], dtype=np.float64)
            continue

        dmin = float(np.min(np.sqrt((a_points[:, 0] - cx) ** 2 + (a_points[:, 1] - cy) ** 2)))
        if dmin >= float(min_anchor_sep_m):
            anchors.append(int(idxs[ci]))
            a_points = np.vstack([a_points, np.array([[cx, cy]], dtype=np.float64)])

    anchors = sorted(list(set(int(a) for a in anchors)))
    return anchors


def pick_anchors_grid(df_pose: pd.DataFrame, spacing_m: float, yaw_jump_deg: float) -> List[int]:
    """
    旧方式: 累積距離の等間隔ターゲット（0, spacing, 2*spacing...）に最近いフレームをアンカーにする。
    """
    if spacing_m <= 0:
        raise ValueError("spacing_m must be > 0")

    idxs = df_pose["frame_index"].to_numpy()
    xs = df_pose["x"].to_numpy()
    ys = df_pose["y"].to_numpy()

    cum = cumulative_distance(xs, ys)
    path_len = float(cum[-1])

    # ターゲット距離列
    num_targets = int(math.floor(path_len / spacing_m)) + 1
    targets = np.array([i * spacing_m for i in range(num_targets)], dtype=np.float64)

    anchors: List[int] = []
    for t in targets:
        j = int(np.argmin(np.abs(cum - t)))
        anchors.append(int(idxs[j]))

    anchors = sorted(list(set(anchors)))

    # yaw jump 補助
    anchors = _maybe_add_yaw_jump_anchors(df_pose, anchors, yaw_jump_deg=yaw_jump_deg)
    return anchors


def pick_anchors_greedy(df_pose: pd.DataFrame, spacing_m: float, yaw_jump_deg: float) -> List[int]:
    """
    今回の方式（学習用アンカー設計A）:
      最初のフレームをアンカーにし、
      直近アンカーからの移動距離（フレーム間距離の累積）が spacing_m を超えた最初のフレームを次アンカーにする。
    """
    if spacing_m <= 0:
        raise ValueError("spacing_m must be > 0")

    idxs = df_pose["frame_index"].to_numpy()
    xs = df_pose["x"].to_numpy()
    ys = df_pose["y"].to_numpy()

    if len(idxs) == 0:
        return []

    anchors: List[int] = [int(idxs[0])]

    # フレーム間距離
    dx = xs[1:] - xs[:-1]
    dy = ys[1:] - ys[:-1]
    step = np.sqrt(dx * dx + dy * dy)  # len = N-1

    acc = 0.0
    for i in range(1, len(idxs)):
        acc += float(step[i - 1])
        if acc >= float(spacing_m):
            anchors.append(int(idxs[i]))
            acc = 0.0

    anchors = sorted(list(set(anchors)))

    # yaw jump 補助
    anchors = _maybe_add_yaw_jump_anchors(df_pose, anchors, yaw_jump_deg=yaw_jump_deg)
    return anchors


def pick_anchors(df_pose: pd.DataFrame, spacing_m: float, yaw_jump_deg: float, anchor_mode: str) -> List[int]:
    """距離条件や間引き条件に従ってアンカー候補を選ぶ。"""
    if anchor_mode == "greedy":
        return pick_anchors_greedy(df_pose, spacing_m=spacing_m, yaw_jump_deg=yaw_jump_deg)
    if anchor_mode == "grid":
        return pick_anchors_grid(df_pose, spacing_m=spacing_m, yaw_jump_deg=yaw_jump_deg)
    raise ValueError(f"Unknown anchor_mode: {anchor_mode}")


def assign_frames_to_anchors(df_pose: pd.DataFrame, anchors: List[int]) -> np.ndarray:
    """
    各フレームを「最も近いアンカー」に割り当てる（2Dユークリッド距離）
    返り値: frame_index順に対応する submap_id (0..M-1) の配列
    """
    xs = df_pose["x"].to_numpy()
    ys = df_pose["y"].to_numpy()
    idxs = df_pose["frame_index"].to_numpy()

    # anchor の行番号
    a_idx_to_row = {int(fi): int(np.searchsorted(idxs, fi)) for fi in anchors}
    a_rows = np.array([a_idx_to_row[fi] for fi in anchors], dtype=int)
    a_xy = np.stack([xs[a_rows], ys[a_rows]], axis=1)  # (M,2)

    f_xy = np.stack([xs, ys], axis=1)  # (N,2)
    diff = f_xy[:, None, :] - a_xy[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    assign = np.argmin(dist, axis=1)
    return assign  # shape (N,)


def _anchor_gap_stats(df_pose: pd.DataFrame, anchors: List[int]) -> Tuple[float, float, float]:
    """
    アンカー間隔（軌跡の累積距離差）で min/median/max を返す
    """
    if len(anchors) <= 1:
        return 0.0, 0.0, 0.0
    idxs = df_pose["frame_index"].to_numpy()
    xs = df_pose["x"].to_numpy()
    ys = df_pose["y"].to_numpy()
    cum = cumulative_distance(xs, ys)

    a_rows = [int(np.searchsorted(idxs, a)) for a in anchors]
    a_cum = cum[np.array(a_rows, dtype=int)]
    gaps = np.diff(a_cum)
    if len(gaps) == 0:
        return 0.0, 0.0, 0.0
    return float(np.min(gaps)), float(np.median(gaps)), float(np.max(gaps))


def save_outputs(
    out_dir: Path,
    subset: str,
    seg_stem: str,
    df_pose: pd.DataFrame,
    anchors: List[int],
    assign_ids: np.ndarray,
    spacing_m: float,
    r_sub_m: float,
    yaw_jump_deg: float,
    anchor_mode: str,
) -> None:
    """生成した CSV / JSON などの出力群をまとめて保存する。"""
    ensure_dir(out_dir)

    # 1) 各フレームの姿勢（デバッグ用）
    df_pose.to_csv(out_dir / "frames_poses.csv", index=False)

    # 2) 各フレーム→最も近いサブマップID
    df_map = pd.DataFrame(
        {
            "frame_index": df_pose["frame_index"].to_numpy(),
            "submap_id": assign_ids.astype(int),
        }
    )
    df_map.to_csv(out_dir / "frame_to_submap.csv", index=False)

    # 2.5) anchors.csv（追加）
    idxs = df_pose["frame_index"].to_numpy()
    xs = df_pose["x"].to_numpy()
    ys = df_pose["y"].to_numpy()
    yaws = df_pose["yaw_deg"].to_numpy()
    ts = df_pose["timestamp_us"].to_numpy()
    cum = cumulative_distance(xs, ys)

    a_rows = [int(np.searchsorted(idxs, a)) for a in anchors]
    df_anchors = pd.DataFrame(
        {
            "submap_id": np.arange(len(anchors), dtype=int),
            "anchor_frame_index": np.array(anchors, dtype=int),
            "timestamp_us": ts[np.array(a_rows, dtype=int)].astype(np.int64),
            "x": xs[np.array(a_rows, dtype=int)].astype(np.float64),
            "y": ys[np.array(a_rows, dtype=int)].astype(np.float64),
            "yaw_deg": yaws[np.array(a_rows, dtype=int)].astype(np.float64),
            "cum_dist_m": cum[np.array(a_rows, dtype=int)].astype(np.float64),
        }
    )
    df_anchors.to_csv(out_dir / "anchors.csv", index=False)

    # 3) submaps.json を保存する
    neighbors = {}
    M = len(anchors)
    for m in range(M):
        nbs = []
        if m - 1 >= 0:
            nbs.append(m - 1)
        if m + 1 < M:
            nbs.append(m + 1)
        neighbors[m] = nbs

    submaps = []
    for m in range(M):
        members = idxs[assign_ids == m].tolist()
        ar = a_rows[m]
        submaps.append(
            {
                "submap_id": m,
                "anchor_frame_index": int(anchors[m]),
                "anchor_pose_xyyaw": {
                    "x": float(xs[ar]),
                    "y": float(ys[ar]),
                    "yaw_deg": float(yaws[ar]),
                },
                "member_frame_indices": [int(v) for v in members],
                "neighbor_submap_ids": neighbors[m],
            }
        )

    gap_min, gap_med, gap_max = _anchor_gap_stats(df_pose, anchors)

    meta = {
        "subset": subset,
        "segment_id": seg_stem,
        "num_frames": int(len(df_pose)),
        "num_submaps": int(M),
        "spacing_m": float(spacing_m),
        "submap_radius_m": float(r_sub_m),
        "yaw_jump_deg": float(yaw_jump_deg),
        "anchor_mode": str(anchor_mode),
        "anchor_gap_stats_m": {
            "min": float(gap_min),
            "median": float(gap_med),
            "max": float(gap_max),
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    obj = {"meta": meta, "submaps": submaps}

    with open(out_dir / "submaps.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

    print(f"[OK] submaps.json を保存: {out_dir}")
    print(f"     サブマップ数: {M}  | フレーム数: {len(df_pose)}")
    if M > 0:
        print(f"     anchor_mode={anchor_mode}")
        print(f"     アンカー間隔(軌跡距離) min/med/max = {gap_min:.3f} / {gap_med:.3f} / {gap_max:.3f} [m]")
        print(f"     最初のアンカー frame_index: {anchors[0]}, 位置(x,y)=({xs[a_rows[0]]:.2f},{ys[a_rows[0]]:.2f})")
        print(f"     anchors.csv を保存: {out_dir / 'anchors.csv'}")


def _process_one_tfrecord(
    tfrecord_path: Path,
    args: argparse.Namespace,
    strict: bool,
) -> str:
    """
    1つのtfrecordを従来通り処理する。
    strict=True の場合は（従来と同様に）致命的エラーで即終了する。
    返り値: "ok" / "skip" / "err"
    """
    if not tfrecord_path.exists():
        print(f"[ERR] tfrecord が見つかりません: {tfrecord_path}")
        if strict:
            sys.exit(1)
        return "err"

    seg_stem = tfrecord_path.stem
    out_dir = Path(args.out_root) / args.subset / seg_stem
    submaps_path = out_dir / "submaps.json"

    # 既存チェック（レジューム対応）
    if submaps_path.exists():
        if args.overwrite:
            print(f"[INFO] 既存の submaps.json を上書きします: {submaps_path}")
        else:
            print(f"[SKIP] 既に submaps.json が存在するためスキップします: {submaps_path}")
            return "skip"

    ensure_dir(out_dir)

    print(f"[INFO] 読み込み: {tfrecord_path}")
    print(f"[INFO] 出力先:  {out_dir}")
    print(
        f"[INFO] anchor_mode={args.anchor_mode}, spacing={args.spacing_m} m, R_sub={args.r_sub_m} m, yaw_jump={args.yaw_jump_deg} deg"
    )

    try:
        # 1) フレーム姿勢の読み出し
        df_pose = load_frames_poses(tfrecord_path)
        if df_pose.empty:
            print("[ERR] フレームが空でした。")
            if strict:
                sys.exit(1)
            return "err"

        # 2) アンカー抽出
        anchors = pick_anchors(
            df_pose,
            spacing_m=float(args.spacing_m),
            yaw_jump_deg=float(args.yaw_jump_deg),
            anchor_mode=str(args.anchor_mode),
        )
        if len(anchors) == 0:
            anchors = [int(df_pose["frame_index"].iloc[0])]

        # 3) 最近接アンカーへのフレーム割り当て
        assign_ids = assign_frames_to_anchors(df_pose, anchors)

        # 4) 保存
        save_outputs(
            out_dir=out_dir,
            subset=str(args.subset),
            seg_stem=seg_stem,
            df_pose=df_pose,
            anchors=anchors,
            assign_ids=assign_ids,
            spacing_m=float(args.spacing_m),
            r_sub_m=float(args.r_sub_m),
            yaw_jump_deg=float(args.yaw_jump_deg),
            anchor_mode=str(args.anchor_mode),
        )
        return "ok"

    except SystemExit:
        # strict=True の場合はここまで来ない想定だが、念のため
        raise
    except Exception as e:
        print(f"[ERR] 処理に失敗: {tfrecord_path}")
        print(f"      {type(e).__name__}: {e}")
        if strict:
            sys.exit(1)
        return "err"


def main() -> None:
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    ap = argparse.ArgumentParser(description="Step 1: 距離ベースのサブマップ設計（アンカー抽出）")
    ap.add_argument(
        "--tfrecord",
        type=str,
        required=True,
        help=(
            "Waymo セグメント .tfrecord または、tfrecordリスト .txt（1行1セグメント）。\n"
            "txtの各行は tfrecordのフルパス / 相対パス / セグメントstem を許可。"
        ),
    )
    ap.add_argument(
        "--tfrecord-root",
        type=str,
        default=DEFAULT_TFRECORD_ROOT,
        help=(
            "txt内がstemの場合に探索するWaymo individual_filesのルート（既定: "
            f"{DEFAULT_TFRECORD_ROOT}）。\n"
            "例: <root>/<subset>/<stem>.tfrecord"
        ),
    )
    ap.add_argument(
        "--out-root",
        type=str,
        default=DEFAULT_OUT_ROOT,
        help=f"出力ルート（既定: {DEFAULT_OUT_ROOT}）",
    )
    ap.add_argument("--subset", type=str, choices=["training", "validation", "testing"], required=True)
    ap.add_argument("--spacing-m", type=float, default=1.0, help="アンカー間隔[m]（既定: 1.0）")
    ap.add_argument(
        "--r-sub-m",
        type=float,
        default=10.0,
        help="サブマップ半径[m]（定義のみ。後段のΔpose判定などで使用）",
    )
    ap.add_argument(
        "--yaw-jump-deg",
        type=float,
        default=0.0,
        help="yaw急変で補助アンカーを入れるしきい値[deg]（0以下で無効。既定: 0）",
    )
    ap.add_argument(
        "--anchor-mode",
        type=str,
        choices=["greedy", "grid"],
        default="greedy",
        help="アンカー抽出方式（greedy / grid）",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="出力先に既存 submaps.json があっても上書きする（既定: False。False の場合は既存があればスキップ）",
    )
    args = ap.parse_args()

    tfrecord_arg = Path(args.tfrecord)
    if not tfrecord_arg.exists():
        print(f"[ERR] 入力が見つかりません: {tfrecord_arg}")
        sys.exit(1)

    # txt指定ならバッチ処理（失敗しても続行し、最後にサマリ＋非ゼロ終了）
    is_list_mode = tfrecord_arg.suffix.lower() == ".txt"
    strict = not is_list_mode

    tfrecord_root = Path(args.tfrecord_root)

    if is_list_mode:
        items = _read_nonempty_lines(tfrecord_arg)
        if len(items) == 0:
            print(f"[ERR] txt が空です: {tfrecord_arg}")
            sys.exit(1)

        print(f"[INFO] list mode: {tfrecord_arg}")
        print(f"[INFO] entries: {len(items)}")
        print(f"[INFO] tfrecord_root: {tfrecord_root}")
        print(f"[INFO] subset: {args.subset}")

        ok = 0
        skip = 0
        err = 0

        for i, item in enumerate(items):
            print("".join(["\n", "=" * 80]))
            print(f"[INFO] ({i + 1}/{len(items)}) item: {item}")

            try:
                tfrecord_path = resolve_tfrecord_path(
                    item=item,
                    subset=str(args.subset),
                    tfrecord_root=tfrecord_root,
                    txt_dir=tfrecord_arg.parent,
                )
            except Exception as e:
                print(f"[ERR] tfrecord 解決に失敗: {item}")
                print(f"      {type(e).__name__}: {e}")
                err += 1
                continue

            status = _process_one_tfrecord(tfrecord_path=tfrecord_path, args=args, strict=False)
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1

        print("".join(["\n", "=" * 80]))
        print("[INFO] finished list mode")
        print(f"[INFO] ok={ok}, skip={skip}, err={err}")

        if err > 0:
            sys.exit(1)
        return

    # 従来通り: 単一tfrecord
    _process_one_tfrecord(tfrecord_path=tfrecord_arg, args=args, strict=True)


if __name__ == "__main__":
    main()
