"""Run Closing dense evaluation or prepare and evaluate a boundary-mask pilot."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from evaluate_retrieval_topk_pipeline import PYTHON, candidate_manifest, flag, read_csv, write_csv
from run_priority_validation374 import N_DB, N_QUERY, TOOLS, execute

W = Path("/mnt/w/its_outputs")
FILL = W / "validation374_fill_retrieval_FFm1ii"
VALID = W / "priority_validation374_d5nms6"
BASE = W / "priority_validation374_abc_gXMLfD/A"


def closing_dense(out, prepare_only):
    a = pd.read_parquet(BASE / "db.parquet")
    closing = pd.read_parquet(FILL / "closing/db.parquet")
    fields = ["pair_id", "db_seg", "anchor_id"]
    if len(a) != N_DB or not a[fields].equals(closing[fields]):
        raise ValueError("Closing and A DB order differs")
    topk = FILL / "closing/retrieval/closing_topk_top50.jsonl"
    remapped = out / "top10_original_A.jsonl"
    records = []
    for line in topk.read_text().splitlines():
        q = json.loads(line)
        q["topk"] = q["topk"][:10]
        for c in [q["strict_gt"], *q["topk"]]:
            i = int(c["db_index"])
            if not 0 <= i < N_DB or c["anchor_png"] != closing.iloc[i].anchor_png:
                raise ValueError(f"Closing path differs at DB index {i}")
            c["anchor_png"] = a.iloc[i].anchor_png
        records.append(json.dumps(q))
    content = "\n".join(records) + "\n"
    if remapped.exists() and remapped.read_text() != content:
        raise ValueError("Previously saved Top-10 differs")
    if not remapped.exists():
        remapped.write_text(content)
    _, _, rows = candidate_manifest(remapped, 10, N_QUERY)
    manifest = out / "top10.csv"
    if not manifest.exists():
        write_csv(manifest, rows)
    elif read_csv(manifest) != [{k: str(v) for k, v in r.items()} for r in rows]:
        raise ValueError("Previously saved candidate manifest differs")
    if prepare_only:
        print(f"Validated Closing manifest: {len(rows)} candidates", flush=True)
        return
    result = out / "dense/all507/all/all507_pnp/integrated_per_query.csv"
    if not result.exists():
        execute([PYTHON, "-u", TOOLS / "audit_priority_pnp_sources.py",
                 "--manifest", manifest, "--output", out / "dense",
                 "--scope", "all507", "--expected-queries", N_QUERY, "--all-only"])
    def by_query(path):
        return {(r["pair_id"], r["q_seg"], r["q_frame_index"]): r for r in read_csv(path)}
    before = by_query(BASE / "dense/all507/all/all507_pnp/integrated_per_query.csv")
    after = by_query(result)
    if before.keys() != after.keys() or len(before) != N_QUERY:
        raise ValueError("Dense query identities differ")
    report = []
    for meters in (.5, 1, 2, 5):
        field = f"success_xy_{meters:g}m"
        pairs = [(flag(before[k][field]), flag(after[k][field])) for k in before]
        report.append(dict(threshold_m=meters, A=sum(a for a, _ in pairs),
                           closing=sum(b for _, b in pairs),
                           improved=sum(not a and b for a, b in pairs),
                           worsened=sum(a and not b for a, b in pairs)))
    target = out / "comparison.csv"
    if not target.exists():
        write_csv(target, report)
    print(pd.DataFrame(report).to_string(index=False), flush=True)


def boundary(depth, valid, radius, spread):
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    low = cv2.erode(np.where(valid, depth, 1e6).astype(np.float32), kernel)
    high = cv2.dilate(np.where(valid, depth, -1e6).astype(np.float32), kernel)
    return (high - low) > spread


def region_pilot(out, radius, spread, audit_only, full=False, prepare_only=False):
    before = read_csv(BASE / "retrieval/A_per_query.csv")
    preserved = FILL / "preserve_A_png"
    after = read_csv(preserved / "retrieval/preserve_A_png_per_query.csv")
    key = "soft_t2m_hit@10"
    chosen, counts = [], {}
    for a, b in zip(before, after):
        if any(a[k] != b[k] for k in ("q_idx", "pair_id", "q_cam_png", "gt_index_db")):
            raise ValueError("Retrieval query identities differ")
        if flag(a[key]) == flag(b[key]):
            continue
        change = "improved" if flag(b[key]) else "worsened"
        group = a["pair_id"], change
        if counts.get(group, 0) < 2:
            chosen.append(dict(q_idx=int(a["q_idx"]), pair_id=a["pair_id"], change=change,
                               gt_index=int(a["gt_index_db"]), camera=a["q_cam_png"]))
            counts[group] = counts.get(group, 0) + 1
    if len(before) != N_QUERY or len(after) != N_QUERY or not chosen:
        raise ValueError("Unexpected query count")
    a = pd.read_parquet(BASE / "db.parquet")
    p = pd.read_parquet(preserved / "db.parquet")
    priority = pd.read_parquet(VALID / "database/db.parquet")
    fields = ["pair_id", "db_seg", "anchor_id"]
    if len(a) != N_DB or not all(a[fields].astype(str).equals(x[fields].astype(str)) for x in (p, priority)):
        raise ValueError("DB identities differ")
    images = out / "region_images"
    images.mkdir()
    overlays = out / "overlays"
    overlays.mkdir()
    region_paths, diagnostics = [], {}
    pixel_counts = []
    wanted = {r["gt_index"] for r in chosen}
    for i, (ap, pp, mp) in enumerate(zip(a.anchor_png, p.anchor_png, priority.anchor_png)):
        if audit_only and i not in wanted:
            continue
        with np.load(Path(ap).with_name("anchor_index.npz"), allow_pickle=False) as data:
            depth = data["depth"]
            valid = (data["point_index"] >= 0) & np.isfinite(depth)
        with np.load(Path(mp).with_name("anchor_index.npz"), allow_pickle=False) as data:
            generated = data["source_is_generated"]
        image = cv2.imread(pp, cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != valid.shape or generated.shape != valid.shape:
            raise ValueError(f"Invalid image or mask: {pp}")
        if np.any(valid & generated):
            raise ValueError(f"Generated mask overlaps measured raster: {mp}")
        blocked = generated & boundary(depth, valid, radius, spread)
        pixel_counts.append(dict(db_index=i, added=int(generated.sum()),
                                 blocked=int(blocked.sum()), kept=int((generated & ~blocked).sum())))
        target = images / f"{i:04d}.png"
        if i in wanted:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            overlay[generated & ~blocked] = (70, 220, 70)
            overlay[blocked] = (60, 60, 255)
            preview = cv2.resize(overlay, (864, 576), interpolation=cv2.INTER_AREA)
            dest = overlays / f"anchor_{i:04d}.png"
            if not cv2.imwrite(str(dest), preview):
                raise OSError(dest)
            diagnostics[i] = dict(overlay=str(dest), added=int(generated.sum()),
                                  boundary_removed=int(blocked.sum()))
        if not audit_only:
            image[blocked] = 0
            if not cv2.imwrite(str(target), image):
                raise OSError(target)
            region_paths.append(str(target))
            if full and ((i + 1) % 50 == 0 or i + 1 == N_DB):
                print(f"Rendered {i + 1}/{N_DB}", flush=True)
    for row in chosen:
        row.update(diagnostics[row["gt_index"]])
    write_csv(out / "selected_cases.csv", chosen)
    (out / "settings.json").write_text(json.dumps(dict(radius_px=radius, spread_m=spread), indent=2))
    print(f"Selected {len(chosen)} queries; overlays: {overlays}", flush=True)
    if audit_only:
        return
    db = p.assign(anchor_png=region_paths)
    if full:
        write_csv(out / "pixel_counts.csv", pixel_counts)
        region_full(out, db, region_paths, preserved, prepare_only)
        return
    selection = [r["q_idx"] for r in chosen]
    command = json.loads((VALID / "retrieval/command.json").read_text())
    comparisons = {}
    for name, folder in (("closing", FILL / "closing"), ("preserve_A_png", preserved), ("region", preserved)):
        dest = out / name
        dest.mkdir()
        pairs = pd.read_parquet(folder / "pairs.parquet").iloc[selection].copy()
        if list(pairs.cam_png) != [r["camera"] for r in chosen]:
            raise ValueError("Pilot query order differs")
        selected_db = db if name == "region" else pd.read_parquet(folder / "db.parquet")
        if name == "region":
            pairs["anchor_png"] = [region_paths[r["gt_index"]] for r in chosen]
        selected_db.to_parquet(dest / "db.parquet", index=False)
        pairs.to_parquet(dest / "pairs.parquet", index=False)
        cmd = command.copy()
        for option, value in (("--pairs", dest / "pairs.parquet"), ("--db", dest / "db.parquet"),
                              ("--out_dir", dest), ("--name", name)):
            cmd[cmd.index(option)+1] = str(value)
        execute(cmd)
        d = json.loads((dest / f"{name}_metrics.json").read_text())
        comparisons[name] = dict(condition=name, queries=len(chosen),
                                 **{f"soft{m}_hits": round(len(chosen) * d["soft"]["R_soft"][f"{m}.0"]["R@"]["10"])
                                    for m in (2, 5)})
    write_csv(out / "comparison.csv", list(comparisons.values()))


def region_full(out, db, paths, preserved, prepare_only):
    dest = out / "region"
    dest.mkdir()
    pairs = pd.read_parquet(preserved / "pairs.parquet")
    source_db = pd.read_parquet(preserved / "db.parquet")
    mapping = dict(zip(source_db.anchor_png, paths))
    mapped = pairs.anchor_png.map(mapping)
    if len(mapping) != N_DB or len(pairs) != N_QUERY or mapped.isna().any():
        raise ValueError("Unexpected validation DB or query mapping")
    pairs.assign(anchor_png=mapped).to_parquet(dest / "pairs.parquet", index=False)
    db.to_parquet(dest / "db.parquet", index=False)
    if prepare_only:
        return
    command = json.loads((VALID / "retrieval/command.json").read_text())
    for option, value in (("--pairs", dest / "pairs.parquet"), ("--db", dest / "db.parquet"),
                          ("--out_dir", dest), ("--name", "region")):
        command[command.index(option) + 1] = str(value)
    execute(command)
    sources = {
        "A": BASE / "retrieval/A_per_query.csv",
        "closing": FILL / "closing/retrieval/closing_per_query.csv",
        "observed_priority": VALID / "retrieval/observed_priority_per_query.csv",
        "preserve_A_png": preserved / "retrieval/preserve_A_png_per_query.csv",
        "region": dest / "region_per_query.csv",
    }
    keys = ("pair_id", "q_cam_png", "gt_index_db")
    frames = {name: {tuple(r[k] for k in keys): r for r in read_csv(path)}
              for name, path in sources.items()}
    reference = frames["A"].keys()
    if len(reference) != N_QUERY or any(frame.keys() != reference for frame in frames.values()):
        raise ValueError("Retrieval query identities differ")
    metrics = ("strict_hit@10", "soft_t2m_hit@10", "soft_t5m_hit@10")
    rows, transitions = [], []
    groups = {"ALL": list(reference)}
    for key in reference:
        groups.setdefault(key[0], []).append(key)
    for pair_id, group in groups.items():
        for name, frame in frames.items():
            rows.append(dict(condition=name, pair_id=pair_id, queries=len(group),
                             **{m: sum(flag(frame[k][m]) for k in group) for m in metrics}))
            if name == "region":
                continue
            for metric in metrics:
                changes = [(flag(frame[k][metric]), flag(frames["region"][k][metric])) for k in group]
                transitions.append(dict(before=name, after="region", pair_id=pair_id, metric=metric,
                                        improved=sum(not a and b for a, b in changes),
                                        worsened=sum(a and not b for a, b in changes)))
    write_csv(out / "comparison.csv", rows)
    write_csv(out / "transitions.csv", transitions)
    print(pd.DataFrame([r for r in rows if r["pair_id"] == "ALL"]).to_string(index=False))
    print(f"Results: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("closing-dense", "region-audit", "region-pilot", "region-full"))
    parser.add_argument("--prepare-only", action="store_true", help="Prepare Closing dense or region-full inputs without GPU")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--spread-m", type=float, default=1.)
    args = parser.parse_args()
    if args.radius < 1 or not np.isfinite(args.spread_m) or args.spread_m <= 0:
        parser.error("Invalid boundary settings")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.action == "closing-dense":
        closing_dense(out, args.prepare_only)
    else:
        region_pilot(out, args.radius, args.spread_m, args.action == "region-audit",
                     full=args.action == "region-full", prepare_only=args.prepare_only)


if __name__ == "__main__":
    main()
