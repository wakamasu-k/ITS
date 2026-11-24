"""Reproduce the observed16/TULIP PnP comparison on the 374-query validation split."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from evaluate_retrieval_topk_pipeline import PYTHON, candidate_manifest, read_csv, write_csv

PROJECT = Path(__file__).resolve().parents[1]
TOOLS = PROJECT / "tools"
REF = PROJECT.parent / "handover_shimizu/src"
SOURCE = PROJECT / "outputs/original_tulip_B_retraining_20261029"
B_MAPS = SOURCE / "data/val/maps/B_tulip32"
TEST = PROJECT / "outputs/shimizu_protocol_507q_754db_abc/retrieval_manifests"
N_QUERY = 374
N_DB = 443


def execute(command, env=None):
    print("RUN:", *command, flush=True)
    subprocess.run(list(map(str, command)), env=env, check=True)


def source_tables():
    import pandas as pd

    db = pd.read_parquet(SOURCE / "val_db_source.parquet")
    pairs = pd.read_parquet(SOURCE / "val_pairs_source.parquet")
    if (len(pairs), len(db)) != (N_QUERY, N_DB):
        raise ValueError("Validation split size differs")
    segments = set(db.db_seg) | set(pairs.q_seg)
    test_pairs = pd.read_csv(TEST / "A_pairs.csv")
    if segments & (set(test_pairs.db_seg) | set(test_pairs.q_seg)):
        raise ValueError("The validation and 507-query test segments overlap")
    jobs = [j for j in json.loads((SOURCE / "plan.json").read_text())["jobs"]
            if j["role"] == "val"]
    if {j["segment"] for j in jobs} != set(db.db_seg):
        raise ValueError("Validation map segments differ")
    for table in (db, pairs):
        table["anchor_id"] = table["anchor_id"].map(int)
        table["anchor_png"] = [
            str(SOURCE / "data/val/anchors" / r.db_subset / r.db_seg / str(r.anchor_id) / "anchor.png")
            for r in table.itertuples()]
    for png in set(db.anchor_png):
        path = Path(png)
        for candidate in (path, path.with_name("anchor.json"), path.with_name("anchor_index.npz"),
                          path.parent.parent / "anchors.csv"):
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
    if not set(pairs.anchor_png) <= set(db.anchor_png):
        raise ValueError("A validation GT anchor is missing in the DB")
    for png in set(pairs.cam_png):
        for path in (Path(png), Path(png).with_suffix(".json")):
            if not path.is_file():
                raise FileNotFoundError(path)
    for job in jobs:
        segment = Path(job["subset"]) / job["segment"]
        for path in (Path(job["tfrecord"]),
                     B_MAPS / segment / "map_static.npz",
                     SOURCE / "data/val/anchors" / segment / "submaps.json"):
            if not path.is_file():
                raise FileNotFoundError(path)
    print(f"Validated inputs: {len(pairs)} queries, {len(db)} anchors, {len(jobs)} map segments")
    return db, pairs, jobs


def render(root):
    import numpy as np

    db, pairs, jobs = source_tables()
    if {j["subset"] for j in jobs} != {"training"}:
        raise ValueError("Expected the validation map segments under training/")
    maps = root / "maps"
    maps.mkdir(parents=True, exist_ok=True)
    anchors = root / "anchors_A"
    for job in jobs:
        segment = Path(job["subset"]) / job["segment"]
        a_map = maps / "A_observed16" / segment / "map_static.npz"
        if not a_map.is_file():
            a_map.parent.mkdir(parents=True, exist_ok=True)
            with np.load(B_MAPS / segment / a_map.name, allow_pickle=False) as data:
                measured = ~data["source_is_generated"].reshape(-1).astype(bool)
                xyz = data["xyz"][measured]
                intensity = data["intensity"].reshape(-1)[measured]
            temporary = a_map.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, xyz=xyz, intensity=intensity)
            temporary.replace(a_map)
        layout = anchors / segment / "submaps.json"
        layout.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / "data/val/anchors" / segment / layout.name, layout)
        execute([sys.executable, REF / "data_make/valid_test/render_anchor_intensity_txt_se3_fix.py",
                 "--tfrecord", job["tfrecord"], "--subset", job["subset"],
                 "--maps-root", maps / "A_observed16", "--submaps-root", anchors,
                 "--crop-mode", "none", "--point-size", "2", "--write-index", "true"],
                dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TF_NUM_INTRAOP_THREADS="2",
                     TF_NUM_INTEROP_THREADS="2"))
    manifests = root / "retrieval_manifests"
    manifests.mkdir(exist_ok=True)
    db.to_csv(manifests / "B_db.csv", index=False)
    pairs.to_csv(manifests / "B_pairs.csv", index=False)
    a = db.copy()
    a["anchor_png"] = [
        str(anchors / r.db_subset / r.db_seg / str(r.anchor_id) / "anchor.png") for r in db.itertuples()]
    for path in a.anchor_png:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    a.to_csv(manifests / "A_db.csv", index=False)


def dense(root):
    topk = root / "retrieval/observed_priority_topk_top50.jsonl"
    _, _, rows = candidate_manifest(topk, 10, N_QUERY)
    manifest = root / "validation_top10.csv"
    if manifest.is_file():
        if read_csv(manifest) != [{k: str(v) for k, v in row.items()} for row in rows]:
            raise ValueError("Saved candidate manifest differs")
    else:
        write_csv(manifest, rows)
    execute([PYTHON, "-u", TOOLS / "audit_priority_pnp_sources.py",
             "--manifest", manifest, "--output", root / "dense", "--scope", "all507",
             "--expected-queries", N_QUERY])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "render", "compose", "retrieval", "dense", "plane", "compare"))
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "check":
        source_tables()
    elif args.action == "render":
        render(root)
    elif args.action == "compose":
        execute([sys.executable, TOOLS / "build_observed_priority_database.py",
                 "--input-root", root, "--output-dir", root / "database",
                 "--b-maps-root", B_MAPS,
                 "--expected-db", N_DB, "--expected-queries", N_QUERY])
    elif args.action == "retrieval":
        execute([sys.executable, TOOLS / "run_observed_priority_study.py",
                 "retrieval", "--root", root])
    elif args.action == "dense":
        dense(root)
    elif args.action == "plane":
        execute([PYTHON, "-u", TOOLS / "replay_priority_pnp_depth_filter.py",
                 "--run", root / "dense", "--output", root / "plane",
                 "--restore-plane", "--plane-rms-m", ".05", "--plane-delta-m", ".10"])
    else:
        for name, folder in (("all", root / "dense/all507/all"), ("plane", root / "plane")):
            data = json.loads((folder / "all507_pnp/integrated_summary.json").read_text())
            if (data["queries"], data["candidate_pairs"]) != (N_QUERY, N_QUERY * 10):
                raise ValueError("Unexpected validation evaluation size")
            print(name, {f"{m:g}m": round(N_QUERY * data[f"success_xy_{m}m_all_queries"])
                         for m in (.5, 1, 2, 5)},
                  "all_failed=", data["all_candidates_failed"], flush=True)
            execute([sys.executable, TOOLS / "audit_near_tie_pose_selection.py",
                     "--root", folder, "--margin", ".05", "--radius-m", "2",
                     "--output", root / f"{name}_selection.csv"])


if __name__ == "__main__":
    main()
