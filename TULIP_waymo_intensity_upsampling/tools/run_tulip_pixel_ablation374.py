"""Isolate replacement and hole filling in the validated 374-query Retrieval run."""
import ast
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from evaluate_retrieval_topk_pipeline import flag, read_csv
from run_priority_validation374 import B_MAPS, N_DB, N_QUERY, REF, execute

W = Path("/mnt/w/its_outputs")
VALID = W / "priority_validation374_d5nms6"
BASE = W / "priority_validation374_abc_gXMLfD"
DEPTH_GATE = "replacement_depth_gate"
NAMES = ("replacement_only", "replacement_and_fill")


def renderer_functions():
    path = REF / "data_make/valid_test/render_anchor_intensity_txt_se3_fix.py"
    required = {"apply_gamma_clip", "masked_hist_equalize_uint8", "apply_clahe_uint8"}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in required]
    if len(nodes) != len(required):
        raise ValueError("Missing inherited renderer functions")
    scope = {"np": np, "cv2": cv2}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def prepare(out, names, depth_limit=None):
    a = pd.read_parquet(BASE / "A/db.parquet")
    b = pd.read_parquet(BASE / "B/db.parquet")
    priority = pd.read_parquet(VALID / "database/db.parquet")
    pairs = pd.read_parquet(BASE / "A/pairs.parquet")
    fields = ["pair_id", "db_seg", "anchor_id"]
    if (len(pairs), len(a)) != (N_QUERY, N_DB) or not all(
        a[fields].astype(str).equals(other[fields].astype(str)) for other in (b, priority)
    ):
        raise ValueError("Expected aligned validation DBs")
    if a.anchor_png.duplicated().any():
        raise ValueError("Expected distinct A anchor paths")
    functions = renderer_functions()
    paths = {name: {} for name in names}
    stats = []

    def post(image, valid, profile):
        pp = profile["postprocess"]
        mask = valid if pp["mask_covered_only"] else np.ones_like(valid)
        if pp["hist_equalize"]:
            image = functions["masked_hist_equalize_uint8"](image, mask)
        if pp["clahe"]:
            image = functions["apply_clahe_uint8"](
                image, mask, pp["clahe_clip_limit"], pp["clahe_tile_grid"])
        return image

    for (subset, segment), group in a.groupby(["db_subset", "db_seg"], sort=False):
        with np.load(B_MAPS / subset / segment / "map_static.npz", allow_pickle=False) as z:
            values = z["intensity"].reshape(-1)
            generated = z["source_is_generated"].reshape(-1).astype(bool)
        measured_values = values[~generated]
        for i, row in group.iterrows():
            ap, bp, pp = map(Path, (row.anchor_png, b.iloc[i].anchor_png, priority.iloc[i].anchor_png))
            meta = json.loads(ap.with_name("anchor.json").read_text())
            bmeta = json.loads(bp.with_name("anchor.json").read_text())
            for field in ("intrinsic", "world_to_cam", "width", "height"):
                if not np.allclose(meta[field], bmeta[field], atol=1e-8, rtol=0):
                    raise ValueError(f"Anchor geometry differs: {ap}/{field}")
            profile = meta["render_profile"]
            if profile["int_mode"] != "global_map" or profile["range_corr_power"] != 0:
                raise ValueError(f"Unsupported intensity profile: {ap}")
            with np.load(ap.with_name("anchor_index.npz"), allow_pickle=False) as z:
                ai = z["point_index"]
                if depth_limit is not None:
                    ad = z["depth"]
            with np.load(bp.with_name("anchor_index.npz"), allow_pickle=False) as z:
                bi = z["point_index"]
                if depth_limit is not None:
                    bd = z["depth"]
            av, bv = ai >= 0, bi >= 0
            if ai.shape != bi.shape or not av.any() or not bv.any():
                raise ValueError(f"Unexpected raster dimensions or coverage: {ap}")
            def quantize(intensity):
                normalized = np.clip(intensity / (profile["intensity_up_val"] + 1e-6), 0, 1)
                normalized = functions["apply_gamma_clip"](normalized, profile["gamma"])
                return np.clip((normalized * 255).round(), 0, 255).astype(np.uint8)
            raw_a, raw_b = np.zeros(ai.shape, np.uint8), np.zeros(ai.shape, np.uint8)
            raw_a[av] = quantize(measured_values[ai[av]])
            raw_b[bv] = quantize(values[bi[bv]])
            gen = bv & generated[np.maximum(bi, 0)]
            replacement, fill = av & gen, ~av & gen
            if not np.array_equal(fill, ~av & bv):
                raise ValueError(f"Some new coverage does not come from generated points: {bp}")
            fill_only = raw_a.copy()
            fill_only[fill] = raw_b[fill]
            if not np.array_equal(post(raw_a.copy(), av, profile), cv2.imread(str(ap), 0)):
                raise ValueError(f"A renderer baseline differs: {ap}")
            if not np.array_equal(post(fill_only, av | fill, profile), cv2.imread(str(pp), 0)):
                raise ValueError(f"Observed-priority baseline differs: {pp}")
            counts = {}
            if depth_limit is not None:
                da, db = ad[replacement], bd[replacement]
                comparable = np.isfinite(da) & np.isfinite(db) & (da > 0) & (db > 0)
                large = np.zeros(da.shape, dtype=bool)
                large[comparable] = np.abs(da[comparable] - db[comparable]) > depth_limit
                blocked = np.zeros_like(replacement)
                blocked[replacement] = large
                counts = dict(blocked=int(large.sum()), kept=int(replacement.sum() - large.sum()),
                              invalid_depth=int((~comparable).sum()))
            for name in names:
                image = raw_a.copy()
                adopted = replacement & ~blocked if name == DEPTH_GATE else replacement
                image[adopted] = raw_b[adopted]
                valid = av
                if name in ("replacement_and_fill", DEPTH_GATE):
                    image[fill] = raw_b[fill]
                    valid = av | fill
                target = out / name / "images" / f"{i:04d}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(target), post(image, valid, profile)):
                    raise OSError(target)
                paths[name][row.anchor_png] = str(target)
            stats.append(dict(db_index=i, db_seg=segment, anchor_id=row.anchor_id,
                              A_pixels=int(av.sum()), replaced=int(replacement.sum()), added=int(fill.sum()), **counts))
            if len(stats) % 50 == 0 or len(stats) == N_DB:
                print(f"Rendered {len(stats)}/{N_DB}", flush=True)
    for name, path_map in paths.items():
        for table, kind in ((a, "db"), (pairs, "pairs")):
            mapped = table.anchor_png.map(path_map)
            if mapped.isna().any():
                raise ValueError(f"Anchor mapping is incomplete: {name}/{kind}")
            table.assign(anchor_png=mapped).to_parquet(out / name / f"{kind}.parquet", index=False)
    pd.DataFrame(stats).to_csv(out / "pixel_counts.csv", index=False)


def evaluate(out, names, reference_run=None):
    template = json.loads((VALID / "retrieval/command.json").read_text())
    for name in names:
        folder = out / name
        command = template.copy()
        for option, value in (("--pairs", folder / "pairs.parquet"), ("--db", folder / "db.parquet"),
                              ("--out_dir", folder), ("--name", name)):
            command[command.index(option) + 1] = str(value)
        if not (folder / f"{name}_per_query.csv").is_file():
            execute(command)
    sources = {
        "A": BASE / "A/retrieval/A_per_query.csv",
        "fill_only": VALID / "retrieval/observed_priority_per_query.csv",
        **{name: out / name / f"{name}_per_query.csv" for name in names},
        "original_B": BASE / "B/retrieval/B_per_query.csv",
    }
    if reference_run is not None:
        for name in NAMES:
            sources[name] = reference_run / name / f"{name}_per_query.csv"
    keys = ("pair_id", "q_cam_png", "gt_index_db")
    frames = {name: {tuple(r[k] for k in keys): r for r in read_csv(path)}
              for name, path in sources.items()}
    reference = frames["A"].keys()
    if len(reference) != N_QUERY or any(f.keys() != reference for f in frames.values()):
        raise ValueError("Retrieval query identities differ")
    metrics = ("strict_hit@10", "soft_t2m_hit@10", "soft_t5m_hit@10")
    rows = []
    for name, frame in frames.items():
        groups = [("ALL", reference)]
        groups.extend((pair, [k for k in reference if k[0] == pair])
                      for pair in sorted({k[0] for k in reference}))
        for pair, group in groups:
            rows.append(dict(condition=name, pair_id=pair, queries=len(group),
                             **{m: sum(flag(frame[k][m]) for k in group) for m in metrics}))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "comparison.csv", index=False)
    print(summary[summary.pair_id == "ALL"].to_string(index=False))
    changes = []
    comparisons = (("A", "replacement_only"), ("A", "fill_only"),
                     ("fill_only", "replacement_and_fill"),
                     ("replacement_only", "replacement_and_fill"))
    if reference_run is not None:
        comparisons = tuple((old, DEPTH_GATE) for old in ("replacement_and_fill", "fill_only", "A"))
    for old, new in comparisons:
        for m in metrics:
            paired = [(flag(frames[old][k][m]), flag(frames[new][k][m])) for k in reference]
            changes.append(dict(before=old, after=new, metric=m,
                                improved=sum(not a and b for a, b in paired),
                                worsened=sum(a and not b for a, b in paired)))
    pd.DataFrame(changes).to_csv(out / "transitions.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--depth-limit-m", type=float)
    parser.add_argument("--reference-run", type=Path)
    args = parser.parse_args()
    if (args.depth_limit_m is None) != (args.reference_run is None):
        parser.error("--depth-limit-m and --reference-run must be supplied together")
    if args.depth_limit_m is not None and (not np.isfinite(args.depth_limit_m) or args.depth_limit_m < 0):
        parser.error("--depth-limit-m must be finite and nonnegative")
    names = NAMES if args.depth_limit_m is None else (DEPTH_GATE,)
    if args.reference_run is not None:
        if args.reference_run.resolve() == args.output.resolve():
            parser.error("--output must differ from --reference-run")
        for name in NAMES:
            if not (args.reference_run / name / f"{name}_per_query.csv").is_file():
                parser.error(f"Missing reference results for {name}")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.depth_limit_m is not None:
        settings = {"depth_limit_m": args.depth_limit_m, "reference_run": str(args.reference_run.resolve())}
        config = out / "depth_gate_config.json"
        if config.exists():
            if json.loads(config.read_text()) != settings:
                parser.error("Output folder already uses different depth gate settings")
        elif (out / "pixel_counts.csv").exists():
            parser.error("Choose a new output folder for the depth gate")
        else:
            config.write_text(json.dumps(settings, indent=2) + "\n")
    if not (out / "pixel_counts.csv").is_file():
        prepare(out, names, args.depth_limit_m)
    if args.depth_limit_m is not None:
        counts = pd.read_csv(out / "pixel_counts.csv")
        print(counts[["replaced", "blocked", "kept", "invalid_depth", "added"]].sum().to_string())
    if not args.prepare_only:
        evaluate(out, names, args.reference_run)


if __name__ == "__main__":
    main()
