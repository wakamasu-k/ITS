"""Reproduce saved rasters exactly before changing normalization/postprocessing.

Uses saved point_index, never changes geometry or baseline images.
Loads only three pure image functions from the inherited renderer via AST.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--preserve-observed', action='store_true', help='Add diagnostic A-covered/B-hole composites')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    renderer = project.parent/'handover_shimizu/src/data_make/valid_test/render_anchor_intensity_txt_se3_fix.py'
    source = renderer.read_text()
    names = {'apply_gamma_clip', 'masked_hist_equalize_uint8', 'apply_clahe_uint8'}
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == 3
    scope = {'np': np, 'cv2': cv2}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(renderer), 'exec'), scope)
    root = project/'outputs/shimizu_protocol_507q_754db_abc'
    cases = pd.read_csv(project/'outputs/gt_vs_anchor102_audit_ax6mcoi2/anchor_comparison.csv', dtype={'anchor': str})
    if args.limit > 0:
        cases = cases.head(args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checks, outputs, preservation = [], [], []

    def postprocess(image, valid, profile):
        pp = profile['postprocess']
        mask = valid if pp['mask_covered_only'] else np.ones_like(valid)
        if pp['hist_equalize']:
            image = scope['masked_hist_equalize_uint8'](image, mask)
        if pp['clahe']:
            image = scope['apply_clahe_uint8'](image, mask, pp['clahe_clip_limit'], pp['clahe_tile_grid'])
        return image

    def render(intensity, idx, profile, upper, post):
        assert profile['int_mode'] == 'global_map'
        assert profile['range_corr_power'] == 0
        valid = idx >= 0
        assert valid.any() and idx[valid].max() < len(intensity)
        values = np.clip(intensity / (upper+1e-6), 0, 1)
        values = scope['apply_gamma_clip'](values, profile['gamma'])
        image = np.zeros(idx.shape, dtype=np.float32)
        image[valid] = values[idx[valid]]
        image = np.clip((image*255).round(), 0, 255).astype(np.uint8)
        pp = profile['postprocess']
        mask = valid if pp['mask_covered_only'] else np.ones_like(valid)
        if post and pp['hist_equalize']:
            image = scope['masked_hist_equalize_uint8'](image, mask)
        if post and pp['clahe']:
            image = scope['apply_clahe_uint8'](image, mask, pp['clahe_clip_limit'], pp['clahe_tile_grid'])
        return image

    for segment, group in cases.groupby('segment'):
        intensities = {}
        for v, mapname in [('A','A_observed16'), ('B','B_tulip32')]:
            with np.load(root/'maps'/mapname/'training'/segment/'map_static.npz') as z:
                intensities[v] = z['intensity'].reshape(-1)
        for case in group.itertuples():
            profiles, indices, baseline = {}, {}, {}
            for v in 'AB':
                folder = root/f'anchors_{v}/training'/segment/case.anchor
                profiles[v] = json.loads((folder/'anchor.json').read_text())['render_profile']
                with np.load(folder/'anchor_index.npz') as z:
                    indices[v] = z['point_index']
                original = cv2.imread(str(folder/'anchor.png'), cv2.IMREAD_GRAYSCALE)
                assert original is not None
                baseline[v] = render(intensities[v], indices[v], profiles[v], profiles[v]['intensity_up_val'], True)
                difference = np.abs(baseline[v].astype(float)-original.astype(float))
                checks.append(dict(segment=segment, anchor=case.anchor, variant=v,
                                   changed_pixels=int((difference>0).sum()), mae=float(difference.mean()), max_diff=float(difference.max())))
                pd.DataFrame(checks).to_csv(args.output_dir/'baseline_check.csv', index=False)
                if difference.max() != 0:
                    raise RuntimeError(f'{v} anchor {case.anchor}: baseline mismatch; see baseline_check.csv. Do not interpret ablations.')
            # All conditions preserve the original variant's selected points.
            configs = [('A_original','A',profiles['A']['intensity_up_val'],True),
                       ('B_original','B',profiles['B']['intensity_up_val'],True),
                       ('B_Aupper_post','B',profiles['A']['intensity_up_val'],True),
                       ('A_Aupper_no_post','A',profiles['A']['intensity_up_val'],False),
                       ('B_Aupper_no_post','B',profiles['A']['intensity_up_val'],False)]
            for name,v,upper,post in configs:
                image = render(intensities[v], indices[v], profiles[v], upper, post)
                dest = args.output_dir/name/segment/case.anchor/'anchor.png'
                dest.parent.mkdir(parents=True, exist_ok=True)
                assert cv2.imwrite(str(dest), image)
                outputs.append(dict(role=case.role, segment=segment, anchor=case.anchor, condition=name,
                                    variant=v, upper=upper, post=post, image=str(dest.resolve())))
            if args.preserve_observed:
                av, bv = indices['A']>=0, indices['B']>=0
                added = ~av & bv
                upper = profiles['A']['intensity_up_val']
                ar = render(intensities['A'], indices['A'], profiles['A'], upper, False)
                br = render(intensities['B'], indices['B'], profiles['B'], upper, False)
                hybrid = ar.copy()
                hybrid[added] = br[added]
                assert np.array_equal(hybrid[av], ar[av])
                # Joint postprocessing may alter even the retained observed pixels.
                processed = postprocess(hybrid.copy(), av|bv, profiles['A'])
                final = baseline['A'].copy()
                final[added] = baseline['B'][added]
                assert np.array_equal(final[av], baseline['A'][av])
                preservation.append(dict(anchor=case.anchor, a_covered=int(av.sum()), added_pixels=int(added.sum()),
                    final_A_changed=int((final[av]!=baseline['A'][av]).sum()),
                    joint_post_A_changed=int((processed[av]!=baseline['A'][av]).sum())))
                for name,image,post in [('AB_holes_no_post',hybrid,False),
                                        ('AB_holes_joint_post',processed,True),
                                        ('AB_holes_final_png',final,True)]:
                    dest = args.output_dir/name/segment/case.anchor/'anchor.png'
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    assert cv2.imwrite(str(dest), image)
                    outputs.append(dict(role=case.role,segment=segment,anchor=case.anchor,condition=name,
                        variant='diagnostic_composite',upper=upper if name!='AB_holes_final_png' else None,
                        post=post,image=str(dest.resolve())))
            print(f'Exact baseline match A/B; prepared anchor={case.anchor}', flush=True)
    pd.DataFrame(outputs).to_csv(args.output_dir/'images.csv', index=False)
    if preservation:
        pd.DataFrame(preservation).to_csv(args.output_dir/'preservation_check.csv', index=False)
    (args.output_dir/'completed.json').write_text(json.dumps(dict(anchors=len(cases),
        renderer=str(renderer), renderer_sha256=hashlib.sha256(source.encode()).hexdigest(),
        preserve_observed=args.preserve_observed,
        note='Saved rasters fixed; optional A-covered/B-hole composites are diagnostics, not physically rerendered maps. Hist-equalization and CLAHE disabled together in no_post. Not a retrieval evaluation.'), indent=2))
    print('DONE: baseline_check.csv, images.csv, completed.json')


if __name__ == '__main__':
    main()
