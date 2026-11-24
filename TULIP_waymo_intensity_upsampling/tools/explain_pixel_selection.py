"""Explain raster provenance with separate overlays, not an error/attention map."""
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pandas as pd

project = Path(__file__).resolve().parents[1]
table = pd.read_csv(project/'outputs/preserve_observed_check_LoT4ys/images/images.csv', dtype={'anchor':str})
maskroot = project/'outputs/priority_replacement_visual_j1r56U/report'
out = Path(tempfile.mkdtemp(prefix='pixel_selection_explanation_',dir=project/'outputs'))
for anchor in ['50','102']:
    rasters = {}
    for v in ['A','B']:
        row = table[(table.anchor==anchor)&(table.condition==v+'_original')]
        assert len(row)==1
        rasters[v] = cv2.imread(row.iloc[0].image)
        assert rasters[v] is not None
    mask = cv2.imread(str(maskroot/f'anchor_{anchor}_replacement_mask.png'))
    assert mask is not None and mask.shape==rasters['B'].shape
    replaced = np.all(mask==[0,0,255],axis=2)
    added = np.all(mask==[0,200,0],axis=2)
    # Overlay separately to avoid confusing replacement with newly covered pixels.
    red = rasters['B'].copy(); green = rasters['B'].copy()
    red[replaced] = (0.25*red[replaced]+0.75*np.array([0,0,255])).astype(np.uint8)
    green[added] = (0.25*green[added]+0.75*np.array([0,255,0])).astype(np.uint8)
    panels = []
    labels = ['1. A: observed 16 lines','2. B: observed + generated points',
              '3. RED: A had a point; B displays a generated point',
              '4. GREEN: A had no point; B fills the pixel']
    for image,label in zip([rasters['A'],rasters['B'],red,green],labels):
        panel=np.full((560,768,3),255,np.uint8)
        panel[48:]=cv2.resize(image,(768,512),interpolation=cv2.INTER_AREA)
        cv2.putText(panel,label,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,0,0),1,cv2.LINE_AA)
        panels.append(panel)
    sheet=np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])])
    assert cv2.imwrite(str(out/f'anchor_{anchor}_explained.png'),sheet)
    print(out/f'anchor_{anchor}_explained.png')
