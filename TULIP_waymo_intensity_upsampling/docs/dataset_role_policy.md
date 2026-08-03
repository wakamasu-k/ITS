# 32→64学習ペアとdataset role

## 32→64ペア

`frame_64_32.npz` から次の学習ペアを生成する。

- input: `input_32 [32,W,2]`
- target: `target_64 [64,W,2]`
- channel 0: range
- channel 1: raw intensity
- observed target rows: `0,2,...,62`
- generated target rows: `1,3,...,63`

32ラインは同じ64ラインの偶数ringの完全な部分集合であり、range、intensity、maskを
個別に完全一致検証してからペアを保存する。

## 881フレームの役割

`manifest_cross_eval_gt_v3.csv` から得た881フレームは、清水研究の自己位置推定評価と
対応している。このため既定の `dataset_role` は `localization_evaluation` とし、
TULIPのtrainingには使用しない。

TULIP学習用データは別のWaymo segmentから作成し、segment単位で次を分離する。

```text
train segments
validation segments
test segments
localization_evaluation segments (the inherited 881 frames)
```

同じsegmentを複数roleへ含めてはならない。

## 5フレームでの生成

dry-run:

```bash
source /home/wakamatsu/ITS/.venv_tulip/bin/activate
bash scripts/55_build_shimizu_32_to_64_pairs.sh --dry-run --max-frames 5
```

実生成:

```bash
bash scripts/55_build_shimizu_32_to_64_pairs.sh --max-frames 5 --resume
```

生成物:

```text
tulip_pair_32_to_64.npz
tulip_pair_32_to_64.json
outputs/shimizu_pairs_32_to_64/index.csv
outputs/shimizu_pairs_32_to_64/failures.csv
```

`index.csv` は必ず `dataset_role` を持つ。Datasetアダプターは明示的に要求したroleの
行だけをロードし、evaluationを暗黙にtrainingへ流用しないものとする。

## Datasetアダプター

```python
from pathlib import Path
from tulip_adapter import Waymo32To64Dataset

dataset = Waymo32To64Dataset(
    Path("outputs/shimizu_pairs_32_to_64/index.csv"),
    dataset_role="localization_evaluation",
    intensity_transform="raw",
)
```

返却shapeはPyTorch形式の `input [2,32,W]`、`target [2,64,W]` である。
`generated_mask [64,W]` は奇数ringかつGT有効な画素だけTrueとなり、主lossに使う。
`dataset_role="train"` を指定して評価indexしかない場合は空Datasetにせずエラーとする。

intensityは `raw` と `log1p` を選択できる。最終採用方式はtraining segment全体の
統計を算出してから固定する。
