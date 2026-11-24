# 標準TULIP 2ch baseline: 点群復元評価

## 条件

- checkpoint: `outputs/waymo_16_to_32_paper_baseline_e30/checkpoint.pth`
- epochs: 30
- train samples: 160
- loss: 標準2ch masked L1
- intensity transform: raw
- evaluated frames: validation 40 / test 40
- coordinate frame: Waymo vehicle frame
- rows: generated odd rows `1,3,...,31`

## validation 40

| 方法 | 点数 | coordinate MAE [m] | Euclidean MAE [m] | intensity MAE |
|---|---:|---:|---:|---:|
| TULIP | 1,537,686 | 1.6659 | 3.8470 | **6.3253** |
| nearest | 1,485,122 | 0.5929 | 1.3374 | 8.3564 |
| linear | 1,530,817 | 0.8052 | 1.8544 | 9.8630 |

## test 40

| 方法 | 点数 | coordinate MAE [m] | Euclidean MAE [m] | intensity MAE |
|---|---:|---:|---:|---:|
| TULIP | 1,491,970 | 1.6959 | 3.8677 | **18.0209** |
| nearest | 1,412,230 | 0.6342 | 1.4126 | 19.2183 |
| linear | 1,479,258 | 0.9873 | 2.2384 | 18.3574 |

## 解釈

標準TULIP構造・標準masked L1でも、欠損ringのintensity復元はnearest/linearより良い。
一方、range誤差は点群の3D位置誤差へ伝播し、nearest/linearを下回る。

この結果は、TULIPの基本構造を維持したままWaymo 2ch intensity upsamplingを進められることを示す。
rangeはTULIP論文の点群復元という観点で継続課題だが、intensity matchingの目的では現方式を継続する根拠が得られた。

## 出力

- `outputs/waymo_16_to_32_paper_baseline_e30/validation_pointcloud_40/summary.json`
- `outputs/waymo_16_to_32_paper_baseline_e30/test_pointcloud_40/summary.json`

tail-weighted結果は構造変更ではなく追加の損失条件として、標準baselineとは別に扱う。

---

作成日（ファイル更新日時から推定）: 2026-08-20
