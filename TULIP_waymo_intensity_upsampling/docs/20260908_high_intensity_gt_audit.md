# Waymo 16→32生成ring 高反射GT監査（2026-09-08）

## 目的

高反射重み付き学習の前に、test 40フレームの生成対象ringに含まれるraw intensityの分布、無効点、フレーム偏在、ring偏在、観測ringから得られる手掛かりを確認する。

## 実行コマンド

```bash
source /home/wakamatsu/ITS/.venv_tulip/bin/activate
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling

python tools/audit_waymo_16_to_32_high_intensity.py \
  --index outputs/training_pairs_16_to_32_full/index.csv \
  --output-root outputs/high_intensity_gt_audit_test40_20260908 \
  --dataset-role test \
  --max-frames 40 \
  --high-threshold 1000
```

## 対象

- test frames: 40
- segments: 2
- mask: valid generated odd rows only
- generated valid pixels: 1,491,970
- 1000以上の点: 1,768
- 1000以上の点を含むフレーム: 40/40

## raw intensity分布

| 指標 | 値 |
|---|---:|
| min | 0.000202 |
| median | 0.119629 |
| p99 | 1.000000 |
| p99.5 | 4.093750 |
| p99.9 | 1,504 |
| p99.99 | 21,120 |
| max | 372,736 |

中央値と最大値の差が非常に大きく、raw intensityは強いlong-tail分布である。

## 閾値別の観測手掛かり

「同一列」は欠落ringの上下にある観測ringの同じcolumn、「±1列」は上下観測ringの近傍columnまで含む。

| 閾値 | 点数 | 全有効点に占める割合 | 無効点上の高反射 | 同一列に手掛かり | ±1列に手掛かり |
|---|---:|---:|---:|---:|---:|
| 20以上 | 4,812 | 0.3225% | 0 | 62.97% | 70.49% |
| 100以上 | 3,160 | 0.2118% | 0 | 66.74% | 74.02% |
| 1000以上 | 1,768 | 0.1185% | 0 | 68.16% | 79.19% |

1000以上の約68%は同一columnの上下観測ringから直接的な手掛かりを得られる。±1 columnまで含めれば約79%である。一方、約21%はこの局所範囲に同じ閾値以上の観測がなく、局所補間だけでは再現しにくい。

## フレーム偏在

- 1000以上は全40フレームに存在する。
- 最大フレームの占有率は4.07%。
- 上位5フレームの占有率は17.76%。

高反射点は少数の異常フレームだけに集中していない。

## 100,000以上の極端値

- 点数: 41
- native ring: すべてring 2
- 距離範囲: 39.21～70.94 m
- 最大点: segment `10876852935525353526_1640_000_1660_000`、frame 20、row 1、native ring 2、column 2566
- 最大点のrange: 70.9431 m
- 最大点のraw intensity: 372,736

最大点を元TFRecordから直接復号した結果も以下で一致した。

```text
native shape: [64, 2650, 4]
range[2,2566]: 70.94307708740234
intensity[2,2566]: 372736.0
elongation[2,2566]: 0.1112511083483696
nlz[2,2566]: -1.0
```

したがって最大値は16→32変換やNPZ保存で発生した破損ではなく、元Waymo TFRecordの復号直後から存在する。ただし100,000以上がring 2の遠距離だけに偏るため、そのまま正確なraw値として回帰するか、上限clipまたは別扱いにするかは比較実験が必要である。

## 判断

1. 1000以上を一律に削除する根拠はない。全フレームに存在し、無効点でもない。
2. 最大級の値は強いring・距離依存を持つため、raw MAEだけを最適化すると学習が不安定になりやすい。
3. 次は元データを消さず、学習損失内だけで最大weight 2のmoderate weightingを試す。
4. raw値の上限clipを試す場合も、GT保存値は変更せず、学習変換の条件として明示する。
5. 比較はvalidation 40フレームで選択し、testは最終条件だけに使用する。

## 出力

- `outputs/high_intensity_gt_audit_test40_20260908/summary.json`
- `outputs/high_intensity_gt_audit_test40_20260908/per_frame.csv`
- `outputs/high_intensity_gt_audit_test40_20260908/high_points.csv`
- `outputs/high_intensity_gt_audit_test40_20260908/run.log`
