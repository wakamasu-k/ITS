# Waymo 32->64 距離・反射強度統計

## 実行条件

- split: training
- scenes: ファイル名順の先頭10
- frames: 各sceneから均等に20
- 合計: 200 frames
- global histogram/percentile: 各frame・各groupから最大10,000値の
  deterministic frame-balanced sample
- per-frame CSV: 全有効画素から計算したexact値

## 実行

```bash
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
source scripts/01_activate_waymo_gpu.sh
TF_CPP_MIN_LOG_LEVEL=2 bash scripts/30_analyze_waymo_32_to_64_statistics.sh
```

## 生成先

`outputs/statistics_32_to_64`

- `statistics.json`
- `per_frame_statistics.csv`
- `per_scene_statistics.csv`
- `outlier_frames.csv`
- `range_histogram.png`
- `intensity_histogram.png`
- `intensity_log_histogram.png`

## 主な結果

### 距離

| group | median [m] | p95 [m] | p99.5 [m] | max [m] |
|---|---:|---:|---:|---:|
| all 64 | 13.520 | 47.861 | 70.088 | 74.995 |
| observed even 32 | 13.596 | 48.037 | 70.112 | 74.995 |
| generated odd 32 | 13.450 | 47.662 | 69.995 | 74.995 |

偶数行と奇数行の距離分布は近い。距離表示・モデル正規化の上限候補として
75mを維持できる。

### 反射強度

| group | median | p95 | p99 | p99.5 | p99.9 | max |
|---|---:|---:|---:|---:|---:|---:|
| all 64 | 0.1177 | 0.3574 | 0.6484 | 1.2031 | 203.0 | 112128 |
| observed even 32 | 0.1187 | 0.3613 | 0.6719 | 1.3828 | 264.0 | 94720 |
| generated odd 32 | 0.1167 | 0.3555 | 0.6211 | 1.0547 | 126.5 | 112128 |

通常域（median～p99）は偶数行と奇数行で近い。一方、p99.5以降に急激な
heavy tailがあり、raw値の平均・標準偏差・L1 lossは少数極端値に支配される。

200フレーム中191フレームで、frame最大反射強度がsampled global p99.9の
10倍を超えた。したがって極端値は特定の壊れた1フレームだけの問題ではなく、
多くのframeに少数画素として存在する。

NaN、Inf、負の距離、負の反射強度はsample内で検出されなかった。

## 暫定方針

学習前に次の候補を比較する。

1. raw intensity + Huber loss
2. `log1p(intensity)` + Huber loss
3. `log1p(clip(intensity, 0, 256)) / log1p(256)` + Huber loss

第3候補の256は、今回のall-64 p99.9=203、even p99.9=264を基にした暫定値。
10 sceneだけで最終決定せず、Datasetアダプターでは値を設定ファイルから
変更可能にする。

評価では、変換空間のlossだけでなく、逆変換後の物理値について奇数行のみの
MAE/RMSE/biasを保存する。clip対象画素率とraw最大値も別指標にする。

CLAHEとヒストグラム平坦化は不可逆なので学習targetには使用しない。
