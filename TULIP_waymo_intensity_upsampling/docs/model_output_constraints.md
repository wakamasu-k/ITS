# TULIPの出力制約と32->64方針

## KITTIで確認された問題

学習済みTULIPのraw出力には、負の距離と範囲外の反射強度が存在した。
これはTransformer本体が完全に誤っているというより、最終出力が非制約の
線形回帰であり、物理的な値域を保証していないことが原因である。

## 後処理だけで解決しない

表示時のclipは必要だが、clip後の画像だけで評価するとraw予測の問題を隠す。
以下を別々に保存・評価する。

- raw prediction
- physically constrained prediction
- observed rowsをGTで保持したfused prediction
- camera projection / visualization

## Waymo 32->64での推奨設計

- input: `[32,W,2]`（range, raw intensity）
- target: `[64,W,2]`
- observed target rows: `0,2,...,62`
- generated target rows: `1,3,...,63`
- 主評価: generated rowsだけ
- range loss: valid mask付きL1またはHuber
- intensity loss: valid mask付きrobust loss
- rangeとintensityの損失重みを分離
- 欠損画素をゼロ背景として損失へ混ぜない

### 距離出力

候補:

- `softplus(raw)`で非負を保証
- または学習範囲を固定した`max_range * sigmoid(raw)`

上限を固定する場合、Waymo TOP LiDARの最大計測距離と学習データ統計を確認して
決定する。現在の代表フレームでは最大約75mだが、1フレームだけを根拠に
全データの上限を決定しない。

### 反射強度出力

Waymo raw intensityには大きな外れ値があり、代表フレームでは中央値約0.155に
対して最大22,016だった。そのまま線形回帰すると外れ値が学習を支配する。

候補:

- dataset全体で固定したpercentileによるscale
- `log1p`等の可逆変換
- 変換空間でrobust loss
- 評価時に逆変換して物理値MAE/RMSEも計算

CLAHEやヒストグラム平坦化は不可逆なので、TULIPの学習targetには使わない。
これらは後段のカメラ照合用画像に限定する。
