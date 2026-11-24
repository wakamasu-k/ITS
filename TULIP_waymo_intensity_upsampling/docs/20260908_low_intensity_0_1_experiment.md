# Waymo反射強度0～1優先実験

作成日: 2026-09-08

## 決定事項

- TULIPは16ライン入力から32ラインのrangeとintensityを全画素で出力する。
- range lossは従来どおり、有効な生成奇数行の全画素を対象とする。
- `low_0_1`実験では、intensity lossを`0 <= raw target intensity <= 1`の生成奇数行だけで計算する。
- intensityは学習時に`log1p`領域を使用する。
- 1を超えるGT値はデータや出力から削除せず、intensity lossからだけ除外する。
- TULIPのTransformer、Encoder、Decoder、Attentionは変更しない。

## 研究上の位置付け

Validationの生成行では、raw intensity 0～1が有効画素の約98.05%を占める。
まずこの主要領域の復元性能を改善し、条件を確定した後に1を超える反射強度を
別段階で再導入する。高反射領域は未解決課題として保持し、既存の強度帯別評価と
閾値別Precision、Recall、F1を継続して保存する。

## 比較条件

以下を同じデータ分割、seed、epoch、optimizer、range scale、生成行maskで比較する。

1. `baseline`: 全有効生成画素のintensityを学習する。
2. `moderate`: 高反射帯へ最大2倍の重みを付ける。
3. `low_0_1`: raw target intensity 0～1だけをintensity lossへ含める。

主指標は生成行の0～1 Raw intensity MAEとlog1p intensity MAEとする。range MAE、
全域Raw/log1p MAE、高反射帯の検出指標も併記し、評価範囲の限定を明示する。

## 次段階

0～1の条件をValidationで確定した後、1を超える値を弱い重み、別損失、または
別ヘッドで扱う実験を検討する。TestデータはValidationで条件を選び終えるまで
モデル選択に使用しない。
