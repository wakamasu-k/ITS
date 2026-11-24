# range正規化・intensity変換実験

作成日: 2026-08-20

## 目的

TULIPのTransformer本体を変更せず、入力Datasetの値域だけを調整して、rangeとintensityのloss支配関係を確認する。

## 条件

- range scale: `75.0 m`
- intensity transform: `log1p`
- loss: 既存TULIPの2ch masked L1
- model structure: 変更なし
- train subset: 20 samples

## 小規模結果

5epoch、20サンプルでのraw値評価:

| split | range MAE | intensity MAE |
|---|---:|---:|
| validation 40 | 3.9482 m | 6.7748 |
| test 40 | 4.2915 m | 13.3934 |

標準baselineは30epoch・160サンプルなので、この結果だけでは最終比較としない。

## 次の比較

標準baselineと同じ160サンプル・30epochで、以下を比較する。

- raw range + raw intensity
- range/75 + log1p intensity

評価は必ずraw range/raw intensityへ戻して行う。

## 注意

`75.0 m`と`log1p`は現時点の候補であり、最終設定として固定していない。
