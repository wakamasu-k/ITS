# range=75m・log1p 30epoch結果

作成日: 2026-08-21

## 学習

- samples: 160
- epochs: 30
- range scale: `75.0 m`
- intensity transform: `log1p`
- normalized loss: `0.0643732 -> 0.0210108`
- checkpoint: `outputs/waymo_16_to_32_scaled75_log1p_e30/checkpoint.pth`

## raw値での評価

| split | TULIP range MAE | TULIP intensity MAE | nearest range MAE | linear range MAE |
|---|---:|---:|---:|---:|
| validation 40 | 0.8907 m | 6.6871 | 1.1749 m | 1.4165 m |
| test 40 | 1.0135 m | 13.2662 | 1.3855 m | 1.6972 m |

標準baselineはvalidation range `3.9983 m`、test range `4.0357 m`であり、range誤差は大きく改善した。intensityもvalidation/testともに小幅改善した。

## 解釈

rangeの未正規化が、2ch学習時のrange復元を不安定にしていた可能性が高い。`75.0 m`と`log1p`は有力候補だが、最終採用前にpoint-cloud、camera projection、intensity binsで再評価する。

## 注意

このcheckpointはrangeを75で割った学習値を出力する。point-cloud/camera評価では、必ずrangeを75倍してraw meterへ戻す必要がある。既存のscale非対応評価スクリプトへそのまま渡してはいけない。
