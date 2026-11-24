# scaled checkpointのpoint-cloud評価

作成日: 2026-08-21

## 目的

`range / 75.0`と`log1p(intensity)`で学習したcheckpointを、raw meter/raw intensityへ戻してWaymo vehicle frameの点群として評価する。

## 入力復元

- range: `prediction_range * 75.0`
- intensity: `expm1(prediction_intensity)`

## 評価範囲

- TOP LiDAR
- native 64ラインの偶数ring `0,2,...,62`
- 生成32ラインの奇数row `1,3,...,31`
- GT/predictionの共通valid画素
- Waymo公式range-image geometryによるvehicle座標

## 注意

scale未対応の既存point-cloud評価へscaled checkpointを直接渡さない。raw domainへ復元してから評価する。
