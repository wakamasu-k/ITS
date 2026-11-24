# Waymoカメラ投影診断

作成日: 2026-08-20

## 目的

16→32 TULIP出力について、GT rangeとTULIP rangeの違いがカメラ画像上の投影位置へ与える影響を分離する。

比較条件は以下の2つである。

- oracle: GT range + TULIP intensity
- predicted: TULIP range + TULIP intensity

## 座標系

既存のhandover_shimizuで確認したWaymo定義を使用する。

- calibration extrinsic: vehicle <- camera
- 逆行列: camera <- vehicle
- カメラ前方深度: X_c
- 画像横方向: -Y_c
- 画像縦方向: -Z_c

TULIP、Transformer、Dataset、handover_shimizuのコードは変更しない。

## 出力

各フレームについて、raw値を`projection_raw.npz`へ保存する。PNGは確認用の99.5パーセンタイル可視化であり、学習・数値評価には使用しない。

## 判定

- oracleとpredictedの投影画素差が大きい場合: range復元がカメラ対応の主因
- 投影画素差が小さく、intensity誤差が大きい場合: intensity復元が主因
- testのintensityのみ悪化する場合: intensity外れ値・split分布差を優先調査
