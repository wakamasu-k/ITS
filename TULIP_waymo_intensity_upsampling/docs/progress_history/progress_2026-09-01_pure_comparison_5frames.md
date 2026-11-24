# Waymo純粋GT・Low・Pred 5サンプル比較

作成日：2026-09-01

## 対象

- checkpoint：outputs/waymo_16_to_32_full_e300/checkpoint.pth
- split：test
- sample index：0, 10, 20, 30, 39
- range scale：75.0 m
- intensity transform：log1p
- 比較対象：生成された奇数行のみ

## サンプル別結果

| sample | generated pixels | range MAE (m) | intensity MAE | GT intensity max | Pred intensity max |
|---:|---:|---:|---:|---:|---:|
| 0 | 37,252 | 1.3967 | 12.9377 | 21,760.0 | 432.3 |
| 10 | 37,071 | 1.5521 | 12.4035 | 21,120.0 | 437.0 |
| 20 | 37,208 | 1.1764 | 27.1234 | 89,088.0 | 440.7 |
| 30 | 37,501 | 0.8482 | 27.4480 | 307,200.1 | 431.3 |
| 39 | 37,482 | 0.9945 | 52.2451 | 344,063.9 | 432.1 |

## 5サンプル集計

- generated pixels：186,514
- range MAE：1.1925 m
- intensity MAE：26.4782
- GT intensity max：344,063.9
- Pred intensity max：440.7
- GT intensity p99：1.1016
- GT intensity p99.9：1,760.0
- GT intensity 1000以上：235点、0.1260%

## 解釈

rangeは5サンプルで概ね1 m前後の復元誤差だった。一方、intensityはサンプルによる変動が大きく、特に高強度外れ値をPredが再現できていない。高強度画素は0.126%程度だが、GT最大値が数十万に達するため、raw intensity MAEを大きく押し上げる。

この結果は、TULIPのforward失敗や画像生成失敗を示すものではない。低〜中強度の構造とrange復元を確認するには有効だが、高強度反射の再現性能は別評価が必要である。

## 次の作業

次は、RGB画像上で純粋なGT・Low・Pred intensityを同一条件で投影する。既存のoracle画像はGT range + Pred intensityであり、純粋なGT画像とは区別する。RGB投影では、同じcalibration、z-buffer、表示スケール、mask条件を使用する。

その後、投影点数、pixel shift、強度帯別誤差、高強度点の再現率を確認する。Waymo全体データへの拡張は、少数シナリオでの視覚・数値確認後に行う。

## コード・既存環境

追加した可視化ツール：

- tools/visualize_waymo_16_to_32_gt_low_pred.py

TULIP_backup、handover_shimizu、既存Phase 1〜3コード、TULIP Transformer構造は変更していない。
