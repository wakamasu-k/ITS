# Waymo純粋GT・Low・Pred画像確認

作成日：2026-08-31

## 実施内容

既存のTULIPモデル本体とDatasetを変更せず、Waymoの16→32出力について、oracleを使わない純粋なGT・Low・Pred比較画像を作成した。

- checkpoint：outputs/waymo_16_to_32_full_e300/checkpoint.pth
- split：test
- sample index：0
- range scale：75.0 m
- intensity transform：log1p
- native width：2650

## 純粋な比較定義

- GT：32ラインtarget
- Low：16ラインinputを32ラインgridの0,2,4,...,30行へ配置
- Pred：TULIPの32ライン出力
- 比較範囲：native [32,2650]
- PNG：共通カラースケールによる確認用
- NPZ：raw float32値とmaskを保存
- oracle range：使用していない

## 生成ファイル

保存先：

outputs/waymo_16_to_32_full_e300/pure_test_sample0/

- range_gt_low_pred.png
- intensity_gt_low_pred_linear.png
- intensity_gt_low_pred_log1p.png
- comparison_raw.npz
- metadata.json

## shape・dtype確認

NPZのrange/intensity配列はすべて [32,2650]、dtypeはfloat32だった。mask配列はすべて [32,2650]、dtypeはboolだった。PNGは3枚とも読み込み可能で、表示画像サイズは3060 x 1020だった。

## sample 0の生成行評価

- generated range MAE：1.3966618 m
- generated intensity MAE：12.9377251

## 重要な解釈

この画像は、GT・Low・Predの信号そのものを比較するための画像であり、カメラ投影のoracle条件ではない。Lowは32行grid上で偶数行だけに有効点があり、奇数行は欠損する。Predはその奇数行を含む32ライン出力である。

## 今後の作業

次は同じ定義で、RGB画像上に以下を投影する。

- RGB + GT intensity
- RGB + Low intensity
- RGB + TULIP Pred intensity

既存のcamera projection画像はGT range + Pred intensityのoracleとPred range + Pred intensityが中心であり、純粋なGT/Low/Pred 3面比較とは区別する。カメラ投影では、同じcalibration、z-buffer、画素サイズ、表示スケールを使用する。

その後、sample 0だけでなくvalidation/testの代表数フレームへ増やし、FRONT_RIGHTと高強度帯を重点確認する。Waymo全体データへの拡張は、その小規模確認後に行う。

## コード変更

追加した正式ファイル：

- tools/visualize_waymo_16_to_32_gt_low_pred.py

TULIP_backup、handover_shimizu、既存Phase 1〜3コード、TULIPモデル構造は変更していない。
