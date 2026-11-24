# Waymo TULIP 16→32 高反射強度対応 進捗記録

作成日：2026-09-05  
作業ルート：/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling

## 1. 目的と現状

Waymo TOP LiDARの16ライン入力から32ライン相当のrange/intensityを復元し、反射強度画像と可視光カメラ画像の対応へ利用する。

Waymo前処理では、TOP LiDAR第1returnのnative Range Image [64,2650,4]を読み込み、channel 0をrange、channel 1をintensityとして使用している。64ラインの偶数行から32ラインGT、32ラインGTの偶数行から16ライン入力を作成した。

Datasetのshapeはinput=[2,16,2688]、target=[2,32,2688]であり、TULIP forwardの出力は[1,2,32,2688]である。幅2650へは出力をcropする。

## 2. TULIP構造

Transformer本体、Encoder、Decoder、Attention、PatchMerging、PatchUnmerging、skip connectionは変更していない。

学習対象はTULIPの既存2ch出力であり、rangeとintensityを同時に復元している。評価では16→32で生成された奇数行のみを対象とした。

## 3. 基準学習条件

- 学習サンプル：160
- batch size：1
- epoch：50
- optimizer：AdamW
- learning rate：1e-4
- seed：0
- range scale：75m
- intensity transform：log1p
- loss対象：生成された奇数行のみ
- range weight：0.2
- intensity weight：1.0

基準checkpoint：outputs/waymo_16_to_32_r02_full_e50/checkpoint.pth

## 4. 高強度帯別重み学習

GTのraw intensityに応じて、log1p intensity lossへ次の重みを付けた。重みはGTから計算し、予測値からは計算していない。

| raw intensity | weight |
|---:|---:|
| 0〜0.5 | 1 |
| 0.5〜2 | 2 |
| 2〜5 | 4 |
| 5〜20 | 8 |
| 20〜100 | 12 |
| 100〜1000 | 16 |
| 1000以上 | 20 |

高強度帯別checkpoint：outputs/waymo_16_to_32_log1p_band_weighted_r02_e50/checkpoint.pth

lossは0.237838 → 0.117274となった。

## 5. Test評価結果

| 指標 | 基準モデル | 帯別重みモデル |
|---|---:|---:|
| range MAE | 1.326m | 2.206m |
| raw intensity MAE | 17.838 | 16.666 |
| log1p intensity MAE | 0.0681 | 0.0765 |

高強度Recall：

| しきい値 | 基準 | 帯別重み |
|---:|---:|---:|
| intensity >= 5 | 31.85% | 45.11% |
| intensity >= 20 | 35.25% | 49.75% |
| intensity >= 100 | 37.32% | 51.43% |
| intensity >= 1000 | 0% | 25.40% |

1000以上のraw MAEは14,596.6 → 13,255.3へ改善し、Precisionは約66.82%だった。

## 6. 高強度重みの課題

高強度帯の検出性能は改善したが、低強度帯とrange性能に悪化がある。

Testの0〜0.5帯raw MAEは0.0700 → 0.3430となった。高強度点を出しやすくなった一方、低強度画素への誤検出も増えている。

したがって、帯別重みモデルは現時点で最終採用モデルではない。高強度を学習へ反映する方向は有効だが、重みの強さとrange維持の調整が必要である。

## 7. 評価方法と注意点

全画素MAEだけでは、約95%を占める低強度画素に高強度の失敗が隠れる。そのため、raw/log1p MAE、強度帯別MAE、しきい値別Recall/Precision、range MAEを併記した。

既存の一部評価CLIはrange scaleを実メートルへ戻さず表示するため、今回の比較では実行時の一時評価でrangeを75倍して正しいメートル値を計算した。

比較JSON：outputs/high_intensity_comparison_log1p_band_weighted_e50.json

## 8. 次の作業

モデル構造を変更せず、次のいずれかを一条件ずつ比較する。

1. 帯別重みを半分程度に下げる
2. range weightを上げてrange悪化を抑える
3. 0〜5の重みを基準値に戻し、高強度帯だけを重視する

モデル選択では、全体MAEだけでなく、>=20、>=100、>=1000のRecall/Precision、強度帯別MAE、range MAEを使用する。

## 9. コード変更

TULIP本体、正式なDataset、正式な学習ソースコードは変更していない。学習・詳細評価は実行時の一時Python処理で行った。正式な一時スクリプトファイルは残していない。

zero-element tensors警告は表示されたが、学習・評価は完走しており、致命的エラーではない。
