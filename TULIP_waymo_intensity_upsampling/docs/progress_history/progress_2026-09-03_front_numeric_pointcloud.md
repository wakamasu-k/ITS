# Waymo FRONT画像・数値・点群評価記録 — 2026-09-03

## 対象

前回の`progress_2026-09-01_kitti_style_front_overlays.md`以降に行った、
FRONTカメラ画像確認、raw数値評価、Pred点群の3次元評価を記録する。

保護対象の`TULIP_backup`、`handover_shimizu`、`/mnt/w/32line`は変更していない。
学習済みcheckpointおよびraw入力データも変更していない。

## 使用条件

- プロジェクト: `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling`
- index: `outputs/training_pairs_16_to_32_full/index.csv`
- checkpoint: `outputs/waymo_16_to_32_full_e300/checkpoint.pth`
- checkpoint epoch: `299`
- dataset role: `test`
- 評価フレーム数: `40`
- range scale: `75.0 m`
- intensity transform: `log1p`
- TULIP入力: `2ch, 16ライン`
- GT: `2ch, 32ライン`
- 点群座標: Waymo vehicle frame

## 1. FRONT画像生成

### 生成先

- `outputs/waymo_16_to_32_full_e300/final_front_sample0/`
- `outputs/waymo_16_to_32_full_e300/camera_compare_front_sample0_kitti_style_intensity_normal/`
- `outputs/waymo_16_to_32_full_e300/camera_compare_front_sample0_kitti_style_depth_piecewise/`

### 表示条件

- カメラ: `FRONT`
- カメラ画像サイズ: `[1280,1920,3]`
- 通常intensity表示: `0..1`
- 通常強度用log1p表示: `0..20`
- 外れ値確認用log1p表示: `0..20000`の既存出力を保持
- 距離: `4..75 m`
- 距離mapping: `piecewise`
- 近距離区間: `4..20 m`, log1p + gamma `0.75`
- 遠距離区間: `20..75 m`, 別色域へ線形展開
- point size: `3`
- overlay alpha: `0.65`

### 投影点数

| mode | カメラ内投影点 | Xc最小 | Xc最大 | intensity最小 | intensity最大 |
|---|---:|---:|---:|---:|---:|
| GT | 9,248 | 6.194 m | 74.335 m | 0.000220 | 11,584.003 |
| Low | 4,654 | 6.300 m | 74.335 m | 0.000220 | 6,079.997 |
| Pred | 9,248 | 4.110 m | 74.181 m | 0.000000 | 427.644 |

### 画像検証

- GT・Low・Predのcolor/overlay/legend/maskを再読込できた
- 画像shapeは`1280x1920x3`
- maskは`0,255`のみの二値画像
- overlayはRGB画像を背景として色点を重畳
- point size `3`により、1画素描画より点を視認しやすくした
- 距離40 m超の点を同じ色へ飽和させないpiecewise表示を追加した

Waymoのカメラ前方深度は`Xc`であり、KITTIの`Z_cam`およびLiDAR原点からの
放射状rangeとは異なる。画像表示上もこの意味を維持した。

## 2. raw数値評価

出力先:

`outputs/waymo_16_to_32_full_e300/final_numeric_test40/summary.json`

40 testフレームをraw range/raw intensity単位で評価した。

| 方法 | range MAE | intensity MAE |
|---|---:|---:|
| TULIP | 0.659763 m | 12.958549 |
| Nearest | 1.385502 m | 9.236806 |
| Linear | 1.697219 m | 10.729665 |

rangeについてはTULIPがNearestおよびLinearを下回った。intensity全体MAEは
Nearestより大きいが、これは`1000以上`の巨大外れ値の影響を強く受ける。

## 3. intensity帯別評価

出力先:

`outputs/waymo_16_to_32_full_e300/final_numeric_test40/intensity_bins.json`

TULIPのintensity MAEは次の傾向になった。

- `0..0.5`: `0.123919`
- `0.5..1`: `0.438721`
- `1..2`: `1.114153`
- `2..5`: `2.856546`
- `5..20`: `9.193715`
- `20..100`: `46.659809`
- `100..1000`: `386.081058`
- `1000以上`: `12104.683099`

通常強度から中程度の強度では復元できている。`1000以上`ではNearestおよび
Linearより悪く、巨大外れ値の再現は未解決である。

## 4. Pred点群の3次元評価

出力先:

`outputs/waymo_16_to_32_full_e300/final_pointcloud_test40/summary.json`

評価maskは生成された奇数行について、GTとPredの有限・正のrangeが共通する
点を使用した。40フレーム、合計`1,491,968`点を評価した。

| 方法 | coordinate MAE | Euclidean MAE | intensity MAE |
|---|---:|---:|---:|
| TULIP | 0.542220 m | 1.223551 m | 17.836247 |
| Nearest | 0.634197 m | 1.412633 m | 19.218291 |
| Linear | 0.987254 m | 2.238441 m | 18.357401 |

TULIPはNearestと比較して、coordinate MAEが約14.5%、Euclidean MAEが約13.4%
改善した。Linearと比較すると、両方とも約45%改善した。intensity MAEも
Nearestより約7.2%、Linearより約2.8%改善した。

したがって、rangeに基づく3D形状復元は単純補間より良い。一方、Euclidean
MAEが約1.22 mであるため、精密な3D位置合わせとして十分かは、カメラ投影後の
画素ずれと合わせて判断する必要がある。

## 5. 警告と実行環境

点群評価では、`.venv_waymo`上でTensorFlowを実行した。TULIP importに必要な
`timm`と`chamfer_distance`はruntime-only stubを使用した。ChamferDistance
自体は呼び出していない。モデルforward、rangeの75 m復元、点群座標変換、
評価計算は実行されている。

表示された以下のTensorFlowメッセージは、今回の処理完走を妨げなかった。

- AVX2/FMA: CPU最適化に関する通知
- TensorRT未検出: 非致命的警告
- NUMA node読込失敗: 非致命的警告
- GPUライブラリ不足: TensorFlowのCPU fallback。今回のFrame読込は完走
- `Placeholder/_0 INVALID_ARGUMENT`: TFRecordDataset内部のDEBUG INFO

## 6. 現時点の結論

### 良好な点

- 16→32のrange復元は、Nearest・Linearより良い
- 3D点群の座標誤差も単純補間より小さい
- FRONTカメラへの投影処理は40フレーム評価済み
- 通常intensity帯ではTULIPが改善する帯域がある
- GT・Low・Predの画像比較環境が整った

### 残る問題

- intensity `1000以上`の外れ値を再現できていない
- intensity全体MAEは外れ値により大きく見える
- Euclidean MAE `1.22 m`がマッチング用途で許容できるか未判定
- 今回の評価はtest 40フレームであり、全Waymo規模ではない
- point size 3は可視化専用で、点数や性能を改善するものではない

## 7. 次の作業

1. FRONTのraw投影画素ずれと重なり率を、現在のe300 checkpointで確認する。
2. 近距離・遠距離別にrange/3D誤差を分けて確認する。
3. intensityは`0..1`、`1..20`、`20以上`、`1000以上`を分離して報告する。
4. 問題がなければ、Pred 32ライン点群を清水氏のFRONTマッチング処理へ渡す。
5. LoFTR/PnPの結果はTULIPの点群復元評価と分離して記録する。

現時点では、TULIP Transformer本体を変更する必要性は確認されていない。

