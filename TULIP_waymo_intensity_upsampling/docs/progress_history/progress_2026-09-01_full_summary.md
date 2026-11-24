# TULIP Waymo 16-to-32 progress summary

作成日: 2026-09-01

## 目的

KITTI向けTULIPの基本構造を維持し、Waymo TOP LiDARの16ライン入力から32ライン相当のrange・intensityを復元する。最終的には、補完したLiDAR反射強度を可視光カメラ画像と対応付ける。

## 重要な前提

- TULIP_backupのTransformer本体は変更していない。
- Transformer block、Encoder、Decoder、Attention、PatchMerging、PatchUnmerging、skip connectionは変更していない。
- Waymo用モデルは既存1ch checkpointの変換ではなく、2chモデルを新規学習した。
- Waymoの入力はTOP LiDAR、第1return、native range image `[64, 2650, 4]`。
- channel 0はrange、channel 1はintensity。
- 64ラインから偶数ring `[0,2,...,62]` を32ラインGTとして使用した。
- 32ラインGTの偶数行 `[0,2,...,30]` を16ライン入力として使用した。
- rangeはLiDAR原点からの放射状距離、Waymoカメラ座標ではXcが前方軸である。

## Phase 1: Waymo native range image確認

実装・確認した内容:

- TFRecordからFrameを読み込み。
- `frame_utils.parse_range_image_and_camera_projection()`を利用。
- TOP LiDARの第1returnを取得。
- range/intensityをfloat32 `[H,W]`へ変換。
- rangeが無効または非有限の画素をmask=Falseとし、range/intensityを0にした。
- 64、32、16ラインのNPZ、metadata JSON、確認PNGを生成。

確認結果:

- native shape: `[64, 2650, 4]`
- range/intensity: float32
- mask: bool
- 32ライン: `[32, 2650]`
- 16ライン: `[16, 2650]`
- NaN/Inf: 0
- ring IDs: 32ライン `[0,2,...,62]`、16ライン `[0,4,...,60]`
- Phase 1 tests: 4 passed

## Phase 2: intensity統計・外れ値確認

複数TFRecord・複数フレームを逐次集計し、valid画素だけでrangeとintensityの統計を取得した。intensityは大部分が0〜0.5付近だが、ごく少数の1000以上の値と数十万級の値が存在することを確認した。

確認した方針:

- 0埋め無効値、NaN、Infは統計から除外。
- rangeとintensityは同じvalid位置を使用。
- 正規化値やclip値をこの時点では固定しない。
- log1pはheavy tail対策の実験条件として扱い、raw値で評価する場合はinverse transform後に評価。

重要な問題:

- raw intensity MAEは少数の巨大外れ値に強く支配される。
- 学習結果のpred intensityは、通常域には近づくが、1000以上や数万以上の値をほぼ再現できない。
- したがって、通常域、100〜1000、1000以上を分離した評価が必要。

## Phase 3: raw 32-line GT保存

Waymo TFRecordからraw 32ラインGTを保存する処理を作成した。

- 出力配列: `[32, 2650, 2]`
- dtype: float32
- channel 0: raw range
- channel 1: raw intensity
- normalization: none
- clip、log1p、uint8化、2688 paddingは保存時に行わない。
- sidecar JSON、manifest、failuresを生成。

1 TFRecordからの5フレーム確認、全フレーム確認、shape・dtype・有限性・valid率・metadata整合性を確認した。

## Phase 4A: 16-to-32 Dataset

学習用の変換を実装・確認した。

- raw `[32,2650,2]`を読み込む。
- 32ラインGTの偶数行から16ライン入力を作る。
- rangeを75 mスケールへ変換。
- intensityはrawまたはlog1pを選択可能。
- HWCからCHWへ変換。
- 幅2650から2688へcircular padding。
- PyTorch float32 tensorで返す。

確認shape:

- input: `[2,16,2688]`
- target: `[2,32,2688]`
- batch input: `[B,2,16,2688]`
- batch target: `[B,2,32,2688]`
- Dataset tests: 4 passed

作成した学習ペアは合計240件:

- train: 160
- validation: 40
- test: 40

全240件でshape、dtype、mask、ring ID、NaN/Infを監査した。

## TULIP forward互換性

Waymo用設定でforward smoke testを実施した。

- input: `[1,2,16,2688]`
- target: `[1,2,32,2688]`
- output: `[1,2,32,2688]`
- crop後: `[1,2,32,2650]`
- loss: finite
- output NaN/Inf: なし

幅2688は、patch embedding、patch merging、window処理、PixelShuffleを通過させるためのpadding幅として使用した。入力2650列についてはDataset側でcircular paddingし、出力後に先頭2650列へcropする。

一時monkey patchによる検証では、patch embeddingの幅方向paddingとPixelShuffle倍率をWaymo用条件に合わせることで、Transformer本体を変更せずforwardできることを確認した。

## 学習・評価

主な条件:

- 2ch range + intensity
- range scale: 75.0 m
- intensity transform: log1p
- device: CUDA
- TULIP基本構造は維持

代表checkpoint:

`outputs/waymo_16_to_32_full_e300/checkpoint.pth`

学習結果:

- train 160 samples、300 epochsでlossは減少。
- validation 40 frames: range MAE `0.5598 m`、intensity MAE `6.5721`
- test 40 frames: range MAE `0.6598 m`、intensity MAE `12.9585`

生成奇数行だけを評価した場合、rangeは概ね1 m前後まで復元できた。一方、intensityは通常域では改善するが、1000以上の外れ値は再現できず、raw MAEが大きくなる。

比較した条件:

- raw intensity
- log1p intensity
- range/intensity loss weight変更
- tail-weighted loss
- intensity priority

観察された傾向:

- range scaleを適切にした条件ではrange復元が大きく改善。
- intensityのloss重みを上げても巨大外れ値の再現は難しい。
- tail weightingは評価帯によって改善するが、通常域を悪化させる場合がある。
- nearest・linear補間は基準値として必要だが、全強度帯でTULIPを上回るわけではない。

## 点群復元

学習済みTULIPの出力rangeをWaymoのbeam inclination・azimuthでvehicle frameの3D点へ復元する処理を確認した。

評価では以下を比較した:

- TULIP predicted point cloud
- nearest interpolation
- linear interpolation

点群評価で確認したこと:

- predicted rangeを用いた点群は生成できる。
- 座標誤差とユークリッド誤差を計算できる。
- intensityも同じ点位置で比較できる。
- 高intensity外れ値が点群intensity MAEを支配する。

### 点群評価の逆正規化確認

既存の`reconstruct_waymo_16_to_32_pointcloud.py`をsample 0で読み取り実行したところ、rangeを75 mスケールからraw meterへ戻さずに点群変換していることが判明した。そのため、このツールが出力した座標MAE 7.70 m、ユークリッドMAE 17.29 mは採用しない。既存の`evaluate_waymo_16_to_32_scaled_pointcloud.py`にはrangeを75倍する`restore_domain`があり、こちらを正しい評価経路として使用した。

e300 checkpoint、test 5フレーム、range scale 75.0 m、intensity log1pでの正しい結果:

- TULIP coordinate MAE: `0.6336 m`
- TULIP Euclidean MAE: `1.4254 m`
- TULIP intensity MAE: `12.7658`
- nearest Euclidean MAE: `1.6177 m`
- linear Euclidean MAE: `2.4474 m`
- paired TULIP points: `185,544`
- 全5フレームのpoint arrayはfloat32かつfinite。

sample 0のraw保存先:

`outputs/waymo_16_to_32_full_e300/pointcloud_test5_corrected/sample_000000/`

この評価では、TULIPのrange予測を実際のmeterへ戻してからWaymo公式のrange-image-to-point-cloud変換を適用している。したがって、今後の点群評価はこのscaled経路を基準にする。

test 40フレームへ拡張した結果:

- TULIP coordinate MAE: `0.5422 m`
- TULIP Euclidean MAE: `1.2236 m`
- TULIP intensity MAE: `17.8362`
- nearest Euclidean MAE: `1.4126 m`
- linear Euclidean MAE: `2.2384 m`
- TULIP paired points: `1,491,968`
- 処理フレーム: `40/40`
- 保存NPZ: `40/40`
- 全点群配列のshape・float32・finite確認: PASS

保存先:

`outputs/waymo_16_to_32_full_e300/pointcloud_test40_corrected/`

この40フレーム評価では、点群のrange geometryはTULIPがnearestとlinearを上回った。一方、raw intensity MAEは依然として大きく、intensity外れ値の扱いは未解決である。

## 個別GT・Low・Pred点群の保存

GT、16ラインLow、TULIP Predを別々のvehicle-frame point cloudとして保存する新規ツールを追加した。

`tools/export_waymo_16_to_32_gt_low_pred_pointcloud.py`

test 5フレームで確認した結果:

- 出力: `5/5`
- failures: `0`
- GT点数: 約73,916〜74,352点/フレーム
- Low点数: 約37,049〜37,171点/フレーム
- Pred点数: 約73,916〜74,352点/フレーム
- XYZ shape: `[N,3]`
- XYZ/intensity dtype: float32
- NaN/Inf: 0
- rangeのXYZ最大絶対値: 約76 m以内

生成行（奇数行）だけのGT/Pred比較は、5フレームでrange geometryのEuclidean MAEが`1.395〜1.476 m`、intensity MAEが`12.35〜13.01`だった。

intensity帯の確認では、GTに1000以上の画素が各フレーム約90〜102点存在したのに対し、Predの1000以上は全5フレームで0点だった。通常強度域の点群復元は成立しているが、高強度反射の再現は未解決である。

保存先:

`outputs/waymo_16_to_32_full_e300/gt_low_pred_pointcloud_test5/`

初回実行では新規ツールのmask幅とDatasetのrange scale指定漏れを検出した。修正後に再実行し、上記の5/5 PASSを得た。初回失敗出力は採用しない。

test 40フレームへ拡張した結果:

- 出力: `40/40`
- failures: `0`
- GT total points: `2,985,442`
- Low total points: `1,493,472`
- Pred total points: `2,985,440`
- 全配列のfinite確認: PASS
- 生成奇数行のrange Euclidean MAE平均: `1.2248 m`
- 生成奇数行のintensity MAE平均: `17.8175`
- GT intensity 1000以上: `3,174`点
- Low intensity 1000以上: `1,406`点
- Pred intensity 1000以上: `0`点
- GT intensity最大: `372,736.06`
- Pred intensity最大: `445.73`

保存先:

`outputs/waymo_16_to_32_full_e300/gt_low_pred_pointcloud_test40/`

この結果から、Predは16ラインLowの約2倍の点数を持つ32ライン相当の点群を生成できている。一方、1000以上の高強度反射はPredで再現されていないため、点密度・range geometryとintensityダイナミックレンジは分けて評価する必要がある。

## 個別range・intensity画像

test sample 0について、GT・Low・Predを個別画像として保存した。

保存先:

`outputs/waymo_16_to_32_full_e300/separate_test_sample0/`

生成画像:

- `range_gt.png`, `range_low.png`, `range_pred.png`
- `intensity_gt_linear.png`, `intensity_low_linear.png`, `intensity_pred_linear.png`
- `intensity_gt_log1p.png`, `intensity_low_log1p.png`, `intensity_pred_log1p.png`

全画像の元データshapeは`[32,2650]`。range表示は0〜75 m、intensity linear表示は0〜20、intensity log1p表示はraw 0〜20,000を共通スケールとした。PNGは確認用であり、raw値や学習データを置き換えない。

sample 0のvalid pixel数はGT 74,352、Low 37,100、Pred 74,352。intensity最大値はGT 21,760、Low 18,560、Pred 432.3であり、個別画像でもPredの高強度不足を確認できる。

## カメラ投影

Waymo vehicle frameから各カメラ座標へ変換し、`Xc > 0`を前方条件として投影した。

既存のcamera evaluationでは、oracle（GT range + predicted intensity）とpredicted rangeを分けてgeometry影響を確認した。FRONT、FRONT_LEFT、FRONT_RIGHTすべてで投影処理は完走した。FRONT_RIGHTはpixel shiftが比較的大きいが、投影点数・投影率は大きく崩れていない。

## 純粋なGT・Low・Pred比較

oracleと混同しないため、次の3条件を独立投影する新規ツールを追加した。

`tools/visualize_waymo_16_to_32_camera_gt_low_pred.py`

- GT: GT range + GT intensity
- Low: 16ラインrange + intensityを32-grid偶数行へ配置
- Pred: TULIP predicted range + intensity

test sample 0で3カメラを確認した。

| camera | GT points | Low points | Pred points |
|---|---:|---:|---:|
| FRONT | 9248 | 4654 | 9248 |
| FRONT_LEFT | 10500 | 5352 | 10504 |
| FRONT_RIGHT | 10923 | 5575 | 10925 |

全ケースでRGB画像は `[1280,1920,3]`、投影mapは `[1280,1920]`。raw NPZの座標・深度・intensityはfloat32、maskはbool、有限性確認に合格した。

この結果は「学習後TULIP出力を点群へ戻し、カメラへ投影できる」ことを確認するものである。ただし、現在のツールはGT・Low・Predを1枚へまとめた比較画像を生成する。最終表示で必要なGT、Low、Predの個別3枚は今後の表示形式として追加検討する。

## 現在できていること

- Waymo TFRecord読み込み
- TOP LiDAR第1returnのrange image取得
- 64→32→16のring生成
- raw GT保存
- DatasetとDataLoader
- 2ch TULIP forward
- 2ch Waymo学習
- checkpoint評価
- range/intensityの帯域別評価
- 点群復元
- 複数カメラ投影
- RGBとLiDAR反射強度の投影比較

## まだ完了していないこと

- Waymo全データを用いた本格的な最終学習
- 複数seed・複数シナリオによる再現性確認
- intensity外れ値の正式な取り扱い方針の確定
- 個別GT・Low・Pred画像の最終形式
- 完成点群の正式な全件保存と再利用API
- RGB特徴と反射強度の実マッチング
- LoFTR、PnPなど清水氏の後段手法との統合
- 論文のKITTI実験値との厳密な再現比較

## 現在の環境上の注意

TensorFlowによるTFRecord読み込みは`.venv_waymo`、TULIPのtimm/chamfer依存は`.venv_tulip`で確認してきた。カメラ・点群の一部確認では、一時的なruntime stubを使用した。これは処理経路の確認であり、正式な統合環境の確認とは区別する。

## 次の優先順位

1. RGBとの位置関係を個別GT・Low・Pred画像で目視確認する。
2. FRONT、FRONT_LEFT、FRONT_RIGHTを分け、通常強度帯・高強度帯・距離帯別の評価を保存する。
3. Pred自身のvalid maskで点密度と誤検出を確認する。現在のGT/Pred比較はGT-valid maskを基準にしているため、この確認が必要である。
4. 学習条件を一つに固定し、追加シナリオで再現性を確認する。
5. その後に全データ学習へ進む。

## 関連記録

詳細な過去記録は`docs/`直下に残している。本ファイルはそれらを削除・置換せず、進捗確認用の統合記録として`docs/progress_history/`へ追加した。
