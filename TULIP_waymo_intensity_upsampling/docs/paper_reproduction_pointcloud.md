# TULIP論文再現: Waymo 16→32 点群復元

## 目的

既存TULIPのrange-image超解像をWaymo TOP LiDARの16→32ラインへ適用し、
推定された32ラインrange/intensity画像をWaymoの公式座標変換で車両座標点群へ戻す。
TULIPのTransformer本体、Encoder、Decoder、Attention、skip connectionは変更していない。

## 今回追加したファイル

- `tulip_adapter/waymo_pointcloud.py`
  - Waymo SDKのbeam inclination、TOP pixel pose、LiDAR extrinsic、vehicle poseを使用する。
  - `range_image_utils.extract_point_cloud_from_range_image`へ委譲し、角度を行番号から推測しない。
  - 出力点群はWaymo vehicle frame。
- `tools/reconstruct_waymo_16_to_32_pointcloud.py`
  - 既存Dataset、Waymo TFRecord、既存TULIP checkpointを接続する。
  - GT、TULIP、nearest、linearの点群とintensityをNPZ保存する。
- `tests/test_waymo_pointcloud.py`
  - 有効range mask、点群ペア誤差、空データ拒否を検証する。

## 入出力

入力は既存の16→32 pair indexとcheckpointである。

- input: `[1,2,16,2688]`
- target: `[1,2,32,2688]`
- native crop: `[32,2650]`
- point cloud: `[N,3]`
- intensity: `[N]`

評価は、GT validかつ生成odd rowかつTULIP rangeが有限・正の画素に限定する。
したがって、点群誤差は最近傍点探索によるChamfer距離ではなく、同一range-image画素に対応する
paired vehicle-frame point errorである。

## 実データ確認

使用したcheckpoint:

`outputs/waymo_16_to_32_tail_weighted_all_t20_w2/checkpoint.pth`

使用したサンプル:

- dataset role: `validation`
- sample index: `0`
- segment: `segment-10206293520369375008_2796_800_2816_800_with_camera_labels`
- frame: `0`
- native range-image: `[32,2650]`
- generated rows: `1,3,...,31`
- selected native ring IDs: `0,2,...,62`

保存先:

`outputs/waymo_16_to_32_tail_weighted_all_t20_w2/validation_pointcloud_sample0/`

保存キー:

- `points_gt_vehicle`: `[39354,3]`, float32
- `intensity_gt`: `[39354]`, float32
- `points_tulip_vehicle`: `[39354,3]`, float32
- `intensity_tulip`: `[39354]`, float32
- nearest / linearも同じ形式で保存

結果:

| 方法 | 有効点数 | 座標MAE [m] | Euclidean MAE [m] | intensity MAE |
|---|---:|---:|---:|---:|
| TULIP | 39354 | 1.3151 | 3.0053 | 0.6410 |
| nearest | 38813 | 0.5672 | 1.2750 | 1.2100 |
| linear | 39312 | 0.6184 | 1.4176 | 1.4793 |

これは1サンプルの確認であり、最終性能評価ではない。rangeについてはこのサンプルでは補間ベースラインが良く、
intensityについてはTULIPが良い結果になった。全validation/testでの判定には既存のgenerated-row評価を使う。

## 依存関係と再現性

Waymo環境にはTensorFlow、Waymo Open Dataset、PyTorchがあるが、`timm`と`chamfer_distance`がない。
そのため今回のsmoke testでは、これら2つだけを実行時stubにした。Waymoのrange-image parsing、
beam inclination、pixel pose、extrinsic、vehicle pose、点群変換は実SDKを使用している。
このため、この確認は「実Waymo変換 + 実TULIP forward + utility-only stub」の結果であり、
依存関係を整備した正式環境での最終確認とは区別する。

## 論文再現範囲

再現できた範囲:

1. LiDARをrange imageとして扱う。
2. Swin/TULIP backboneで低解像度range imageから高解像度range imageを推定する。
3. LiDAR向けの非正方形patch/windowを利用する。
4. 推定range imageを点群へ戻し、intensityを点属性として保持する。

Waymo向け拡張であり、論文そのものと同一ではない範囲:

1. Waymoの16→32ライン設定。
2. range + raw intensityの2チャンネル入力・出力。
3. Waymoの幅2650とモデル用2688 padding/crop。
4. Waymo公式のmotion-compensated vehicle-frame変換。
5. range/intensityのmask付き評価。

未実装:

- Chamfer距離など論文と同一条件の点群評価。
- 可視光カメラ画像とのマッチング。
- LoFTR、PnP、カメラ投影。
- 多数フレーム・全testデータでの点群評価。

## 次の手順

1. 同じcheckpointでvalidation/testの複数サンプルを点群化する。
2. generated odd rowsだけで、range、intensity、paired 3D errorを集計する。
3. 高intensity帯と通常intensity帯を分けて点群誤差を報告する。
4. その後、必要なら公式依存関係を整えた環境でstubなしのforwardを確認する。

---

作成日（ファイル更新日時から推定）: 2026-08-20
