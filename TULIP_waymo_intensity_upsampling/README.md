# TULIP Waymo Intensity Upsampling

Waymo Open DatasetのTOP LiDARを使い、TULIPによる距離・反射強度の
アップサンプリングを検証する独立プロジェクトです。

## 研究上の基準

前処理仕様は次の資料と実データを基準にします。

- `E:/shimizu.k.pdf` 第3章、特に3.2～3.3節
- `E:/JSPE202612__shimizu_k_final.pdf`
- 元データ: `W:/waymo_tf/perception_v1.4.3/individual_files`
- 32ライン成果物: `W:/32line`

既存の `TULIP_waymo_16to32` は参照実装としてのみ扱い、このプロジェクトから
importしません。

## 2つのデータ処理を分離する

### A. TULIP用の1フレームRange Image

```text
Waymo TFRecord
  -> TOP LiDAR first return [64,W,C]
  -> range/intensity/mask [64,W]
  -> 偶数ring [0,2,...,62] の32ラインGT
  -> さらに16ライン入力
  -> TULIPで16->32を学習・評価
```

この段階では車両姿勢による時系列重畳、動的物体除去、カメラ投影、
CLAHEを行いません。TULIPが復元する物理量を加工前の値で評価するためです。

### B. 清水研究の静的点群地図（後段）

```text
各フレームのTOP LiDAR
  -> 偶数ringで32ライン化
  -> 車両・歩行者・自転車などの動的点をBBoxで除去
  -> 車体近傍点を除去
  -> vehicle座標からglobal座標へ変換
  -> 全時刻を重畳してmap_static.npz
  -> FRONTカメラへz-buffer投影
  -> 99.5 percentile正規化
  -> masked histogram equalization
  -> CLAHE
  -> 反射強度画像
```

TULIP出力を将来この処理へ入力し、自己位置推定性能が改善するか評価します。

## 実行順序

### 0. 環境確認

```bash
source /home/wakamatsu/ITS/.venv_waymo/bin/activate
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
bash scripts/00_check_environment.sh
```

確認できること:

- TensorFlow、Waymo SDK、NumPy、PyTorchの利用可否
- データは変更しない

### 1. 元データと32ライン成果物の監査

```bash
source /home/wakamatsu/ITS/.venv_tulip/bin/activate
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
python tools/audit_shimizu_dataset.py \
  --waymo-root /mnt/w/waymo_tf/perception_v1.4.3/individual_files \
  --line32-root /mnt/w/32line \
  --output outputs/shimizu_dataset_audit.json
```

生成物:

- `outputs/shimizu_dataset_audit.json`
- TFRecord split件数
- `maps/training`、`cam_gray/training`のscene数
- TFRecordと32ライン成果物で共通するscene
- 代表`map_static.npz`の配列名・shape・dtype
- `render_profile.json`に記録された32ライン化条件

### 2. 同一フレームの64ライン・32ラインを抽出

```bash
source /home/wakamatsu/ITS/.venv_waymo/bin/activate
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling

python waymo_preprocess/export_one_frame_64_32.py \
  --tfrecord /mnt/w/waymo_tf/perception_v1.4.3/individual_files/training/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord \
  --frame-index 0 \
  --output-dir outputs/one_frame_000000
```

生成物:

- `waymo_top_frame_000000.npz`
  - `range_64`, `intensity_64`, `valid_mask_64`
  - `range_32`, `intensity_32`, `valid_mask_32`
  - `ring_ids_32=[0,2,...,62]`
- `metadata.json`
  - scene名、timestamp、shape、値域、有効画素数、使用ring

この32ラインは修論の「64ラインから走査線を1本おきに間引く」処理と同じ
偶数ring規則です。

### 3. テスト

```bash
source /home/wakamatsu/ITS/.venv_tulip/bin/activate
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
python -m unittest discover -s tests -v
```

確認できること:

- 64->32が偶数ringの厳密な部分集合であること
- 距離・反射強度・maskの対応が崩れていないこと
- 異常shapeを拒否できること

### 4. 引き継ぎmanifestを1フレーム1行へ変換

LoFTR用ペアmanifestをWaymoフレーム単位へ集約し、segment、frame index、
timestamp、元TFRecordを照合します。

```bash
source /home/wakamatsu/ITS/.venv_tulip/bin/activate
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
bash scripts/40_build_handover_frame_manifest.sh
```

TFRecord本文まで読む厳密照合はWaymo環境で明示的に実行します。

```bash
source /home/wakamatsu/ITS/.venv_waymo/bin/activate
bash scripts/40_build_handover_frame_manifest.sh --verify-tfrecord
```

詳細は [`docs/handover_frame_manifest.md`](docs/handover_frame_manifest.md) を参照してください。

### 5. 清水研究の評価GTをフレーム対応表へ変換

`manifest_cross_eval_gt_v3.csv` を `q_subset/q_seg/q_frame_index` で集約し、
カメラJSONのtimestampと元TFRecordの存在を確認します。

```bash
source /home/wakamatsu/ITS/.venv_tulip/bin/activate
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
bash scripts/35_build_shimizu_frame_manifest.sh
```

詳細は [`docs/inherited_frame_correspondence.md`](docs/inherited_frame_correspondence.md)
を参照してください。

### 6. manifestから64/32ラインを段階的に抽出

最初にTFRecordを読まないdry-runで計画を確認します。

```bash
source /home/wakamatsu/ITS/.venv_tulip/bin/activate
bash scripts/45_export_shimizu_frames_64_32.sh --dry-run --max-frames 1
```

実抽出はWaymo環境で1フレームから開始します。

```bash
source /home/wakamatsu/ITS/.venv_waymo/bin/activate
bash scripts/45_export_shimizu_frames_64_32.sh --max-frames 1 --resume
```

進捗まとめは [`docs/progress_2026-07-30.md`](docs/progress_2026-07-30.md) を
参照してください。

### 7. 64/32ラインを監査し16→32ペアを生成

```bash
source /home/wakamatsu/ITS/.venv_tulip/bin/activate
bash scripts/50_audit_and_prepare_shimizu_16_32.sh \
  --max-frames 5 --prepare-16 --resume
```

shape、偶数ringの完全一致、NaN/Inf、manifest identityを監査し、
`tulip_16_32.npz` を各フレームへ保存します。

16→32は補助検証に限定し、主タスクは32→64とします。

### 8. 32→64学習ペアを生成

```bash
source /home/wakamatsu/ITS/.venv_tulip/bin/activate
bash scripts/55_build_shimizu_32_to_64_pairs.sh --max-frames 5 --resume
```

`input_32 [32,W,2]`、`target_64 [64,W,2]` と追跡用index CSVを生成します。
引き継ぎ881フレームは既定で `localization_evaluation` とし、学習へ混ぜません。
詳細は [`docs/dataset_role_policy.md`](docs/dataset_role_policy.md) を参照してください。

## 次に実装するもの

1. 64/32/16の固定スケール可視化
2. intensity外れ値のclip/log変換比較
3. 16->32のring対応を可視化で検証
4. 32→64 TULIP Datasetアダプター
5. 1フレーム推論
6. scene単位のtrain/validation/test分割
7. 複数sceneの距離・反射強度評価
8. TULIP出力を静的点群地図生成へ接続

7/30

現在まで進んだこと
waymoの64ラインを取得
偶数から32ラインのGTを作成
range・intensity・valid maskの対応を維持
q_subset / q_seg / q_frame_index / timestamp を対応付け
元TFRecord欠損が0件であることを確認
代表1フレームの実抽出に成功
manifest生成処理をGitHubへpush済み
代表フレームでは以下を確認できています

64ライン: [64, 2650]
32ライン: [32, 2650]
64ライン有効画素: 152,616
32ライン有効画素: 76,240

工夫した点
TULIP用の加工前Range Imageと、清水研究の静的地図処理を分離
segment・frame index・timestampの三点でフレームを照合
Waymoのcontext名とTFRecordファイル名の表記差を正規化
同じカメラJSONをキャッシュして重複読込を削減
segment単位でTFRecordを1回だけ走査する設計
--dry-run と --max-frames で段階的に確認可能
--resume ではNPZ・metadata・identityが一致した場合だけスキップ
一時ファイルから置換する原子的保存
エラーを failures.csv に記録
TensorFlowとWaymo SDKを実処理時までimportしない