# TULIP Waymo Intensity Upsampling

Waymo Open Datasetの距離・反射強度画像をTULIPでアップサンプリングするための、
独立した研究用プロジェクトです。

このフォルダは次の既存フォルダをimportせず、別タスクとして管理します。

- `TULIP_backup`
- `TULIP_waymo_16to32`
- `handover_shimizu`

## 最初に行うこと

いきなり学習を始めず、まずWaymoから取り出した1フレームの内容を検査します。
入力NPZは次の配列を持つものとします。

- `range`: 距離画像 `[H, W]`、単位m
- `intensity`: 反射強度画像 `[H, W]`
- `valid_mask`: 有効画素マスク `[H, W]`（省略時は `range > 0`）

### 1. 環境確認

```bash
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
bash scripts/00_check_environment.sh
```

できること:

- Pythonの場所とバージョンを確認
- NumPy、PyTorch、TensorFlow、Waymo関連パッケージの有無を確認
- ファイルの作成やデータ変更は行わない

### 2. 1フレーム検査

```bash
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
python tools/inspect_waymo_npz.py \
  --input /path/to/frame_000000.npz \
  --output outputs/frame_000000_report.json
```

できること:

- 距離・反射強度・マスクの形状一致を確認
- NaN、Inf、負の距離、マスク外の値を検出
- 有効点数、値域、平均、標準偏差、パーセンタイルを集計
- 結果をJSONへ保存

### 3. テスト

```bash
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
python -m unittest discover -s tests -v
```

できること:

- 検査コードが正常データを正しく集計できることを確認
- 形状不一致をエラーとして検出できることを確認

## 今後の順番

1. Waymo TFRecordからTOP LiDARのrange/intensityを1フレーム抽出
2. native 64ライン画像を固定スケールで可視化
3. 64ラインからGTと低解像度入力を同一画素規則で作成
4. 低解像度を元の角度行へ戻せることを確認
5. TULIP用Datasetと入出力アダプターを実装
6. 1フレーム推論
7. 複数sceneで学習・評価分割を作成
8. 距離と反射強度を別々のmask付き指標で評価

学習コードは、手順1～4でデータの対応関係を確認してから追加します。
