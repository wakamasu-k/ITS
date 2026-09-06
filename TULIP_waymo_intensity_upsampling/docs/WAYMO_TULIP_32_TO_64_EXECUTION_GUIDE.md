# Waymo版TULIP 32→64ライン実行手順

## 目的と重要な前提

Waymo TOP LiDARの32ライン距離画像を入力し、同一フレームのネイティブ64ラインをGTとしてTULIPを学習・評価する。距離baselineを確立した後で反射強度を追加し、最後に清水研究の静的点群地図・画像照合へ接続する。

- 観測行: `[0,2,...,62]`
- 生成・主評価行: `[1,3,...,63]`
- 偶数行は推論後も観測値を保持する
- 既存の16→32コードは補助実験用で、この手順では使わない

KITTIの`.bin`はXYZIなので球面投影が必要だが、Waymo TFRecordにはTOP LiDARのネイティブRange Image `[64,2650,channels]` がある。したがって学習前にXYZへ変換して再度球面投影しない。再投影は、XYZしかない入力を共通格子へ載せる場合などに限る。推論後のRange ImageからXYZへ戻す処理は「逆投影」である。

## 使用場所

```text
プロジェクト: /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
TULIP本体:    /home/wakamatsu/ITS/TULIP_backup/tulip
Waymo TFRecord:/mnt/w/waymo_tf/perception_v1.4.3/individual_files
既存成果物:   /mnt/w/32line
```

基本の開始操作:

```bash
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
source /home/wakamatsu/ITS/.venv_waymo/bin/activate
```

`.venv_waymo`と`.venv_tulip`を同じシェルで重ねてactivateしない。

## 全体の順番

```text
環境/GPU確認
 → 評価用881フレームを1～5件抽出・監査
 → 学習用segmentを評価用から分離
 → 学習用32/64ペア作成
 → forward確認
 → 1～2件overfit
 → validation評価
 → 規模拡大
 → 反射強度追加
 → XYZ化・清水研究へ接続
```

## 1. 環境とGPU確認

```bash
bash scripts/00_check_environment.sh
source scripts/01_activate_waymo_gpu.sh
TF_CPP_MIN_LOG_LEVEL=2 python tools/check_tensorflow_gpu.py
python -m unittest discover -s tests -v
```

`tensorflow_gpus`にGPUが表示され、`operation_device`がGPU、`tensorflow_gpu_test=passed`、全テスト成功なら合格。GPUテストが通る場合、TensorRTやWSL NUMAの警告は致命的ではない。

## 2. 清水研究の評価manifest

```bash
bash scripts/35_build_shimizu_frame_manifest.sh
```

確認済み結果は、元GT 16,963 pair、一意なWaymoフレーム881件、TFRecord欠損0件。出力は`outputs/shimizu_frame_manifest.csv`。

この881件は`localization_evaluation`用であり、学習へ混ぜない。同一走行segmentがtrainへ入ると評価リークになる。

## 3. 評価用64/32データを1件、次に5件抽出

```bash
bash scripts/45_export_shimizu_frames_64_32.sh --dry-run --max-frames 1
bash scripts/45_export_shimizu_frames_64_32.sh --max-frames 1 --resume
bash scripts/45_export_shimizu_frames_64_32.sh --max-frames 5 --resume
```

出力: `outputs/shimizu_frames_64_32/`

確認事項:

- 64ラインのrange/intensity/maskが`[64,2650]`
- 32ラインのrange/intensity/maskが`[32,2650]`
- 32ラインが64ラインの偶数行と完全一致
- NaN/Infなし
- manifestとmetadataのsegment/index/timestampが一致
- intensityのmin/median/percentile/maxを記録

`--resume`はNPZとmetadataが揃いidentityも一致するときだけ再利用する。

## 4. 評価用32→64ペア作成

```bash
bash scripts/55_build_shimizu_32_to_64_pairs.sh --max-frames 1
bash scripts/55_build_shimizu_32_to_64_pairs.sh --max-frames 5 --resume
```

出力:

```text
outputs/shimizu_pairs_32_to_64/index.csv
outputs/shimizu_pairs_32_to_64/failures.csv
```

偶数行完全一致、奇数行が生成対象、未解決failureなしを確認する。

## 5. 学習用segmentを評価用から分離

```bash
bash scripts/60_build_training_segment_manifest.sh --dry-run
bash scripts/60_build_training_segment_manifest.sh
```

出力: `outputs/training_segment_manifest.csv`

train/validation/test/localization_evaluationをsegment単位で分け、同じsegmentを複数roleへ入れない。フレーム単位のランダム分割は禁止する。

## 6. 学習用フレーム抽出

```bash
bash scripts/65_export_training_frames_64_32.sh --dry-run --max-frames 1
bash scripts/65_export_training_frames_64_32.sh --max-frames 1 --resume
bash scripts/65_export_training_frames_64_32.sh --max-frames 5 --resume
```

出力は`outputs/training_frames_64_32/`、`index.csv`、`failures.csv`。5件が通ってから20件、1 segment、複数segmentの順に増やす。

## 7. 学習用32→64ペア作成

```bash
bash scripts/70_build_training_32_to_64_pairs.sh --dry-run --max-frames 1
bash scripts/70_build_training_32_to_64_pairs.sh --max-frames 5 --resume
```

出力:

```text
outputs/training_pairs_32_to_64/index.csv
outputs/training_pairs_32_to_64/failures.csv
```

Datasetの意味は`input=[channels,32,2650]`、`target=[channels,64,2650]`、偶数行=観測、奇数行=生成対象を維持する。

## 8. TULIP forward確認

```bash
bash scripts/75_check_range_only_tulip_forward.sh \
  --dataset-role train --device cuda
```

幅2650はpatch/window条件に合わないため、内部で循環paddingして2816にし、forward後に2650へcropする。paddingをlossへ含めない。入力`[B,1,32,2650]`、出力/GT`[B,1,64,2650]`、NaN/Infなし、有効奇数行だけがloss対象なら合格。

## 9. 距離のみ1～2件overfit

距離120 m正規化版を使う。

```bash
bash scripts/82_overfit_normalized_range_tulip.sh \
  --dataset-role train --max-samples 2 --epochs 20 --device cuda
```

loss低下を確認してから:

```bash
bash scripts/82_overfit_normalized_range_tulip.sh \
  --dataset-role train --max-samples 2 --epochs 300 --device cuda
```

checkpointは`outputs/range_only_normalized_overfit/checkpoint.pth`。同じ2件へoverfitできなければ、全体学習へ進まずmask、正規化、shape、padding、学習率を調べる。

## 10. 数値評価

```bash
# checkpoint再読込みとtrain再現
bash scripts/92_evaluate_normalized_range_tulip.sh \
  --dataset-role train --max-samples 2 --device cuda

# 未学習validationの小規模評価
bash scripts/92_evaluate_normalized_range_tulip.sh \
  --dataset-role validation --max-samples 2 --device cuda

# 問題がなければ拡大
bash scripts/92_evaluate_normalized_range_tulip.sh \
  --dataset-role validation --max-samples 40 --device cuda
```

出力: `outputs/range_only_normalized_evaluation/`

同じ有効奇数行でTULIP、nearest、linearのMAE[m]・RMSE[m]を比較する。距離帯別MAE、valid数、負値率も保存する。疑似カラーがきれいなだけでは成功と判断しない。

## 11. 全体学習へ進むゲート

次をすべて満たした後、5件→20件→1 segment→複数segment→全対象へ増やす。

1. 64/32 shapeと偶数ringが完全一致
2. segment/index/timestamp一致
3. segmentリークなし
4. 1～2件overfit成功
5. TULIPと補間baselineが同一mask
6. checkpoint再読込み後も評価値を再現
7. NaN/Inf、負距離、欠損率を記録可能

## 12. 距離成功後に反射強度を追加

- `-1`等の無効値を変換前にmask
- clip上限はtrain統計だけで決定
- 候補変換: `log1p(clip(x,0,cap))/log1p(cap)`
- 距離lossとintensity lossを個別記録
- 奇数行の有効画素だけで評価
- raw/変換後/逆変換後の統計を保存

```text
total_loss = range_weight * masked_range_loss
           + intensity_weight * masked_intensity_loss
```

`cap`とloss weightはvalidationで選び、testは使わない。

## 13. 点群化と清水研究への接続

TULIP出力の64行を、Waymoのbeam inclinations、azimuth、LiDAR extrinsic、pixel pose/frame pose、valid maskでXYZへ逆投影する。

```text
32ライン入力
 → 奇数行をTULIP予測
 → 偶数=観測、奇数=予測の64行
 → XYZへ逆投影
 → 静的点群地図
 → カメラ投影・画像照合
 → 既存32ラインとの自己位置推定比較
```

評価順はRange Imageの精度→XYZ幾何→自己位置推定効果とする。

## 14. よくある問題

- GPU未使用: 新しいシェルで環境を1つだけactivateしGPU検査を再実行。
- OOM: batch/max samplesを1へ。2816幅paddingもメモリを使う。
- 負距離: 可視化だけで隠さず負値率を記録し、正規化・loss・出力制約を比較。
- 画像は良いがMAEが悪い: log表示や色clipは見た目だけ。元のm単位で評価。
- 評価が良すぎる: segmentリーク、881件のtrain混入、偶数行の評価混入を確認。
- intensity loss異常: 無効値mask、percentile、clip/log、チャンネル別lossを確認。

## 15. 実行とデバッグの分担

全データ抽出、長時間学習、全881件処理はユーザー側ターミナルで実行する。Codex側ではコード作成・レビュー・小規模デバッグを行う。

問題時は、実行コマンド全文、ログ全文、`git status -sb`、`which python`、`python --version`、関連summary/index/failures、期待結果と実結果を共有する。

## 16. 現時点で次に実行するもの

881件manifestは完成済みなので、次は評価用抽出の1～5件確認である。

```bash
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
source /home/wakamatsu/ITS/.venv_waymo/bin/activate

bash scripts/45_export_shimizu_frames_64_32.sh --dry-run --max-frames 1
bash scripts/45_export_shimizu_frames_64_32.sh --max-frames 1 --resume
bash scripts/45_export_shimizu_frames_64_32.sh --max-frames 5 --resume
```

5件のshape、identity、偶数ring、NaN/Inf、intensity統計が正常なら、Step 5の学習用segment分離へ進む。
