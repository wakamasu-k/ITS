# 学習用64/32フレーム抽出

outputs/training_segment_manifest.csvからroleごとにsegmentを選び、Waymo TOP LiDARの
64ラインrange imageと、その偶数ringから作る32ライン入力を抽出する。

既定の小規模設定は次のとおり。

- train: 8 segment
- validation: 2 segment
- test: 2 segment
- 各segment: 20 frames
- frame index: 0, 10, 20, ..., 190

同一TFRecordは1回だけ順方向に走査する。出力は
outputs/training_frames_64_32/{role}/{segment}/{frame_index}/以下へ保存し、
index.csvにrole、元split、segment、frame index、timestamp、元TFRecordを記録する。

## 抽出なしの確認

```bash
source /home/wakamatsu/ITS/.venv_waymo/bin/activate
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
bash scripts/65_export_training_frames_64_32.sh --dry-run
```

dry-runはTFRecordを開かず、選択予定だけを表示する。

## 最小実抽出

```bash
bash scripts/65_export_training_frames_64_32.sh \
  --train-segments 1 \
  --validation-segments 0 \
  --test-segments 0 \
  --frames-per-segment 2 \
  --frame-stride 10 \
  --resume
```

成功後に既定の240フレームへ広げる。

```bash
bash scripts/65_export_training_frames_64_32.sh --resume
```

中断後も--resumeで再開できる。完成判定ではNPZとJSONの存在だけでなく、role、
segment、frame index、元splitを照合する。評価用segmentとの重複は抽出直前にも再検査し、
見つかった場合は処理を開始せず失敗する。
