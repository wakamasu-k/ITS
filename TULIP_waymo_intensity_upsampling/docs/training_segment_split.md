# TULIP学習用segment分割

清水研究の自己位置推定評価881フレームと同じWaymo segmentを、TULIPの学習・検証・
テストから除外する。フレーム単位ではなくsegment単位で分割し、同一走行系列からの
情報漏洩を防ぐ。

この段階ではTFRecordの中身を読み込まない。*.tfrecordのファイル名だけを列挙するため、
range imageの展開やGPU処理は発生しない。

## dry-run

```bash
source /home/wakamatsu/ITS/.venv_tulip/bin/activate
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
bash scripts/60_build_training_segment_manifest.sh --dry-run
```

小規模な分割確認には次を使う。

```bash
bash scripts/60_build_training_segment_manifest.sh \
  --dry-run \
  --max-segments 12
```

## CSV生成

```bash
bash scripts/60_build_training_segment_manifest.sh
```

outputs/training_segment_manifest.csvへdataset_role、source_split、
segment_id、source_tfrecordを出力する。既定seedは20260730、比率は
train/validation/test = 80/10/10であり、入力順に依存しないSHA-256順で固定する。

このCSVはsegment選定表であり、まだ32→64フレームペアを生成しない。次段階で各segmentの
フレームを間引いて抽出し、学習用indexへ接続する。
