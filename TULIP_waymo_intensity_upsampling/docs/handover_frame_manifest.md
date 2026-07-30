# 引き継ぎmanifestのフレーム単位変換

ペア単位manifestは同じカメラフレームの複数の `k` を1行ずつ保持します。
本ツールは `(segment_id, frame_index)` で集約し、1フレーム1行にします。

出力列は `segment_id`、`frame_index`、`timestamp_us`、`source_split`、
`source_tfrecord`、`source_manifest_rows`、`metadata_json`、`status` です。
manifestとJSONのsegment不一致、同一フレームのtimestamp競合、元TFRecord重複は
エラーになります。`--line32-root` は古いパスの `/datasets/` 以下を現在のrootへ
付け替え、任意のprefixは `--path-map OLD=NEW` で置換できます。

## 軽量変換

TFRecordのファイル名だけを走査し、本文は読みません。

```bash
source /home/wakamatsu/ITS/.venv_tulip/bin/activate
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
bash scripts/40_build_handover_frame_manifest.sh
```

## 元TFRecordとの厳密照合

Waymo環境で参照対象TFRecordを読み、segment、frame index、timestampを照合します。
I/Oが重いため明示的に実行してください。

```bash
source /home/wakamatsu/ITS/.venv_waymo/bin/activate
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
bash scripts/40_build_handover_frame_manifest.sh --verify-tfrecord
```

元TFRecordがない行は既定でエラーです。調査時のみ
`--allow-missing-tfrecord` で `status=missing_tfrecord` として残せます。

## 軽量テスト

```bash
source /home/wakamatsu/ITS/.venv_tulip/bin/activate
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
python -m unittest tests.test_handover_frame_manifest -v
```
