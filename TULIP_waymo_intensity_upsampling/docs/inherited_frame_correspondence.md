# 引き継ぎ研究データのフレーム対応

入力 `manifest_cross_eval_gt_v3.csv` はクエリ画像と複数の地図anchorの組です。
同じクエリフレームがanchorごとに繰り返されるため、
`(q_subset, q_seg, q_frame_index)` で重複除去します。

## 対応関係

- `q_subset`: 元Waymo split
- `q_seg`: Waymoのsegment/context名
- `q_frame_index`: TFRecord内の0始まりフレーム位置
- `q_meta_json`: クエリカメラJSON
- `frame_timestamp_micros`: JSONに保存されたWaymo timestamp
- `source_tfrecord`: `WAYMO_ROOT/q_subset/q_seg.tfrecord`
- `source_manifest_rows`: 同じフレームへ集約したGT pair数

古い `/mnt/d/waymo/data/cam_gray/...` は、`--line32-root /mnt/w/32line`
によって `/mnt/w/32line/datasets/cam_gray/...` へ解決します。

## 実行

```bash
source /home/wakamatsu/ITS/.venv_tulip/bin/activate
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
bash scripts/35_build_shimizu_frame_manifest.sh
```

出力は `outputs/shimizu_frame_manifest.csv` です。カメラJSONがない、
JSONのframe indexがmanifestと異なる、または同一キーでtimestampが競合する場合は
処理を停止します。元TFRecordがない場合も既定で停止します。
調査目的で欠損行を残す場合のみ `--allow-missing-tfrecord` を指定します。

この処理はTFRecord本文を読みません。存在確認はファイルパスに対してのみ行います。

## 前回追加したmanifestとの違い

`handover_frame_manifest.py` は
`loftr_pairs_shifted/training/manifest_matches_step2a_paper_final.csv` を対象とし、
`cam_json` の `(segment_id, frame_index)` を集約する汎用・学習pair向け処理です。

今回の `shimizu_frame_manifest.py` は評価GT
`loftr_gt/manifest_cross_eval_gt_v3.csv` 専用で、manifest自身が持つ
`q_subset/q_seg/q_frame_index` を基準にします。TULIPへ引き継ぐフレーム対応表としては
今回のCSVが直接的です。
