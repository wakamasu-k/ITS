# Waymo 16→32 点群評価

1フレームだけの点群復元確認を、複数validation/testフレームへ拡張するための評価ツールを追加した。

評価対象は生成odd rows `1,3,...,31`で、GT-validかつTULIP rangeが有限・正の画素を使う。
点群はWaymo SDKのbeam inclination、TOP pixel pose、extrinsic、vehicle poseからvehicle frameへ変換する。

出力:

- `sample_XXXXXX/waymo_16_to_32_pointcloud.npz`
- `sample_XXXXXX/metadata.json`
- `per_frame.csv`
- `summary.json`

指標:

- coordinate MAE [m]
- Euclidean point MAE [m]
- raw intensity MAE

同一range-image画素を対応付けるpaired metricであり、Chamfer距離ではない。
まず複数フレームでこの評価を実行し、その後に必要なら論文条件に近い点群指標を追加する。

---

作成日（ファイル更新日時から推定）: 2026-08-20
