# TULIP KITTI再現とWaymo 2ch実装の比較

調査日: 2026-08-18

## 結論

現在のWaymo実装は、KITTIのTULIP本体を維持しながら、Dataset、幅padding、最終upsampling head、mask処理を変更した構成である。

重要な点として、現在の正式Waymo adapterは **32ライン入力から64ラインを復元する構成** であり、最終目標の16ライン入力から32ライン復元はまだ正式実装されていない。

## 1. KITTIの実際の設定

参照:

- `/home/wakamatsu/ITS/TULIP_backup/bash_scripts/tulip_upsampling_kitti.sh`
- `/home/wakamatsu/ITS/TULIP_backup/bash_scripts/tulip_evaluation_kitti.sh`

|項目|KITTI|
|---|---|
|入力|`[B,2,16,1024]`|
|教師|`[B,2,64,1024]`|
|`in_chans`|2|
|`patch_size`|`(1,4)`|
|`window_size`|`(2,8)`|
|`pixel_shuffle`|True|
|`patch_unmerging`|True|
|`circular_padding`|True|
|`log_transform`|True|
|model|`tulip_base`|
|embed_dim|96|
|depths|`(2,2,2,2)`|
|num_heads|`(3,6,12,24)`|
|batch size|8|
|epoch数|30|
|learning rate|`5e-4`|

`tulip_base`のTransformer設定は、Waymoでも同じものを使っている。

## 2. KITTI Datasetの処理

参照: `/home/wakamatsu/ITS/TULIP_backup/tulip/util/datasets.py`

- `.npy`をfloat32で読み込む。
- channel 0をrange、channel 1をintensityとして扱う。
- `ScaleChannelZero(1/80)`でrangeだけを`1/80`にする。
- 64ライン高解像度から4行間隔で16ライン入力を作る。
- `log_transform=True`の場合、rangeとintensityの両方へ`log1p`を適用する。
- KITTI DatasetはWaymo Datasetのような明示的boolean valid maskをモデルへ返さない。

主な箇所:

- `npy_loader`: `datasets.py:175-179`
- `DownsampleTensor`: `datasets.py:117-125`
- `ScaleChannelZero`: `datasets.py:280-288`
- `build_kitti_upsampling_dataset`: `datasets.py:289-318`

## 3. KITTIのrange正規化とmaximum_range

KITTIでは次の関係になっている。

|処理|式|
|---|---|
|Dataset|`normalized_range = raw_range / 80`|
|点群復元|`range_meter = normalized_range * maximum_range`|

`img_to_pcd_kitti()`は、評価画像が0〜1正規化済みであることを前提に`maximum_range`を掛ける。

評価処理では`maximum_range=80`が使用されている。

Waymoでは、KITTIの80mをまだ固定していない。現在は学習スクリプトの`range_scale`を実験条件として渡し、intensityの数値scaleは固定していない。

## 4. KITTIとWaymoで同じ部分

- TULIPのTransformer本体
- PatchEmbedding
- Encoder / Decoder
- PatchMerging
- Attention / Transformer Block
- Skip connection
- `embed_dim=96`
- `depths=(2,2,2,2)`
- `num_heads=(3,6,12,24)`
- `in_chans=2`
- channel 0=range、channel 1=intensityという意味
- `decoder_pred`による2ch出力
- 基本的なL1損失の考え方

参照: `/home/wakamatsu/ITS/TULIP_backup/tulip/model/tulip.py`

- コンストラクタ: `tulip.py:531-535`
- PatchEmbedding: `tulip.py:571-572`
- decoder_pred: `tulip.py:574`
- 最終headの選択: `tulip.py:576-582`
- forward: `tulip.py:702-737`

これらのTransformer本体はWaymo対応のために変更していない。

## 5. Waymoで変更した部分

### 入出力の高さ

KITTI:

- `16 → 64`
- 入力`[B,2,16,1024]`
- 教師`[B,2,64,1024]`

現在のWaymo 2ch:

- `32 → 64`
- 入力`[B,2,32,2650]`
- 教師`[B,2,64,2650]`

### Dataset形式

参照: `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/tulip_adapter/waymo_32_to_64_dataset.py`

Waymoはpair NPZとindex CSVを使う。

- `input_32`: `[32,W,2]`
- `target_64`: `[64,W,2]`
- `input_mask`: `[32,W]`
- `target_mask`: `[64,W]`
- Dataset出力: input`[2,32,W]`、target`[2,64,W]`

intensityは`raw`または`log1p`を選択できる。rangeとintensityのscaleは同じにしていない。

### 水平方向の幅

KITTIの幅は1024である。

Waymoのnative widthは2650で、TULIP backboneへ入れる前に2816へpaddingする。

- 入力: `2650 → 2816`、右側へcircular padding
- 出力: `2816 → 2650`、native widthへcrop

### 最終upsampling head

KITTI:

- `pixel_shuffle=True`
- `patch_unmerging=True`

Waymo現在:

- `pixel_shuffle=False`
- 既存Transformerを通過した後の最終headだけをadapterで置換
- vertical scale=2
- horizontal scale=4

このheadにより、patch size `(1,4)`で縮小された幅を戻しつつ、高さを32から64へ2倍にする。

参照: `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/tulip_adapter/waymo_32_to_64_tulip.py:68-149`

### maskとloss

KITTIの元実装は、基本的に全要素のL1平均を使用する。

Waymoでは無効画素を除外するため、target maskを使う。

- rangeとintensityに同じ空間valid maskを適用
- valid pixelだけでlossを計算
- 2channelの平均L1を計算

## 6. 変更点の一覧

|項目|KITTI|Waymo現在|
|---|---|---|
|入力高さ|16|32|
|出力高さ|64|64|
|幅|1024|2650、内部2816|
|patch size|`(1,4)`|`(1,4)`|
|window size|`(2,8)`|`(2,8)`|
|in_chans|2|2|
|PixelShuffle|使用|不使用|
|PatchUnmerging|使用|不使用|
|range scale|`1/80`|`range_scale`を実験指定|
|intensity|log1p|raw/log1p選択|
|valid mask|明示的でない|使用|
|最終head|KITTI既存head|Waymo adapter head|
|checkpoint|KITTI checkpoint|Waymo用に新規学習|

## 7. checkpoint互換性

`in_chans=2`と`decoder_pred`の2ch出力は共通している。

しかし、以下が異なるため、KITTIの2ch checkpointをWaymoへそのまま使用する方針ではない。

- KITTIはPixelShuffle head、Waymoは異方的最終head
- 入出力サイズが異なる
- KITTIはrangeを80で正規化、Waymoは実験指定
- mask処理が異なる
- KITTIは16→64、Waymo現在は32→64

Waymo用2ch checkpointは新規学習する。Transformer部分の重み転用は、strict loadと性能確認なしには決定しない。

## 8. 現在の実装・検証状況

完了:

- Waymo 2ch Dataset
- 2ch正式adapter
- 実データforward
- 2ch学習スクリプト
- 2ch評価スクリプト
- range/intensity別評価
- 1サンプル1epochの学習接続確認
- 単体テスト26件PASS

未完了:

- 複数epochの本格学習
- validationでの性能評価
- rawとlog1pの比較
- 学習済みcheckpointの性能判定
- 16→32専用Dataset
- 16→32専用adapter

## 9. 次の段階

1. raw intensityで複数epoch学習する。
2. validationでrange/intensity別MAEを測定する。
3. nearestおよびlinear補間と比較する。
4. log1pでも同じ実験を行う。
5. 外れ値とlossバランスを確認する。
6. その後、16→32専用構成を追加する。

この順序なら、KITTI、既存range-only、TULIP Transformer本体を維持したまま、Waymo 2chの有効性を確認できる。

## 10. 参照ファイル

KITTI:

- `/home/wakamatsu/ITS/TULIP_backup/bash_scripts/tulip_upsampling_kitti.sh`
- `/home/wakamatsu/ITS/TULIP_backup/bash_scripts/tulip_evaluation_kitti.sh`
- `/home/wakamatsu/ITS/TULIP_backup/tulip/main_lidar_upsampling.py`
- `/home/wakamatsu/ITS/TULIP_backup/tulip/util/datasets.py`
- `/home/wakamatsu/ITS/TULIP_backup/tulip/model/tulip.py`
- `/home/wakamatsu/ITS/TULIP_backup/tulip/util/evaluation.py`

Waymo:

- `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/tulip_adapter/waymo_32_to_64_dataset.py`
- `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/tulip_adapter/waymo_32_to_64_tulip.py`
- `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/tulip_adapter/waymo_2ch_training.py`
- `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/tools/train_waymo_2ch_tulip.py`
- `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/tools/evaluate_waymo_2ch_tulip.py`

---

作成日（ファイル更新日時から推定）: 2026-08-18
