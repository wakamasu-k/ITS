# 清水研究のWaymo前処理仕様

## 論文と実データから確認した内容

- Waymo Open Dataset Perception v1.4.3 (with maps)を使用
- 車両には5カメラ、1台の64ビームTOP LiDAR、4台の短距離LiDARを搭載
- TOP LiDARのfirst returnを主対象とする
- レンジ画像にはrange、intensity、elongation等のチャンネルがある
- 64ラインから走査線を1本おきに除き、32ライン相当の疎な点群を生成
- 各フレームの点群を車両姿勢でglobal座標へ変換して時系列重畳
- Waymoの3D BBoxを用いて動的物体由来の点を除去
- 車体近傍点も除去
- 反射強度画像はカメラ平面への投影時に最近点を採用
- 欠損緩和のため近傍画素へ描画
- 外れ値を抑えるためパーセンタイル正規化
- ヒストグラム平坦化とCLAHEを適用

## 実装メタデータで確認した内容

代表scene:

`segment-1005081002024129653_5313_150_5333_150_with_camera_labels`

- frames: 199
- downsample_32l: true
- ring_keep_even: true
- ring: 0,2,...,62
- map points: 14,127,289
- output: `map_static.npz`
- arrays: `xyz`, `intensity`
- tool metadata: `make_static_map.py`, version `v1.1`
- intensity upper percentile: rendering code default 99.5

## TULIP研究で変更しない物理データ

以下は学習・数値評価前に加工しません。

- range
- intensity
- valid mask
- beam/ring correspondence
- frame timestamp
- vehicle pose
- TOP LiDAR calibration

ヒストグラム平坦化やCLAHEは、後段のカメラ照合用画像にのみ適用します。
TULIPの反射強度誤差をCLAHE後の8bit画像だけで評価してはいけません。

## 検証時に必ず保存する識別情報

- source TFRecordの絶対パス
- Waymo context/segment名
- frame index
- timestamp_micros
- lidar name
- return index
- native shape
- kept ring IDs
- range/intensityの有効値統計
- preprocessing version
