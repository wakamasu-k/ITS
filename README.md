# ITS

コード説明
export_cam_gray_txt.py
Waymoの.tfrecordからカメラ画像を取り出し、グレースケール画像と姿勢内部パラメータなどのメタ情報を保存するコード

引数を受け、入力が.tfrecordかテキストファイルなのかを確認し、
書く.tfrecordを開く
waymono Frameを全フレーム読み込む
指定カメラ画像を探す
画像をopencvでデコード
グレースケール化
必要なら、CLAHE
カメラ内部パラメータ・姿勢行列を取得
PNG画像とJSONメタ情報を保存

これはのちのretrievalやLoFTRのカメラ画像側の入力になる

CLAKE　局所的なコントラスト補正
カメラ画像と反射強度画像を見比べやすくするため、明暗差を強調

_process_one_tfrecord()

このコードの本体です。

やっていることは以下です。

1. tfrecord が存在するか確認
2. セグメント名を取得
3. カメラ名を Waymo のカメラIDに変換
4. 出力フォルダを作成
5. TFRecordDataset で全フレーム読み込み
6. 各フレームから対象カメラ画像を取得
7. グレースケール化
8. カメラキャリブレーション取得
9. 内部パラメータ fx, fy, cx, cy を取得
10. vehicle pose と world_to_cam を取得
11. png 保存
12. json 保存

画像保存とJSON保存の部分では、png_path と json_path を作り、既存ファイルがある場合は --overwrite false ならスキップします。

make_static_map_txt_.py
Waymoの.tfrecordかたTOPLidar点群を取り出して、道的物体の自車ルーフ周辺を除去して、静的な三次元点群地図を作成

目的
Waymo .tfrecord
↓
TOP LiDAR の Range Image を読む
↓
Range Image を3次元点群へ変換
↓
64ライン LiDAR を32ライン相当に間引く
↓
動的物体の点を除去
↓
自車ルーフ周辺の点を除去
↓
各フレームの点群を world 座標へ変換
↓
全フレームを重畳
↓
map_static.npz を保存

車両座標系の点群をworldっ座標系へ変換している
これは20秒くらいの走行セグメント内で共通に使える座標系に変換した？
⇒真衣フレームで車の位置や、向きが変わり、そのまま点群を足すと、座標がバラバラになる
そこで、各フレームの車両姿勢を使って、点群を同じWorld座標へ変換する
その後、時系列の全点群を重ねることで、一つの三次元点群地図を作成

また、道的物体除去と32ライン化も行っている
歩行者、など、BBoxない点を消し、64ラインから32ライン総統に間引いている

実行コマンドで入力しているもの

たとえばこのコマンドです。

python make_static_map_txt.py \
  --tfrecord "$TFREC" \
  --out-root "$HOME/ITS/waymo_outputs/static_map_test" \
  --subset training \
  --lidar-lines 32 \
  --ring-keep-even true \
  --bbox-vehicle-y-scale 2.5 \
  --overwrite true


make_submaps_distance_txt.py
Waymoの.tfrecordから各フレームの車両一を読み取り、走行軌跡上に約１m感覚でアンカーを配置市、
各フレームがどのアンカー・サブマップに属するかをCSV,JSONで保存

.tfrecord
↓
各フレームの車両位置・yaw角を取得
↓
走行軌跡に沿ってアンカーを作成
↓
各フレームを最も近いアンカーに割り当て
↓
submaps.json / anchors.csv などを保存

[INFO] 出力先:  /home/wakamatsu/ITS/waymo_outputs/submaps_test/training/segment-10017090168044687777_6380_000_6400_000_with_camera_labels
[INFO] anchor_mode=greedy, spacing=1.0 m, R_sub=10.0 m, yaw_jump=0.0 deg
2026-05-14 00:36:29.212223: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:982] could not open file to read NUMA node: /sys/bus/pci/devices/0000:08:00.0/numa_node

[OK] submaps.json を保存: /home/wakamatsu/ITS/waymo_outputs/submaps_test/training/segment-10017090168044687777_6380_000_6400_000_with_camera_labels
     サブマップ数: 88  | フレーム数: 198
     anchor_mode=greedy
     アンカー間隔(軌跡距離) min/med/max = 1.001 / 1.275 / 1.484 [m]
     最初のアンカー frame_index: 0, 位置(x,y)=(-1257.18,10546.04)
     anchors.csv を保存: /home/wakamatsu/ITS/waymo_outputs/submaps_test/training/segment-10017090168044687777_6380_000_6400_000_with_camera_labels/anchors.csv