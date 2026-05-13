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

