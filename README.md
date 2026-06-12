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


render_anochor_intensity_txt.py
アンカーごとの反射強度画像を作成するコード

今までの流れ
make_static_map_txt.py
→map_static.npzを作成

make_submap_distance_txt.py
→submaps.json/anchors.csvを作成

render_anchor_intensity.py
→map_static.npz+submaps.jsonを使ってanchor.pngを作成

目的
静的点群地図　map_static.npz
+
アンカー情報　submaps.json
+
Waymoのかメタ内部、外部パラメータ
→各アンカー視点に点群を投影死
反射強度画像anchor.pngを作成

処理の中身
1. .tfrecordを読む
2. map_static.npzを読む
3. submaps.jsonを読む
4. 各アンカーの位置を取り出す
5. アンカー周辺の点群だけをcropする
6. アンカーフレームのカメラ姿勢・内部パラメータを取得する
7. world座標の点群をカメラ画像平面へ投影する
8. z-bufferで手前の点を優先する
9. intensityを0〜255へ正規化する
10. ヒストグラム平坦化とCLAHEを適用する
11. anchor.png / anchor.json / anchor_index.npz を保存する

特に重要なのは z-buffer です。
同じ画素に複数の3D点が投影された場合，手前の点を優先します。これにより，奥の点が手前の物体を突き抜けて描画されるのを防いでいます

実行コード
python render_anchor_intensity_txt.py \
  --tfrecord "$TFREC" \
  --subset training \
  --maps-root "$HOME/ITS/waymo_outputs/static_map_test" \
  --submaps-root "$HOME/ITS/waymo_outputs/submaps_test" \
  --cam FRONT \
  --int-mode global_map \
  --crop-mode circle \
  --crop-radius-m auto \
  --crop-margin-m 2.0 \
  --point-size 2 \
  --write-index true \
  --post-hist-eq true \
  --post-clahe true \
  --overwrite true

  実行結果
  [INFO] segment=segment-10017090168044687777_6380_000_6400_000_with_camera_labels | frames=198 | pts=12965659 | submaps=88 | cam=FRONT
[INFO] crop=circle, r=auto, margin=2.0, z=(None,None), mode=global_map
[INFO] post: hist_eq=True, clahe=True, mask_covered_only=True
[INFO] overwrite=True (サブマップ単位で anchor 出力の再計算を制御)
render anchors: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 88/88 [00:37<00:00,  2.32submap/s]
[OK] anchors rendered (resume-aware): 88 -> /home/wakamatsu/ITS/waymo_outputs/submaps_test/training/segment-10017090168044687777_6380_000_6400_000_with_camera_labels

1. カメラ画像
   export_cam_gray_txt.py
   → 198枚の *_cam.png / *_cam.json

2. 静的点群地図
   make_static_map_txt.py
   → map_static.npz

3. アンカー情報
   make_submaps_distance_txt.py
   → submaps.json / anchors.csv

4. アンカー反射強度画像
   render_anchor_intensity.py
   → 各 submap_id の anchor.png / anchor.json / anchor_index.npz

make_retrieval_shifted_ri_txt.py
疎探索Retrieval学習用のデータセットを作るコード

これまでに作ったカメラ画像を使って、カメラ画像一枚に対して、一を少しずらして複数の反射強度画像を生成するコード

目的
カメラ画像
+
アンカー位置の反射強度画像
+
少し位置・姿勢をずらした反射強度画像
↓
Retrieval粗探索用の学習ペアを作る

粗探索では
カメラ画像に対して、どのRI画像がどのRIがぞうが同じ場所化を探す
必要がある

これはretrieval学習用に
Δs：前後方向 [-0.45, +0.45] m
Δl：左右方向 [-1.0, +1.0] m
Δψ：yaw方向 [-2°, +2°]
この範囲でずらしている

これによって一つのアンカーからみたRIだけでなく、
少し前にずらしたRI
少し後ろにずらしたRI
少し左にずらしたRI
少し右にずらしたRI
少し回転させたRI
を作る

Retrievalの学習時に「同じ場所の近傍変化」に強くする目的があります。

皆さん，お疲れさまです！

こちらのグループは，ドット生活内で困ったこと，わからないこと，心配なことを気軽に相談してもらうためのグループです。

このグループでは，事務局に関することだけでなく，集客，PG，クライアント対応，支部運営，選考会関連など，支部活動の中で出てくるさまざまな相談をしてもらえたらと思っています。

今後，実績相談などの専用グループも作成される予定ですが，まずは「どこに相談すればいいかわからない」「早めに確認したい」「他の人にも共有したほうがよさそう」と思うことがあれば，このグループで気軽に聞いてください！

また，このグループには，代表，事務局，クライアント，PG，選考会責任者など，支部の中で「この人にも入ってもらったほうがよい」「この人に相談を見てもらいたい」と思う方がいれば，ぜひ招待をお願いします。

ただし，相談内容については，グループで相談するものと個別チャットで相談するものの基準を設けたいと思います。

【グループで相談してほしい内容】
・実績，SF関連の相談
・早急に対応が必要なこと
・領収書や精算書など，お金に関わること
・他の支部スタッフやユニットにも共有したほうがよい内容

【個別チャットで相談してほしい内容】
・ユニットの叩きに関する質問
・個人のタスクに関する質問
・個別で確認したほうがよい細かい相談

このように分ける理由は，グループで相談することで，他の支部スタッフやユニットにも内容を共有でき，必要な人が早く対応できるようにするためです。

皆さんが安心して活動できるように，このグループをうまく活用していきたいと思っています。
困ったことや不安なことがあれば，遠慮せずに相談してください！

よろしくお願いします！

cd ~/ITS/handover_shimizu/src/data_make/train
source ~/ITS/.venv_waymo/bin/activate

TFREC="/mnt/waymo_net/waymo_tf/perception_v1.4.3/individual_files/training/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"

python make_retrieval_shifted_ri_txt.py \
  --tfrecord "$TFREC" \
  --subset training \
  --maps-root "$HOME/ITS/waymo_outputs/static_map_test" \
  --submaps-root "$HOME/ITS/waymo_outputs/submaps_test" \
  --cam-gray-root "$HOME/ITS/waymo_outputs/cam_gray_test" \
  --out-root "$HOME/ITS/waymo_outputs/retrieval_pairs_test" \
  --cam FRONT \
  --num-samples-per-anchor 16 \
  --zero-ri-mode reuse \
  --overwrite true


RETRIEVALフォルダ
make_retrieval_pairs_parquet.py

retrieval_pairs_test の中に作った cam.json と ri_*.png を走査して，学習コードが読みやすい pairs.parquet を作るコードです。
学習本体ではなく、学習用リスト、マニフェスト作成の段階
pairs.parquetを作成するコード

このコードは書くsubmap_id＞フォルダを見て、
cam.json から camera画像パスを取得
ri_*.png からRI画像を1枚選ぶ
↓
1行のペアとして保存
している

出力されるのは
cam_png
anchor_png
label
segment_id
submap_id
anchor_frame_index
anchor_choice

です。

重要なのは、一つのカメラ画像につき基本一行だけ作ると言う点
これはInfoNCEでは同じcam_pngを副ルウ業に増やすと、同一パッチ内で重複が起きて学習が破綻しやすくなるため、
実行コマンド
python make_retrieval_pairs_parquet.py \
  --pairs-root "$HOME/ITS/waymo_outputs/retrieval_pairs_test" \
  --subset training \
  --out-parquet "$HOME/ITS/waymo_outputs/retrieval_manifests_test/training/pairs.parquet" \
  --out-summary-json "$HOME/ITS/waymo_outputs/retrieval_manifests_test/training/summary.json" \
  --anchor-select random \
  --seed 42 \
  --min-ri 1 \
  --require-cam-exists true \
  --require-ri000 false \
  --drop-dup-cam true

compute_modality_stats_from_pairs.py
粗探索モデルの学習で使う正規化用の平均値・標準偏差を計算するコード

pairs.parquet に書かれている cam_png と anchor_png を読み，カメラ画像側とRI画像側それぞれの mean/std を計算して，modality_stats.json として保存します


学習時には画像をそのままモデルに入れるのではなく，

画像値を 0〜1 に正規化
↓
平均 mean を引く
↓
標準偏差 std で割る

という正規化を行うことが多いです。

このコードは，そのために必要な値を計算します。

カメラ画像の mean/std
RI画像の mean/std
を別々に求めます。
出力される JSON には，例えば以下のような値が入ります。
{  "cam_mean": 0.45,  "cam_std": 0.21,  "anc_mean": 0.18,  "anc_std": 0.25}
ここで，
cam = camera imageanc = anchor / RI image
です。

統計計算
python compute_modality_stats_from_pairs.py \
  --pairs "$HOME/ITS/waymo_outputs/retrieval_manifests_test/training/pairs.parquet" \
  --out-json "$HOME/ITS/waymo_outputs/retrieval_manifests_test/training/modality_stats.json" \
  --only-label1 \
  --anchor-mode anchor_png

  compute_modality_stats_from_pairs.py は，pairs.parquet の cam_png と anchor_png を読み，画像をグレースケール 0〜1 に正規化して，カメラ側とRI側の平均・標準偏差を計算するコードです
今回の値の見方
カメラ画像側
"cam_mean": 0.46063928193643705,
"cam_std": 0.1845111912725241

カメラ画像は CLAHE 済みのグレースケールなので，平均が 0.46，標準偏差が 0.18 というのはかなり自然です。

つまり，画像全体としては中間より少し暗めで，明暗差もある程度あります。

RI画像
"anc_mean": 0.16721673386336358,
"anc_std": 0.27375476378681163
RI画像の平均がカメラ画像より低いのも自然です。

理由は，反射強度画像は背景が黒，つまり 0 の画素が多くなりやすいからです。
その一方で，LiDAR点が投影された画素はCLAHEなどで強く明るくなるため，標準偏差は 0.27 と大きめになります。

つまり，
RI画像は黒背景が多い
でも点がある場所は明るい
→ 平均は低いが，ばらつきは大きい
という状態です。

これはRI画像としてはかなりあり得る分布です

ピクセル数も整合しています
"cam_pixels": 216268800,
"anc_pixels": 216268800

88枚で割ると，

216268800 / 88 = 2457600

です。

これは，

1920 × 1280 = 2457600

なので，カメラ画像とRI画像が同じ解像度で読まれている可能性が高いです。

これはとても良いです。
カメラ画像とRI画像のサイズが揃っているので，後続の学習でも扱いやすい


train.py
粗探索モデルの学習本体
pairs.parquet
modality_stats.json
を入力して、カメラ画像と反射強度画像RIを同じ特徴空間に近づける学習をおこなう

1. この train.py がやること

このコードは，

camera image encoder
+
RI / anchor image encoder
+
NetVLAD
+
InfoNCE loss

で粗探索モデルを学習します。

簡単にいうと，

カメラ画像を入力
↓
カメラ用エンコーダ
↓
特徴ベクトル z_cam

RI画像を入力
↓
RI用エンコーダ
↓
特徴ベクトル z_ri

z_cam と z_ri が近くなるように学習

です。

実行コード（エポック数を1にしている）

python train.py \
  --pairs "$HOME/ITS/waymo_outputs/retrieval_manifests_test/training/pairs.parquet" \
  --modality-stats "$HOME/ITS/waymo_outputs/retrieval_manifests_test/training/modality_stats.json" \
  --out-dir "$HOME/ITS/waymo_outputs/retrieval_train_test/one_segment_smoke" \
  --epochs 1 \
  --batch-size 16 \
  --backbone resnet18 \
  --clusters 8 \
  --embed-dim 128 \
  --image-h 192 \
  --image-w 320 \
  --device cuda \
  --num-workers 0 \
  --scheduler none \
  --best-by loss \
  --train-ri-mode epoch_cycle \
  --train-ri-count 16 \
  --save-every-epochs 1 \
  --no-tb

  ログ
  cat "$OUT/logs/train_metrics.jsonl"
epoch,loss,lr
1,2.2385653257369995,0.0003
{"epoch": 1, "loss": 2.2385653257369995, "lr": 0.0003}

epoch,loss,lr
1,2.238445258140564,0.0003
2,1.7595301151275635,0.0003
3,1.4265324592590332,0.0003
4,0.7792956590652466,0.0003
5,0.6645824909210205,0.0003


evaluate.py
学習済みの粗探索モデルを使って、カメラ画像から正しいRIアンカー画像を検索できるかを評価するコード
pairs.parquet
modality_stats.json
train.py
  ↓
best.ckpt / last.ckpt
  ↓
evaluate.py
  ↓
R@1, R@5, R@10, MRR, MedianRank などを計算

このコードでやること
1. checkpoint を読み込む
2. pairs ファイルから query カメラ画像と正解RIを読む
3. db ファイルから検索対象のRIアンカー一覧を読む
4. カメラ画像を cam encoder で特徴ベクトル化
5. RI画像を int encoder で特徴ベクトル化
6. 類似度行列を計算
7. 正解RIが何位に来るかを調べる
8. Recall@K / MRR / MedianRank を出力する
Strict指標、Soft指標、per-queryCSV,topKJSONL,failure dumpなどを出力する拡張評価器

Strict評価と、Soft評価の違い
Strict評価はこのカメラ画像の正解RIはこの一枚として評価する
正解RIが検索結果の一位ならR@1二成功、5位以内ならR@5に成功
見る指標は、
R@1
R@5
R@10
R@30
R@50
MRR
MedianRank

Soft指標は
正解RIそのものではなくても、一的に近いRIなら正解扱いする
評価
Softhyouka niha ICPでDB側セグメントとQuery側セグメントを位置合わせしたCSVが必要になる
evaluate.pyでは要求が難しく
pairs 側:
cam_png
anchor_png または gt_anchor_png
pair_id

db 側:
anchor_png
pair_id
anchor_x_db
anchor_y_db
anchor_z_dbが必要になる
そのため、評価に簡単なpairs/db manifestを作ってから、実行する必要がある

実行コード
cd ~/ITS/handover_shimizu/src/retrieval
source ~/ITS/.venv_waymo/bin/activate

CKPT="$HOME/ITS/waymo_outputs/retrieval_train_test/one_segment_overfit/checkpoints/best.ckpt"

python evaluate.py \
  --checkpoint "$CKPT" \
  --pairs "$HOME/ITS/waymo_outputs/retrieval_eval_manifests_test/training/eval_pairs.parquet" \
  --db "$HOME/ITS/waymo_outputs/retrieval_eval_manifests_test/training/eval_db.parquet" \
  --out_dir "$HOME/ITS/waymo_outputs/retrieval_eval_test/one_segment_overfit" \
  --name one_segment_trainset \
  --backbone resnet18 \
  --clusters 8 \
  --embed_dim 128 \
  --freeze_stages 2 \
  --image_h 192 \
  --image_w 320 \
  --batch_size 32 \
  --num_workers 0 \
  --device cuda \
  --amp false \
  --modality_stats "$HOME/ITS/waymo_outputs/retrieval_manifests_test/training/modality_stats.json" \
  --strict_ks "1,5,10,30,50" \
  --write_failures true \
  --write_topk_jsonl true \
  --topk_dump_k 20

  今回のエラー整理
  エラー名：
_pickle.UnpicklingError: Weights only load failed

発生場所：
evaluate.py の _load_model()
ckpt = torch.load(args.checkpoint, map_location="cpu")

原因：
PyTorchの新しい挙動で，torch.load が checkpoint 全体ではなく安全な重みだけを読もうとした。
しかし train.py が保存した checkpoint には numpy のRNG状態なども含まれているため失敗した。

修正：
自分で作成した信頼できる checkpoint なので，
torch.load(..., weights_only=False)
に変更する。

88枚のカメラ画像をクエリとして，88枚のRIアンカー画像の中から正解を探す評価
出た結果
"R@": {
  "1": 0.045454545454545456,
  "5": 0.18181818181818182,
  "10": 0.32954545454545453,
  "30": 0.5113636363636364,
  "50": 0.75
},
"MRR": 0.12379925971100816,
"MedianRank": 29.5

R@1  = 4 / 88 件くらい正解
R@5  = 16 / 88 件くらい正解
R@10 = 29 / 88 件くらい正解
R@30 = 45 / 88 件くらい正解
R@50 = 66 / 88 件くらい正解

ただまだまだ結果としてはよわい
今回は1セグメント88ペアのみ
また、今回はSoftはカラになっている
Soft評価は，ICPでDB側とQuery側を位置合わせした icp_csv が必要です。今回は --icp_csv を渡していないので，Strict評価だけが出ています。evaluate.py はStrict指標と，ICPがある場合のSoft指標を同時に出す設計です。

一覧まとめ
種類	場所
元 .tfrecord	/mnt/waymo_net/waymo_tf/perception_v1.4.3/individual_files/training/<SEG>.tfrecord
カメラ画像	~/ITS/waymo_outputs/cam_gray_test/training/<SEG>/FRONT/
静的点群地図	~/ITS/waymo_outputs/static_map_test/training/<SEG>/
アンカー情報	~/ITS/waymo_outputs/submaps_test/training/<SEG>/anchors.csv
サブマップ情報	~/ITS/waymo_outputs/submaps_test/training/<SEG>/submaps.json
アンカーRI画像	~/ITS/waymo_outputs/submaps_test/training/<SEG>/<submap_id>/anchor.png
Retrieval用ずらしRI	~/ITS/waymo_outputs/retrieval_pairs_test/training/<SEG>/<submap_id>/ri_*.png
学習用ペア表	~/ITS/waymo_outputs/retrieval_manifests_test/training/pairs.parquet
正規化統計	~/ITS/waymo_outputs/retrieval_manifests_test/training/modality_stats.json
評価用manifest	~/ITS/waymo_outputs/retrieval_eval_manifests_test/training/
学習結果	~/ITS/waymo_outputs/retrieval_train_test/
評価結果	~/ITS/waymo_outputs/retrieval_eval_test/

３０epochモデルを評価
cd ~/ITS/handover_shimizu/src/retrieval
source ~/ITS/.venv_waymo/bin/activate

CKPT="$HOME/ITS/waymo_outputs/retrieval_train_test/one_segment_fixed_overfit/checkpoints/best.ckpt"

if [ ! -f "$CKPT" ]; then
  CKPT="$HOME/ITS/waymo_outputs/retrieval_train_test/one_segment_fixed_overfit/checkpoints/last.ckpt"
fi

python evaluate.py \
  --checkpoint "$CKPT" \
  --pairs "$HOME/ITS/waymo_outputs/retrieval_eval_manifests_test/training/eval_pairs.parquet" \
  --db "$HOME/ITS/waymo_outputs/retrieval_eval_manifests_test/training/eval_db.parquet" \
  --out_dir "$HOME/ITS/waymo_outputs/retrieval_eval_test/one_segment_fixed_overfit" \
  --name one_segment_fixed_trainset \
  --backbone resnet18 \
  --clusters 8 \
  --embed_dim 128 \
  --freeze_stages 2 \
  --image_h 192 \
  --image_w 320 \
  --batch_size 32 \
  --num_workers 0 \
  --device cuda \
  --amp false \
  --modality_stats "$HOME/ITS/waymo_outputs/retrieval_manifests_test/training/modality_stats.json" \
  --strict_ks "1,5,10,30,50" \
  --write_failures true \
  --write_topk_jsonl true \
  --topk_dump_k 20

  評価結果EVAL_OUT="$HOME/ITS/waymo_outputs/retrieval_eval_test/one_segment_fixed_overfit"

find "$EVAL_OUT" -maxdepth 2 -type f | sort
cat "$EVAL_OUT/one_segment_fixed_trainset_metrics.json"

fixed_trainset_topk_top20.jsonl
{
  "name": "one_segment_fixed_trainset",
  "strict": {
    "Q": 88,
    "DB": 88,
    "R@": {
      "1": 0.8181818181818182,
      "5": 0.9886363636363636,
      "10": 1.0,
      "30": 1.0,
      "50": 1.0
    },
    "MRR": 0.9048295454545454,
    "MedianRank": 1.0,
    "R@1%": 0.8181818181818182,
    "K@1%": 1
  },
  "soft": {}

cat "$OUT/one_segment_fixed_trainset_metrics.json"
ls -lh "$OUT"
{
  "name": "one_segment_fixed_trainset",
  "strict": {
    "Q": 88,
    "DB": 88,
    "R@": {
      "1": 0.8181818181818182,
      "5": 0.9886363636363636,
      "10": 1.0,
      "30": 1.0,
      "50": 1.0
    },
    "MRR": 0.9048295454545454,
    "MedianRank": 1.0,
    "R@1%": 0.8181818181818182,
    "K@1%": 1
  },
  "soft": {}
}total 892K
drwxr-xr-x 2 wakamatsu wakamatsu 4.0K May 15 09:37 _thumbs
-rw-r--r-- 1 wakamatsu wakamatsu  20K May 15 09:37 one_segment_fixed_trainset_anchor_failure_summary_strictR@5.csv
-rw-r--r-- 1 wakamatsu wakamatsu  382 May 15 09:37 one_segment_fixed_trainset_fail_strictR@5_all.csv
-rw-r--r-- 1 wakamatsu wakamatsu   92 May 15 09:37 one_segment_fixed_trainset_fail_strictR@5_by_pair.csv
-rw-r--r-- 1 wakamatsu wakamatsu 4.3K May 15 09:37 one_segment_fixed_trainset_fail_strictR@5_detail_top10.csv
-rw-r--r-- 1 wakamatsu wakamatsu 4.7K May 15 09:37 one_segment_fixed_trainset_fail_strictR@5_detail_top10.html
-rw-r--r-- 1 wakamatsu wakamatsu 4.2K May 15 09:37 one_segment_fixed_trainset_fail_strictR@5_detail_top10.jsonl
-rw-r--r-- 1 wakamatsu wakamatsu  516 May 15 09:37 one_segment_fixed_trainset_fail_strictR@5_summary.json
-rw-r--r-- 1 wakamatsu wakamatsu   94 May 15 09:37 one_segment_fixed_trainset_fail_strictR@5_top1_pair_confusion.csv
-rw-r--r-- 1 wakamatsu wakamatsu  333 May 15 09:37 one_segment_fixed_trainset_metrics.json
-rw-r--r-- 1 wakamatsu wakamatsu  60K May 15 09:37 one_segment_fixed_trainset_per_query.csv
-rw-r--r-- 1 wakamatsu wakamatsu 762K May 15 09:37 one_segment_fixed_trainset_topk_top20.jsonl


３－５このセグメントを作成して、学習を行う

リストができたら次に進む処理
3行出たら，次はこの5ステップを3セグメント分まとめて実行します
1. カメラ画像作成
2. 静的点群地図作成
3. アンカー作成
4. アンカーRI画像作成
5. Retrieval用ずらしRI作成

cd ~/ITS/handover_shimizu/src/data_make/train
source ~/ITS/.venv_waymo/bin/activate

EXP=multi3
BASE="$HOME/ITS/waymo_outputs"

CAM_ROOT="$BASE/cam_gray_$EXP"
MAP_ROOT="$BASE/static_map_$EXP"
SUBMAP_ROOT="$BASE/submaps_$EXP"
RETRIEVAL_PAIRS_ROOT="$BASE/retrieval_pairs_$EXP"

LIST="$BASE/lists/train_3_tfrecords.txt"

while read -r TFREC; do
  [ -z "$TFREC" ] && continue

  SEG=$(basename "$TFREC" .tfrecord)
  echo
  echo "============================================================"
  echo "[SEG] $SEG"
  echo "============================================================"

  echo "[1/5] export_cam_gray_txt.py"
  python export_cam_gray_txt.py \
    --tfrecord "$TFREC" \
    --out-root "$CAM_ROOT" \
    --subset training \
    --cam FRONT \
    --apply-clahe true \
    --save-raw true \
    --overwrite true

  echo "[2/5] make_static_map_txt.py"
  python make_static_map_txt.py \
    --tfrecord "$TFREC" \
    --out-root "$MAP_ROOT" \
    --subset training \
    --lidar-lines 32 \
    --ring-keep-even true \
    --bbox-vehicle-y-scale 2.5 \
    --overwrite true

  echo "[3/5] make_submaps_distance_txt.py"
  python make_submaps_distance_txt.py \
    --tfrecord "$TFREC" \
    --out-root "$SUBMAP_ROOT" \
    --subset training \
    --spacing-m 1.0 \
    --r-sub-m 10.0 \
    --anchor-mode greedy \
    --overwrite

  echo "[4/5] render_anchor_intensity.py"
  python render_anchor_intensity.py \
    --tfrecord "$TFREC" \
    --subset training \
    --maps-root "$MAP_ROOT" \
    --submaps-root "$SUBMAP_ROOT" \
    --cam FRONT \
    --int-mode global_map \
    --crop-mode circle \
    --crop-radius-m auto \
    --crop-margin-m 2.0 \
    --point-size 2 \
    --write-index true \
    --post-hist-eq true \
    --post-clahe true \
    --overwrite true

  echo "[5/5] make_retrieval_shifted_ri_txt.py"
  python make_retrieval_shifted_ri_txt.py \
    --tfrecord "$TFREC" \
    --subset training \
    --maps-root "$MAP_ROOT" \
    --submaps-root "$SUBMAP_ROOT" \
    --cam-gray-root "$CAM_ROOT" \
    --out-root "$RETRIEVAL_PAIRS_ROOT" \
    --cam FRONT \
    --num-samples-per-anchor 16 \
    --zero-ri-mode reuse \
    --overwrite true

done < "$LIST"


multi3のpairsparquetをつくる
コード
cd ~/ITS/handover_shimizu/src/retrieval
source ~/ITS/.venv_waymo/bin/activate

EXP=multi3
BASE="$HOME/ITS/waymo_outputs"

RETRIEVAL_PAIRS_ROOT="$BASE/retrieval_pairs_$EXP"
MANIFEST_ROOT="$BASE/retrieval_manifests_$EXP"

mkdir -p "$MANIFEST_ROOT/training"

python make_retrieval_pairs_parquet.py \
  --pairs-root "$RETRIEVAL_PAIRS_ROOT" \
  --subset training \
  --out-parquet "$MANIFEST_ROOT/training/pairs.parquet" \
  --out-summary-json "$MANIFEST_ROOT/training/summary.json" \
  --anchor-select random \
  --seed 42 \
  --min-ri 1 \
  --require-cam-exists true \
  --require-ri000 false \
  --drop-dup-cam true

  modality_stats.jsonを作成cd ~/ITS/handover_shimizu/src/retrieval
source ~/ITS/.venv_waymo/bin/activate

EXP=multi3
BASE="$HOME/ITS/waymo_outputs"
MANIFEST_ROOT="$BASE/retrieval_manifests_$EXP"

python compute_modality_stats_from_pairs.py \
  --pairs "$MANIFEST_ROOT/training/pairs.parquet" \
  --out-json "$MANIFEST_ROOT/training/modality_stats.json" \
  --only-label1 \
  --anchor-mode anchor_png
  
  
  multi3 の fixed_smoke モデルを評価する
  cd ~/ITS/handover_shimizu/src/retrieval
source ~/ITS/.venv_waymo/bin/activate

EXP=multi3
BASE="$HOME/ITS/waymo_outputs"

CKPT="$BASE/retrieval_train_$EXP/fixed_smoke/checkpoints/best.ckpt"

if [ ! -f "$CKPT" ]; then
  CKPT="$BASE/retrieval_train_$EXP/fixed_smoke/checkpoints/last.ckpt"
fi

echo "CKPT=$CKPT"
ls -lh "$CKPT"

python evaluate.py \
  --checkpoint "$CKPT" \
  --pairs "$BASE/retrieval_eval_manifests_$EXP/training/eval_pairs.parquet" \
  --db "$BASE/retrieval_eval_manifests_$EXP/training/eval_db.parquet" \
  --out_dir "$BASE/retrieval_eval_$EXP/fixed_smoke" \
  --name multi3_fixed_smoke \
  --backbone resnet18 \
  --clusters 8 \
  --embed_dim 128 \
  --freeze_stages 2 \
  --image_h 192 \
  --image_w 320 \
  --batch_size 32 \
  --num_workers 0 \
  --device cuda \
  --amp false \
  --modality_stats "$BASE/retrieval_manifests_$EXP/training/modality_stats.json" \
  --strict_ks "1,5,10,30,50" \
  --write_failures true \
  --write_topk_jsonl true \
  --topk_dump_k 20

  評価結果
  {
  "name": "multi3_fixed_smoke",
  "strict": {
    "Q": 295,
    "DB": 295,
    "R@": {
      "1": 0.2271186440677966,
      "5": 0.7559322033898305,
      "10": 0.9423728813559322,
      "30": 0.9966101694915255,
      "50": 1.0
    },
    "MRR": 0.440898355260959,
    "MedianRank": 3.0,
    "R@1%": 0.5457627118644067,
    "K@1%": 3
  },
  "soft": {}

確認したいこと
複数セグメントでもデータ生成が壊れていないか
pairs.parquet / modality_stats.json が正しいか
train.py が動くか
evaluate.py が動くか
checkpoint が読めるか
検索順位が計算できるか

16エポックでの学習結果　ロス
epoch,loss,lr
1,1.3623321950435638,0.0003
2,0.460955035355356,0.0003
3,0.3902992001838154,0.0003
4,0.4067186506258117,0.0003
5,0.3161168470978737,0.0003
6,0.375753672586547,0.0003
7,0.36515334082974327,0.0003
8,0.27180032307902974,0.0003
9,0.300997670325968,0.0003
10,0.29577426984906197,0.0003
11,0.22871390336917508,0.0003
12,0.24313746144374213,0.0003
13,0.26561521200670135,0.0003
14,0.24399427738454607,0.0003
15,0.2302823890414503,0.0003
16,0.3569098421268993,0.0003


epoch_cycleモデルを評価
[RESULT] Strict:
{
  "Q": 295,
  "DB": 295,
  "R@": {
    "1": 0.20677966101694914,
    "5": 0.7491525423728813,
    "10": 0.9322033898305084,
    "30": 1.0,
    "50": 1.0
  },
  "MRR": 0.42035075832610946,
  "MedianRank": 3.0,
  "R@1%": 0.5220338983050847,
  "K@1%": 3
}
[DONE]
(.venv_waymo) wakamatsu@1973700X:~/ITS/handover_shimizu/src/retrieval$ cat "$HOME/ITS/waymo_outputs/retrieval_eval_multi3/epoch_cycle_16ep/multi3_epoch_cycle_16ep_metrics.json"
{
  "name": "multi3_epoch_cycle_16ep",
  "strict": {
    "Q": 295,
    "DB": 295,
    "R@": {
      "1": 0.20677966101694914,
      "5": 0.7491525423728813,
      "10": 0.9322033898305084,
      "30": 1.0,
      "50": 1.0
    },
    "MRR": 0.42035075832610946,
    "MedianRank": 3.0,
    "R@1%": 0.5220338983050847,
    "K@1%": 3
  },
  "soft": {}


  date 5/17
  データ作成　1セグメントからmulti3まで完了
  粗探索
  multi3でfixed/epoch_cycleの学習と簡易評価まで完了


本番の評価は
train / validation / test を分ける
DB側とQuery側を分ける
ICPで座標系を合わせる
Soft Recallを出す

match_q_frames_to_db_anchors_by_icp.pyを実行
cd ~/ITS/handover_shimizu/src/data_make/valid_test
source ~/ITS/.venv_waymo/bin/activate

BASE="$HOME/ITS/waymo_outputs"
OUT="$BASE/mini_cross"

python match_q_frames_to_db_anchors_by_icp.py \
  --pairs_txt "$OUT/pairs_mini_same_segment.txt" \
  --icp_csv "$OUT/icp_identity.csv" \
  --db_submaps_root "$BASE/submaps_multi3" \
  --q_cam_root "$BASE/cam_gray_multi3" \
  --cam FRONT \
  --max_match_dist_m 5.0 \
  --invert_icp false \
  --out_csv "$OUT/mini_q_to_anchor.csv" \
  --write_dropped_csv true \
  --out_dropped_csv "$OUT/mini_q_to_anchor_dropped.csv"

  5/18時点
  pairs_mini_same_segment.txt
icp_identity.csv
        ↓
match_q_frames_to_db_anchors_by_icp.py
        ↓
mini_q_to_anchor.csv

Q側カメラフレーム
DB側アンカーRI
ICP変換
        ↓
Qフレームに対応するDBアンカーを決める

cross manifestを作成

manifestの出力を確認
cd ~/ITS/handover_shimizu/src/retrieval
source ~/ITS/.venv_waymo/bin/activate

BASE="$HOME/ITS/waymo_outputs"
OUT="$BASE/mini_cross"

find "$OUT/manifests" -maxdepth 1 -type f | sort

evaluate.pyを---icp_csv付きで実行
cd ~/ITS/handover_shimizu/src/retrieval
source ~/ITS/.venv_waymo/bin/activate

BASE="$HOME/ITS/waymo_outputs"
OUT="$BASE/mini_cross"

CKPT="$BASE/retrieval_train_multi3/epoch_cycle_16ep/checkpoints/best.ckpt"

python evaluate.py \
  --checkpoint "$CKPT" \
  --pairs "$OUT/manifests/mini_same_segment_all_pairs.parquet" \
  --db "$OUT/manifests/mini_same_segment_all_db.parquet" \
  --icp_csv "$OUT/icp_identity.csv" \
  --out_dir "$OUT/eval_epoch_cycle_16ep" \
  --name mini_same_segment_epoch_cycle_16ep \
  --backbone resnet18 \
  --clusters 8 \
  --embed_dim 128 \
  --freeze_stages 2 \
  --image_h 192 \
  --image_w 320 \
  --batch_size 32 \
  --num_workers 0 \
  --device cuda \
  --amp false \
  --modality_stats "$BASE/retrieval_manifests_multi3/training/modality_stats.json" \
  --strict_ks "1,5,10,30,50" \
  --soft_thresholds_m "2,5,10" \
  --soft_ks "1,5,10,30,50" \
  --write_topk_jsonl true \
  --topk_dump_k 20

  評価結果
  {
  "name": "mini_same_segment_epoch_cycle_16ep",
  "strict": {
    "Q": 198,
    "DB": 88,
    "R@": {
      "1": 0.025252525252525252,
      "5": 0.07575757575757576,
      "10": 0.19696969696969696,
      "30": 0.4292929292929293,
      "50": 0.6414141414141414
    },
    "MRR": 0.08301080109527129,
    "MedianRank": 37.5,
    "R@1%": 0.025252525252525252,
    "K@1%": 1
  },
  "soft": {
    "R_soft": {
      "2.0": {
        "valid": 0
      },
      "5.0": {
        "valid": 0
      },
      "10.0": {
        "valid": 0
      }
    },
    "SoftMRR_t2m": NaN,
    "SoftMedianRank_t2m": NaN,
    "SoftValid_t2m": 0
  }

  Soft評価を実行
  cd ~/ITS/handover_shimizu/src/retrieval
source ~/ITS/.venv_waymo/bin/activate

BASE="$HOME/ITS/waymo_outputs"
OUT="$BASE/mini_cross"

CKPT="$BASE/retrieval_train_multi3/epoch_cycle_16ep/checkpoints/best.ckpt"

python evaluate.py \
  --checkpoint "$CKPT" \
  --pairs "$OUT/manifests/mini_same_segment_all_pairs.parquet" \
  --db "$OUT/manifests/mini_same_segment_all_db.parquet" \
  --icp_csv "$OUT/icp_identity.csv" \
  --out_dir "$OUT/eval_epoch_cycle_16ep_fixsoft" \
  --name mini_same_segment_epoch_cycle_16ep_fixsoft \
  --backbone resnet18 \
  --clusters 8 \
  --embed_dim 128 \
  --freeze_stages 2 \
  --image_h 192 \
  --image_w 320 \
  --batch_size 32 \
  --num_workers 0 \
  --device cuda \
  --amp false \
  --modality_stats "$BASE/retrieval_manifests_multi3/training/modality_stats.json" \
  --strict_ks "1,5,10,30,50" \
  --soft_thresholds_m "2,5,10" \
  --soft_ks "1,5,10,30,50" \
  --write_per_query_csv true \
  --write_topk_jsonl true \
  --topk_dump_k 20

multi10学習の実行
cd ~/ITS/handover_shimizu/src/retrieval
source ~/ITS/.venv_waymo/bin/activate

BASE="$HOME/ITS/waymo_outputs"

python train.py \
  --pairs "$BASE/retrieval_manifests_multi10/training/pairs.parquet" \
  --modality-stats "$BASE/retrieval_manifests_multi10/training/modality_stats.json" \
  --out-dir "$BASE/retrieval_train_multi10/epoch_cycle_16ep" \
  --epochs 16 \
  --batch-size 16 \
  --backbone resnet18 \
  --clusters 8 \
  --embed-dim 128 \
  --freeze-stages 2 \
  --image-h 192 \
  --image-w 320 \
  --train-ri-mode epoch_cycle \
  --device cuda \
  --save-every-epochs 1
  　

  cd ~/ITS/handover_shimizu/src/retrieval
source ~/ITS/.venv_waymo/bin/activate

MANI="$HOME/ITS/waymo_outputs/old32_retrieval/manifests_wdrive/val4_test4"
ICP="/mnt/w/32line/datasets/splits/cross/all/icp_startend_staticmaps.csv"

CKPT="$HOME/ITS/waymo_outputs/old32_retrieval/train_wdrive_epoch_cycle_16ep_b8/checkpoints/last.ckpt"
STATS="$HOME/ITS/waymo_outputs/old32_retrieval/modality_stats.json"

python evaluate.py \
  --checkpoint "$CKPT" \
  --pairs "$MANI/val4_test4_all_pairs.parquet" \
  --db "$MANI/val4_test4_all_db.parquet" \
  --icp_csv "$ICP" \
  --out_dir "$HOME/ITS/waymo_outputs/old32_retrieval/eval_wdrive_val4_test4" \
  --name old32_wdrive_val4_test4 \
  --backbone resnet18 \
  --clusters 8 \
  --embed_dim 128 \
  --freeze_stages 2 \
  --image_h 192 \
  --image_w 320 \
  --batch_size 32 \
  --num_workers 0 \
  --device cuda \
  --amp false \
  --modality_stats "$STATS" \
  --strict_ks "1,5,10,30,50" \
  --write_per_query_csv true \
  --write_topk_jsonl true \
  --topk_dump_k 50

  4. 今回の結果の評価

今回の結果は，こう見ればよいです。

Strict R@10 = 0.331
Soft R@10@10m = 0.700

つまり，

完全一致アンカーはTop10に約33%
位置的に近いアンカーならTop10に約70%

です。

これは，画像単位で完全一致を当てるのは難しいが，近い場所の候補はある程度取れているという状態です。

修論の流れでは，この後に密探索をするので，Soft R@10 や Soft R@50 が特に重要になります。

5. 生成されたファイル

今回，評価結果として重要なファイルも作成できています。

old32_wdrive_val4_test4_metrics.json
old32_wdrive_val4_test4_per_query.csv
old32_wdrive_val4_test4_topk_top50.jsonl
old32_wdrive_val4_test4_fail_strictR@5_detail_top10.html

特に使うのは，

metrics.json      : 全体指標
per_query.csv     : クエリごとの順位・成否
topk_top50.jsonl  : 各クエリのTop50候補
fail_detail.html  : 失敗例の可視化

です。

epoch,R@1,R@5,R@10,R@50,MedianRank,Soft10m_R@10,Soft2m_MedianRank
001,0.008021390374331552,0.045454545454545456,0.0855614973262032,0.48663101604278075,51.0,0.35561497326203206,34.0
002,0.040106951871657755,0.12299465240641712,0.21657754010695188,0.660427807486631,28.5,0.6149732620320856,18.0
003,0.0106951871657754,0.07754010695187166,0.1497326203208556,0.6417112299465241,36.5,0.6203208556149733,20.0
004,0.0213903743315508,0.10962566844919786,0.19786096256684493,0.6176470588235294,39.0,0.6096256684491979,24.0
005,0.0427807486631016,0.14705882352941177,0.29411764705882354,0.8155080213903744,20.0,0.8101604278074866,10.0
006,0.034759358288770054,0.1711229946524064,0.27540106951871657,0.7005347593582888,29.5,0.6631016042780749,16.0
007,0.034759358288770054,0.1497326203208556,0.232620320855615,0.7032085561497327,27.0,0.7032085561497327,13.0
008,0.05614973262032086,0.19518716577540107,0.30213903743315507,0.7406417112299465,23.0,0.6149732620320856,14.0
009,0.0427807486631016,0.14705882352941177,0.24598930481283424,0.6764705882352942,31.5,0.660427807486631,19.0
010,0.0374331550802139,0.12299465240641712,0.22192513368983957,0.7540106951871658,28.0,0.6631016042780749,17.0
011,0.03208556149732621,0.13903743315508021,0.25133689839572193,0.8048128342245989,23.0,0.6898395721925134,14.0
012,0.029411764705882353,0.16042780748663102,0.25668449197860965,0.7459893048128342,29.0,0.6042780748663101,20.5
013,0.034759358288770054,0.18181818181818182,0.339572192513369,0.7914438502673797,19.0,0.6925133689839572,10.0
014,0.034759358288770054,0.20588235294117646,0.3048128342245989,0.7754010695187166,21.0,0.679144385026738,11.5
015,0.026737967914438502,0.17647058823529413,0.2914438502673797,0.7994652406417112,21.0,0.7379679144385026,10.0
016,0.040106951871657755,0.18449197860962566,0.2994652406417112,0.8235294117647058,20.5,0.7085561497326203,11.0
(.venv_waymo) wakamatsu@1973700X:~/ITS/handover_shimizu/src/retrieval$ 
これは各epochのチェックポイントでretrieval粗探索性能を評価下結果
そうするに、Camera画像から、正しいRIアンカー画像を検索できるかを評価している

R@K は Strict評価 です。
つまり，正解アンカー画像そのものを当てられたかを見ています。

一方，Soft10m_R@10 は Soft評価 です。
完全一致でなくても，正解位置から10m以内のアンカーなら正解扱いします。

Retrievalは後段のLoFTR/SuperGlueへ候補を渡す粗探索なので，実用上は Soft10m_R@10 もかなり重要です

Top-K候補ファイルの存在

drwxr-xr-x 2 wakamatsu wakamatsu  36K Jun  8 23:11 _thumbs
-rw-r--r-- 1 wakamatsu wakamatsu 450K Jun  8 23:11 old32_wdrive_test4_b16_epoch_014_anchor_failure_summary_strictR@5.csv
-rw-r--r-- 1 wakamatsu wakamatsu 159K Jun  8 23:09 old32_wdrive_test4_b16_epoch_014_fail_strictR@5_all.csv
-rw-r--r-- 1 wakamatsu wakamatsu  287 Jun  8 23:09 old32_wdrive_test4_b16_epoch_014_fail_strictR@5_by_pair.csv
-rw-r--r-- 1 wakamatsu wakamatsu 1.4M Jun  8 23:09 old32_wdrive_test4_b16_epoch_014_fail_strictR@5_detail_top10.csv
-rw-r--r-- 1 wakamatsu wakamatsu 1.3M Jun  8 23:11 old32_wdrive_test4_b16_epoch_014_fail_strictR@5_detail_top10.html
-rw-r--r-- 1 wakamatsu wakamatsu 1.6M Jun  8 23:09 old32_wdrive_test4_b16_epoch_014_fail_strictR@5_detail_top10.jsonl
-rw-r--r-- 1 wakamatsu wakamatsu 2.0K Jun  8 23:09 old32_wdrive_test4_b16_epoch_014_fail_strictR@5_summary.json
-rw-r--r-- 1 wakamatsu wakamatsu  314 Jun  8 23:09 old32_wdrive_test4_b16_epoch_014_fail_strictR@5_top1_pair_confusion.csv
-rw-r--r-- 1 wakamatsu wakamatsu 1.5K Jun  8 23:09 old32_wdrive_test4_b16_epoch_014_metrics.json
-rw-r--r-- 1 wakamatsu wakamatsu 433K Jun  8 23:09 old32_wdrive_test4_b16_epoch_014_per_query.csv
-rw-r--r-- 1 wakamatsu wakamatsu  11M Jun  8 23:09 old32_wdrive_test4_b16_epoch_014_topk_top50.jsonl


LoFTR密度のデバッグ実行は結果
(.venv_waymo) wakamatsu@1973700X:~/ITS/handover_shimizu/src/LoFTR$ OUT_DIR="$HOME/ITS/waymo_outputs/old32_dense/eval_loftr_retrieval_top10_debug100"

find "$OUT_DIR" -maxdepth 2 -type f | sort | head -n 50
cat "$OUT_DIR/summary.json"
/home/wakamatsu/ITS/waymo_outputs/old32_dense/eval_loftr_retrieval_top10_debug100/per_pair.csv
/home/wakamatsu/ITS/waymo_outputs/old32_dense/eval_loftr_retrieval_top10_debug100/per_query.csv
/home/wakamatsu/ITS/waymo_outputs/old32_dense/eval_loftr_retrieval_top10_debug100/per_query_strict.csv
/home/wakamatsu/ITS/waymo_outputs/old32_dense/eval_loftr_retrieval_top10_debug100/per_query_t10m.csv
/home/wakamatsu/ITS/waymo_outputs/old32_dense/eval_loftr_retrieval_top10_debug100/per_query_t1m.csv
/home/wakamatsu/ITS/waymo_outputs/old32_dense/eval_loftr_retrieval_top10_debug100/per_query_t2m.csv
/home/wakamatsu/ITS/waymo_outputs/old32_dense/eval_loftr_retrieval_top10_debug100/per_query_t5m.csv
/home/wakamatsu/ITS/waymo_outputs/old32_dense/eval_loftr_retrieval_top10_debug100/summary.json
/home/wakamatsu/ITS/waymo_outputs/old32_dense/eval_loftr_retrieval_top10_debug100/z_align_dz_offsets.csv
{
  "args": {
    "manifest": "/home/wakamatsu/ITS/waymo_outputs/old32_dense/loftr_manifest_from_retrieval_top10_test4_b16_epoch014.csv",
    "ckpt": "/mnt/w/32line/experiment/loftr/train/run/2026_01_28/coarse_fine_bacbone/latest_model.pth",
    "out_dir": "/home/wakamatsu/ITS/waymo_outputs/old32_dense/eval_loftr_retrieval_top10_debug100",
    "resize_long": 840,
    "stride": 8,
    "cell_point": "center",
    "device": "cuda",
    "amp_dtype": "bf16",
    "match_source": "auto",
    "model_enable_fine": "auto",
    "match_conf_th": 0.0,
    "max_matches": 4096,
    "min_depth": 0.5,
    "max_depth": 200.0,
    "pnp_reproj_th": 4.0,
    "pnp_max_iter": 2000,
    "pnp_confidence": 0.999,
    "pnp_refine": false,
    "gt_type": "t10m",
    "q_segments_file": "/mnt/w/32line/datasets/splits/cross/all/4/test4_q.txt",
    "assume_same_frame": false,
    "icp_csv": "/mnt/w/32line/datasets/splits/cross/all/icp_startend_staticmaps.csv",
    "icp_db_col": "",
    "icp_q_col": "",
    "icp_mat_col": "",
    "icp_mat_cols": "",
    "skip_error_rows": false,
    "no_verify_files": false,
    "limit": 100,
    "summary_radii": "1,2,5,10",
    "dist_bins": "0,1,2,3,4,5,6,10",
    "z_align": "segpair_median_nearest",
    "no_write_per_radius": false,
    "write_oracle": false,
    "save_match_vis": false,
    "match_vis_dir": "match_vis",
    "match_vis_gt_th": 4.0,
    "match_vis_max_draw": 0,
    "match_vis_line_thickness": 1,
    "match_vis_point_radius": 2,
    "match_vis_no_text": false,
    "match_vis_save_vertical": false,
    "match_vis_only_pnp_fail": false,
    "match_vis_only_gt_bad": false,
    "match_vis_gt_bad_th": 2.0,
    "match_vis_dist_bins": "0,0.5,1,2,5,10"
  },
  "n_pairs": 100,
  "pair_pnp_success_rate": 1.0,
  "pair_summary_by_dist": {
    "strict": {
      "n_pairs": 6,
      "pnp_success_rate": 1.0,
      "n_pnp_success": 6,
      "n_comparable": 6,
      "t_err_xy": {
        "median": 0.2242295327899232,
        "p90": 0.4545184841318284,
        "p95": 0.4769349776240157,
        "mean": 0.23983753620065404
      },
      "t_err_xyz": {
        "median": 0.2876706387402155,
        "p90": 0.4745459138164363,
        "p95": 0.48697541338924966,
        "mean": 0.27935445996753033
      },
      "t_err_z": {
        "median": 0.060128072652226194,
        "p90": 0.20441455081081727,
        "p95": 0.2280535015116989,
        "mean": 0.08948159762670353
      },
      "t_err_xyz_zalign": {
        "median": 0.28382592599530754,
        "p90": 0.474644555996865,
        "p95": 0.5021646303588803,
        "mean": 0.2802385033458748
      },
      "t_err_z_zalign": {
        "median": 0.10924090902461359,
        "p90": 0.1732711510939069,
        "p95": 0.1749729893495804,
        "mean": 0.1099299871734279
      },
      "rot_err_deg": {
        "median": 0.4926722452747814,
        "p90": 0.7243417897770024,
        "p95": 0.7746206929458387,
        "mean": 0.5246278200854096
      },
      "yaw_err_deg": {
        "median": 0.10600322462668998,
        "p90": 0.13297390258155417,
        "p95": 0.13683611928428974,
        "mean": 0.043400788509326084
      },
      "dist_xy_m": {
        "median": 0.18056347081028395,
        "p90": 0.33751650461856064,
        "p95": 0.36711214088233085,
        "mean": 0.19826282547872767
      },
      "delta_xy": {
        "median": -0.04197773281568422,
        "p90": 0.26787416164890393,
        "p95": 0.29438724716892495,
        "mean": -0.0415747107219264
      },
      "improved_rate_xy": 0.5,
      "ratio_xy": {
        "median": 1.754195675066844,
        "p90": 5.59049009322686,
        "p95": 6.481487692358167,
        "mean": 2.494479492411321
      },
      "dist_xyz_m": {
        "median": 0.22925650921210158,
        "p90": 0.375723803565415,
        "p95": 0.4024112767102014,
        "mean": 0.2554951847546993
      },
      "delta_xyz": {
        "median": -0.0856485222036831,
        "p90": 0.29987697621947584,
        "p95": 0.32658332132287476,
        "mean": -0.023859275212831016
      },
      "improved_rate_xyz": 0.5,
      "ratio_xyz": {
        "median": 1.5292276444814032,
        "p90": 2.6889866578745822,
        "p95": 2.7286131833226674,
        "mean": 1.4747517352904314
      }
    },
    "dist_0_1": {
      "n_pairs": 13,
      "pnp_success_rate": 1.0,
      "n_pnp_success": 13,
      "n_comparable": 13,
      "t_err_xy": {
        "median": 0.3726516211226911,
        "p90": 0.4814182763224532,
        "p95": 0.648088341749286,
        "mean": 0.3262445035251546
      },
      "t_err_xyz": {
        "median": 0.38580056841913124,
        "p90": 0.49610612686025685,
        "p95": 0.6515764535988207,
        "mean": 0.36712852140377356
      },
      "t_err_z": {
        "median": 0.07867360125517564,
        "p90": 0.2573632778163244,
        "p95": 0.2611637334387694,
        "mean": 0.11755100930074364
      },
      "t_err_xyz_zalign": {
        "median": 0.381636455971411,
        "p90": 0.5076686452312833,
        "p95": 0.6667815204091463,
        "mean": 0.35504320675580797
      },
      "t_err_z_zalign": {
        "median": 0.09741054424449658,
        "p90": 0.16842368674778357,
        "p95": 0.17259041579163747,
        "mean": 0.10771907452857914
      },
      "rot_err_deg": {
        "median": 0.47381715937026847,
        "p90": 0.6099671552510323,
        "p95": 0.7042302285094676,
        "mean": 0.4785540483822622
      },
      "yaw_err_deg": {
        "median": 0.1233133033090894,
        "p90": 0.14038262002324017,
        "p95": 0.15208027649176184,
        "mean": 0.08753228373564727
      },
      "dist_xy_m": {
        "median": 0.7277923873000811,
        "p90": 0.903409328719999,
        "p95": 0.9375830415098574,
        "mean": 0.5423912209336577
      },
      "delta_xy": {
        "median": 0.3209003326889459,
        "p90": 0.5759789605909594,
        "p95": 0.6028159615572344,
        "mean": 0.2161467174085031
      },
      "improved_rate_xy": 0.6923076923076923,
      "ratio_xy": {
        "median": 0.39714590946840495,
        "p90": 3.7028604692644684,
        "p95": 5.234091053574332,
        "mean": 1.4162860146967862
      },
      "dist_xyz_m": {
        "median": 0.7426945096090821,
        "p90": 0.9114714044949479,
        "p95": 0.9464042379788885,
        "mean": 0.5751776565170835
      },
      "delta_xyz": {
        "median": 0.3231213229921119,
        "p90": 0.5497163035909299,
        "p95": 0.5708102595488604,
        "mean": 0.20804913511331005
      },
      "improved_rate_xyz": 0.6923076923076923,
      "ratio_xyz": {
        "median": 0.5059673968593876,
        "p90": 2.5746126015312396,
        "p95": 2.673136047695348,
        "mean": 0.9681074197821077
      }
    },
    "dist_1_2": {
      "n_pairs": 13,
      "pnp_success_rate": 1.0,
      "n_pnp_success": 13,
      "n_comparable": 13,
      "t_err_xy": {
        "median": 1.1155337270785708,
        "p90": 1.8391935887166155,
        "p95": 2.1230094261635335,
        "mean": 1.1920045626781237
      },
      "t_err_xyz": {
        "median": 1.1626002747442852,
        "p90": 1.8458445530484422,
        "p95": 2.129928942527306,
        "mean": 1.2071084980765905
      },
      "t_err_z": {
        "median": 0.11228945277426305,
        "p90": 0.24159555361685536,
        "p95": 0.28139471781418013,
        "mean": 0.1260645642374363
      },
      "t_err_xyz_zalign": {
        "median": 1.1266788796063347,
        "p90": 1.8396378113553447,
        "p95": 2.1230106370872175,
        "mean": 1.1974520524118684
      },
      "t_err_z_zalign": {
        "median": 0.08493650938291353,
        "p90": 0.15378363480649285,
        "p95": 0.16118716090937255,
        "mean": 0.08599633231881618
      },
      "rot_err_deg": {
        "median": 0.3903509176861213,
        "p90": 0.4827228756519676,
        "p95": 0.5207816864388152,
        "mean": 0.38783273082361613
      },
      "yaw_err_deg": {
        "median": 0.11980299752326573,
        "p90": 0.24299718507681978,
        "p95": 0.2651706180696351,
        "mean": 0.11852527642713759
      },
      "dist_xy_m": {
        "median": 1.4910296188742893,
        "p90": 1.9594794514412803,
        "p95": 1.9728097487495617,
        "mean": 1.527845966140893
      },
      "delta_xy": {
        "median": 0.38652048693150787,
        "p90": 0.7837046313456115,
        "p95": 1.1368386995627016,
        "mean": 0.3358414034627692
      },
      "improved_rate_xy": 0.6153846153846154,
      "ratio_xy": {
        "median": 0.7514425255084071,
        "p90": 1.1416272288058886,
        "p95": 1.187038478281114,
        "mean": 0.7851675046161893
      },
      "dist_xyz_m": {
        "median": 1.49537106544504,
        "p90": 1.9594854488164537,
        "p95": 1.9725897089221707,
        "mean": 1.5323610671462582
      },
      "delta_xyz": {
        "median": 0.3876906681122503,
        "p90": 0.7744976499130986,
        "p95": 1.1059125989165313,
        "mean": 0.3252525690696677
      },
      "improved_rate_xyz": 0.6153846153846154,
      "ratio_xyz": {
        "median": 0.743347639403637,
        "p90": 1.1428183486032677,
        "p95": 1.1897353939031083,
        "mean": 0.7917058340865969
      }
    },
    "dist_2_3": {
      "n_pairs": 12,
      "pnp_success_rate": 1.0,
      "n_pnp_success": 12,
      "n_comparable": 12,
      "t_err_xy": {
        "median": 2.361119601800176,
        "p90": 3.2461502293147877,
        "p95": 3.851990304915806,
        "mean": 2.481322009625117
      },
      "t_err_xyz": {
        "median": 2.36267817522773,
        "p90": 3.250213371194638,
        "p95": 3.8551929951280717,
        "mean": 2.4848253424823756
      },
      "t_err_z": {
        "median": 0.12445658218383926,
        "p90": 0.15373749712229454,
        "p95": 0.1590655810807263,
        "mean": 0.11743577722951078
      },
      "t_err_xyz_zalign": {
        "median": 2.3665816026734765,
        "p90": 3.246163219167345,
        "p95": 3.852044154356218,
        "mean": 2.4830818355920035
      },
      "t_err_z_zalign": {
        "median": 0.04491239949300052,
        "p90": 0.13342870369397988,
        "p95": 0.176267304904411,
        "mean": 0.06009406176623718
      },
      "rot_err_deg": {
        "median": 0.3785174482738597,
        "p90": 0.7314039305878566,
        "p95": 0.7876190953713158,
        "mean": 0.4421107064355924
      },
      "yaw_err_deg": {
        "median": 0.046023523902874786,
        "p90": 0.20401750691441406,
        "p95": 0.24208910586721255,
        "mean": 0.05485459622887845
      },
      "dist_xy_m": {
        "median": 2.348804442905962,
        "p90": 2.9531687075368365,
        "p95": 2.972889715214608,
        "mean": 2.4559980880581413
      },
      "delta_xy": {
        "median": -0.027478789526961367,
        "p90": 0.7608920670837823,
        "p95": 1.3424414031547656,
        "mean": -0.025323921566976142
      },
      "improved_rate_xy": 0.5,
      "ratio_xy": {
        "median": 1.0181757645787557,
        "p90": 1.382251831443265,
        "p95": 1.4559045382423959,
        "mean": 1.0149711148204623
      },
      "dist_xyz_m": {
        "median": 2.3521770337276333,
        "p90": 2.952659164683426,
        "p95": 2.971206902204056,
        "mean": 2.457304575597374
      },
      "delta_xyz": {
        "median": -0.02619287412529392,
        "p90": 0.7716539103533918,
        "p95": 1.3424680909052353,
        "mean": -0.027520766885001546
      },
      "improved_rate_xyz": 0.5,
      "ratio_xyz": {
        "median": 1.017844987835458,
        "p90": 1.3867036783267979,
        "p95": 1.459972119638333,
        "mean": 1.0163297705514702
      }
    },
    "dist_3_4": {
      "n_pairs": 13,
      "pnp_success_rate": 1.0,
      "n_pnp_success": 13,
      "n_comparable": 13,
      "t_err_xy": {
        "median": 3.3764547603921304,
        "p90": 4.107619357964522,
        "p95": 4.947267434532156,
        "mean": 3.339501954595121
      },
      "t_err_xyz": {
        "median": 3.377213740095482,
        "p90": 4.1128876338279845,
        "p95": 4.951344197467106,
        "mean": 3.343056005101972
      },
      "t_err_z": {
        "median": 0.14746510485397835,
        "p90": 0.1773319858235169,
        "p95": 0.19481215010215458,
        "mean": 0.14182896840164444
      },
      "t_err_xyz_zalign": {
        "median": 3.377870102252633,
        "p90": 4.107873801841433,
        "p95": 4.947499594566194,
        "mean": 3.3399745934606817
      },
      "t_err_z_zalign": {
        "median": 0.028558103758811626,
        "p90": 0.08890403999047013,
        "p95": 0.09822559323539223,
        "mean": 0.03742339890030385
      },
      "rot_err_deg": {
        "median": 0.44429257327648813,
        "p90": 0.821269475813671,
        "p95": 0.8559528527932488,
        "mean": 0.5211947035022537
      },
      "yaw_err_deg": {
        "median": 0.0309572644976015,
        "p90": 0.23532139335433158,
        "p95": 0.24257024698914048,
        "mean": 0.06492145004281287
      },
      "dist_xy_m": {
        "median": 3.3216089590992657,
        "p90": 3.7731380604025135,
        "p95": 3.843681843432099,
        "mean": 3.3868216099134107
      },
      "delta_xy": {
        "median": -0.20810450157657234,
        "p90": 1.9022029939789298,
        "p95": 2.206926834156325,
        "mean": 0.04731965531829036
      },
      "improved_rate_xy": 0.46153846153846156,
      "ratio_xy": {
        "median": 1.0531661778904586,
        "p90": 1.2284331876120613,
        "p95": 1.4495802057052065,
        "mean": 0.996051208664591
      },
      "dist_xyz_m": {
        "median": 3.3301792026237957,
        "p90": 3.769545111684338,
        "p95": 3.8412881279849342,
        "mean": 3.386037403610086
      },
      "delta_xyz": {
        "median": -0.2149422371998182,
        "p90": 1.8997798493271167,
        "p95": 2.1977835912822843,
        "mean": 0.042981398508114395
      },
      "improved_rate_xyz": 0.46153846153846156,
      "ratio_xyz": {
        "median": 1.0549267650201106,
        "p90": 1.2317109784549467,
        "p95": 1.4523727174367238,
        "mean": 0.9975605331906758
      }
    },
    "dist_4_5": {
      "n_pairs": 11,
      "pnp_success_rate": 1.0,
      "n_pnp_success": 11,
      "n_comparable": 11,
      "t_err_xy": {
        "median": 4.361578078713876,
        "p90": 6.544195575674267,
        "p95": 6.731638230205956,
        "mean": 4.596100348038867
      },
      "t_err_xyz": {
        "median": 4.361611522783537,
        "p90": 6.545654008158387,
        "p95": 6.734894401337726,
        "mean": 4.599956539170143
      },
      "t_err_z": {
        "median": 0.13816896122250455,
        "p90": 0.2645039944877965,
        "p95": 0.28238910611455026,
        "mean": 0.16263746336442414
      },
      "t_err_xyz_zalign": {
        "median": 4.364235919226296,
        "p90": 6.544269949654506,
        "p95": 6.732002420272397,
        "mean": 4.596830400743574
      },
      "t_err_z_zalign": {
        "median": 0.044789395175584445,
        "p90": 0.13090523606446425,
        "p95": 0.14159692502160937,
        "mean": 0.06504997115297328
      },
      "rot_err_deg": {
        "median": 0.4275532568363377,
        "p90": 0.5833681861382392,
        "p95": 0.5948785777655561,
        "mean": 0.4439664217433577
      },
      "yaw_err_deg": {
        "median": 0.01065687445091612,
        "p90": 0.2943374889092354,
        "p95": 0.3009372470192915,
        "mean": 0.047927888067859345
      },
      "dist_xy_m": {
        "median": 4.260286453712942,
        "p90": 4.796602394556241,
        "p95": 4.855392844335103,
        "mean": 4.377036907261101
      },
      "delta_xy": {
        "median": -0.11183006037489207,
        "p90": 0.6044352478259039,
        "p95": 1.1935665876194406,
        "mean": -0.2190634407777658
      },
      "improved_rate_xy": 0.36363636363636365,
      "ratio_xy": {
        "median": 1.027279710614434,
        "p90": 1.4573251701018277,
        "p95": 1.4796558557781467,
        "mean": 1.049270098269938
      },
      "dist_xyz_m": {
        "median": 4.25111669879412,
        "p90": 4.793935495632309,
        "p95": 4.852570568933049,
        "mean": 4.3747038978263255
      },
      "delta_xyz": {
        "median": -0.12320111520822685,
        "p90": 0.6014141798991299,
        "p95": 1.1832620189720562,
        "mean": -0.22525264134381826
      },
      "improved_rate_xyz": 0.36363636363636365,
      "ratio_xyz": {
        "median": 1.0300944418635696,
        "p90": 1.4596948926948483,
        "p95": 1.4812816260253,
        "mean": 1.050758959000439
      }
    },
    "dist_5_6": {
      "n_pairs": 8,
      "pnp_success_rate": 1.0,
      "n_pnp_success": 8,
      "n_comparable": 8,
      "t_err_xy": {
        "median": 5.711888014331871,
        "p90": 6.572985571254927,
        "p95": 6.986826798400556,
        "mean": 5.795066346155311
      },
      "t_err_xyz": {
        "median": 5.715102010508625,
        "p90": 6.576480310723037,
        "p95": 6.990357508347145,
        "mean": 5.7976061369771665
      },
      "t_err_z": {
        "median": 0.17356829617229153,
        "p90": 0.21424617559473802,
        "p95": 0.2220190807296234,
        "mean": 0.16755789384190578
      },
      "t_err_xyz_zalign": {
        "median": 5.7119400208711415,
        "p90": 6.573141765829019,
        "p95": 6.98702822504726,
        "mean": 5.795268339700581
      },
      "t_err_z_zalign": {
        "median": 0.03460816480500739,
        "p90": 0.0725915943615817,
        "p95": 0.0867882828978132,
        "mean": 0.03723358774071883
      },
      "rot_err_deg": {
        "median": 0.41600162406490737,
        "p90": 0.5903859629045354,
        "p95": 0.6224687627802238,
        "mean": 0.4485322061938146
      },
      "yaw_err_deg": {
        "median": -0.035997723153300853,
        "p90": 0.32239886907352683,
        "p95": 0.32893134400685536,
        "mean": 0.06422399146285684
      },
      "dist_xy_m": {
        "median": 5.361889837685553,
        "p90": 5.617608775751186,
        "p95": 5.765882145385234,
        "mean": 5.387256435587963
      },
      "delta_xy": {
        "median": -0.11013972827422291,
        "p90": 0.16871185245270182,
        "p95": 0.34457878243313544,
        "mean": -0.4078099105673483
      },
      "improved_rate_xy": 0.25,
      "ratio_xy": {
        "median": 1.0214732223823453,
        "p90": 1.2140934616082308,
        "p95": 1.2809967066918486,
        "mean": 1.0752337060137682
      },
      "dist_xyz_m": {
        "median": 5.354600342090303,
        "p90": 5.610724713592579,
        "p95": 5.761376297048955,
        "mean": 5.381948640062692
      },
      "delta_xyz": {
        "median": -0.11662930815512107,
        "p90": 0.16468061333577752,
        "p95": 0.3415371364290152,
        "mean": -0.4156574969144742
      },
      "improved_rate_xyz": 0.25,
      "ratio_xyz": {
        "median": 1.0227032053376133,
        "p90": 1.2170783099586315,
        "p95": 1.2839114687397484,
        "mean": 1.0768314565468835
      }
    },
    "dist_6_10": {
      "n_pairs": 30,
      "pnp_success_rate": 1.0,
      "n_pnp_success": 30,
      "n_comparable": 30,
      "t_err_xy": {
        "median": 7.506663336469789,
        "p90": 9.717385934639205,
        "p95": 9.8261384683696,
        "mean": 7.72417916437217
      },
      "t_err_xyz": {
        "median": 7.50903217910254,
        "p90": 9.721273619019373,
        "p95": 9.828043565643387,
        "mean": 7.726657216480784
      },
      "t_err_z": {
        "median": 0.1899387113888764,
        "p90": 0.22759520357488017,
        "p95": 0.25575395911513243,
        "mean": 0.19305958275519686
      },
      "t_err_xyz_zalign": {
        "median": 7.506703533895843,
        "p90": 9.717976480322935,
        "p95": 9.826171572015477,
        "mean": 7.724269596213367
      },
      "t_err_z_zalign": {
        "median": 0.021138289796844845,
        "p90": 0.05822622189804038,
        "p95": 0.08638497743829267,
        "mean": 0.02804678167807424
      },
      "rot_err_deg": {
        "median": 0.4487661236506004,
        "p90": 0.580564938552049,
        "p95": 0.6625318142407155,
        "mean": 0.43363704584679963
      },
      "yaw_err_deg": {
        "median": 0.0020454316742331002,
        "p90": 0.1244948382393631,
        "p95": 0.16340107708373208,
        "mean": -0.004210203411850936
      },
      "dist_xy_m": {
        "median": 8.202578146048548,
        "p90": 9.72502767661875,
        "p95": 9.79320308038027,
        "mean": 8.018589891972368
      },
      "delta_xy": {
        "median": -0.03546581552262307,
        "p90": 1.4898142257168996,
        "p95": 2.043251151475001,
        "mean": 0.29441072760019815
      },
      "improved_rate_xy": 0.4,
      "ratio_xy": {
        "median": 1.0044946023315613,
        "p90": 1.0422211040926586,
        "p95": 1.0480103091280937,
        "mean": 0.9703185460740916
      },
      "dist_xyz_m": {
        "median": 8.193464797068309,
        "p90": 9.719702900524249,
        "p95": 9.790980912737748,
        "mean": 8.011817221813367
      },
      "delta_xyz": {
        "median": -0.04298073766083066,
        "p90": 1.4792626100104624,
        "p95": 2.0351405851500686,
        "mean": 0.28516000533258357
      },
      "improved_rate_xyz": 0.4,
      "ratio_xyz": {
        "median": 1.0054732593503108,
        "p90": 1.0438667600118168,
        "p95": 1.0495709071921346,
        "mean": 0.9714868527186089
      }
    },
    "dist_ge_10": {
      "n_pairs": 0
    }
  },
  "dist_bins": [
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    10.0
  ],
  "query_summary": {
    "n_query": 20,
    "pnp_success_rate": 1.0,
    "t_err_xy_median": 6.4319161067342545,
    "t_err_xy_p95": 9.83604544416674,
    "t_err_xyz_median": 6.433788584010902,
    "t_err_xyz_p95": 9.837783702763796,
    "t_err_xyz_zalign_median": 6.431931702478196,
    "t_err_xyz_zalign_p95": 9.836057890678951,
    "t_err_z_zalign_median": 0.017613424102250974,
    "t_err_z_zalign_p95": 0.09892933891364993,
    "rot_err_deg_median": 0.3979991976651507,
    "rot_err_deg_p95": 0.5848251830736851,
    "yaw_err_deg_median": 0.010525173832064638,
    "yaw_err_deg_p95": 0.19399873129182343,
    "success_xy@0.5m": 0.2,
    "success_xy@1.0m": 0.2,
    "success_xy@2.0m": 0.25,
    "success_xy@5.0m": 0.4,
    "success_xy@10.0m": 1.0
  },
  "query_summary_by_setting": {
    "strict": {
      "n_query": 20,
      "pnp_success_rate": 1.0,
      "t_err_xy_median": 2.180689787023491,
      "t_err_xy_p95": 9.083835538812206,
      "t_err_xyz_median": 2.1873976041168968,
      "t_err_xyz_p95": 9.086128478062795,
      "t_err_xyz_zalign_median": 2.1806909667430068,
      "t_err_xyz_zalign_p95": 9.083903416401858,
      "t_err_z_zalign_median": 0.03942763027551166,
      "t_err_z_zalign_p95": 0.17020784223369462,
      "rot_err_deg_median": 0.47458110561643035,
      "rot_err_deg_p95": 0.7442278225181876,
      "yaw_err_deg_median": 0.1056619821529523,
      "yaw_err_deg_p95": 0.20623353202481526,
      "success_xy@0.5m": 0.4,
      "success_xy@1.0m": 0.4,
      "success_xy@2.0m": 0.5,
      "success_xy@5.0m": 0.7,
      "success_xy@10.0m": 1.0
    },
    "t1m": {
      "n_query": 8,
      "pnp_success_rate": 1.0,
      "t_err_xy_median": 0.2708144251386028,
      "t_err_xy_p95": 0.40070283058169814,
      "t_err_xyz_median": 0.32983434787622384,
      "t_err_xyz_p95": 0.4770580735398087,
      "t_err_xyz_zalign_median": 0.29678064138407995,
      "t_err_xyz_zalign_p95": 0.4184361032247354,
      "t_err_z_zalign_median": 0.1162445686134248,
      "t_err_z_zalign_p95": 0.1673408458717013,
      "rot_err_deg_median": 0.4440979872713466,
      "rot_err_deg_p95": 0.544775252266168,
      "yaw_err_deg_median": 0.12343110273037894,
      "yaw_err_deg_p95": 0.15864148637059824,
      "success_xy@0.5m": 1.0,
      "success_xy@1.0m": 1.0,
      "success_xy@2.0m": 1.0,
      "success_xy@5.0m": 1.0,
      "success_xy@10.0m": 1.0
    },
    "t2m": {
      "n_query": 11,
      "pnp_success_rate": 1.0,
      "t_err_xy_median": 0.38230916203834675,
      "t_err_xy_p95": 2.180689787023491,
      "t_err_xyz_median": 0.46618838555810876,
      "t_err_xyz_p95": 2.1873976041168968,
      "t_err_xyz_zalign_median": 0.39452390232845447,
      "t_err_xyz_zalign_p95": 2.1806909667430068,
      "t_err_z_zalign_median": 0.08805089010688505,
      "t_err_z_zalign_p95": 0.16184314011906054,
      "rot_err_deg_median": 0.4143788151724247,
      "rot_err_deg_p95": 0.5179870047201289,
      "yaw_err_deg_median": 0.1233133033090894,
      "yaw_err_deg_p95": 0.1804445830879331,
      "success_xy@0.5m": 0.6363636363636364,
      "success_xy@1.0m": 0.6363636363636364,
      "success_xy@2.0m": 0.9090909090909091,
      "success_xy@5.0m": 1.0,
      "success_xy@10.0m": 1.0
    },
    "t5m": {
      "n_query": 14,
      "pnp_success_rate": 1.0,
      "t_err_xy_median": 0.9697444882169863,
      "t_err_xy_p95": 3.826987724669945,
      "t_err_xyz_median": 1.0088771917760635,
      "t_err_xyz_p95": 3.831585762436873,
      "t_err_xyz_zalign_median": 0.9749346706449881,
      "t_err_xyz_zalign_p95": 3.827063668866172,
      "t_err_z_zalign_median": 0.08635221482639821,
      "t_err_z_zalign_p95": 0.15943583978001072,
      "rot_err_deg_median": 0.4202443892702812,
      "rot_err_deg_p95": 0.5605721584671882,
      "yaw_err_deg_median": 0.11955795131831337,
      "yaw_err_deg_p95": 0.207575245480767,
      "success_xy@0.5m": 0.5,
      "success_xy@1.0m": 0.5,
      "success_xy@2.0m": 0.7142857142857143,
      "success_xy@5.0m": 1.0,
      "success_xy@10.0m": 1.0
    },
    "t10m": {
      "n_query": 20,
      "pnp_success_rate": 1.0,
      "t_err_xy_median": 6.4319161067342545,
      "t_err_xy_p95": 9.83604544416674,
      "t_err_xyz_median": 6.433788584010902,
      "t_err_xyz_p95": 9.837783702763796,
      "t_err_xyz_zalign_median": 6.431931702478196,
      "t_err_xyz_zalign_p95": 9.836057890678951,
      "t_err_z_zalign_median": 0.017613424102250974,
      "t_err_z_zalign_p95": 0.09892933891364993,
      "rot_err_deg_median": 0.3979991976651507,
      "rot_err_deg_p95": 0.5848251830736851,
      "yaw_err_deg_median": 0.010525173832064638,
      "yaw_err_deg_p95": 0.19399873129182343,
      "success_xy@0.5m": 0.2,
      "success_xy@1.0m": 0.2,
      "success_xy@2.0m": 0.25,
      "success_xy@5.0m": 0.4,
      "success_xy@10.0m": 1.0
    }
  },
  "oracle_summary_by_setting": {},
  "candidate_stats_by_setting": {
    "strict": {
      "avg_candidates": 1.0,
      "median_candidates": 1.0,
      "added_vs_prev_avg": 0.0
    },
    "t1m": {
      "avg_candidates": 1.625,
      "median_candidates": 2.0,
      "added_vs_prev_avg": 0.625
    },
    "t2m": {
      "avg_candidates": 2.3636363636363638,
      "median_candidates": 2.0,
      "added_vs_prev_avg": 0.7386363636363638
    },
    "t5m": {
      "avg_candidates": 4.428571428571429,
      "median_candidates": 3.5,
      "added_vs_prev_avg": 2.064935064935065
    },
    "t10m": {
      "avg_candidates": 5.0,
      "median_candidates": 3.0,
      "added_vs_prev_avg": 0.5714285714285712
    }
  },
  "z_align": {
    "mode": "segpair_median_nearest",
    "n_seg_pairs_with_offset": 1,
    "dz_offset_stats": {
      "median": -0.16936898167683978,
      "min": -0.16936898167683978,
      "max": -0.16936898167683978,
      "mean": -0.16936898167683978,
      "std": 0.0
    },
    "dz_offset_by_segpair": [
      {
        "db_seg": "segment-6410495600874495447_5287_500_5307_500_with_camera_labels",
        "q_seg": "segment-2367305900055174138_1881_827_1901_827_with_camera_labels",
        "dz_med": -0.16936898167683978
      }
    ]
  },
  "elapsed_sec": 53.100592613220215
}(.venv_waymo) wakamatsu@1973700X:~/ITS/handover_shimizu/src/LoFTR$ 

処理したペア数: 100
対象クエリ数: 20
PnP成功率: 100%
処理時間: 約53秒
t10m設定の中央値誤差: 6.43 m
success_xy@1.0m: 0.20
success_xy@2.0m: 0.25
success_xy@5.0m: 0.40
success_xy@10.0m: 1.00

t10m設定の中央値誤差: 6.43 m
success_xy@1.0m: 0.20
success_xy@2.0m: 0.25
success_xy@5.0m: 0.40
success_xy@10.0m: 1.00
10m以内には全クエリ入っているが，1〜2m以内の高精度位置推定はまだ弱い