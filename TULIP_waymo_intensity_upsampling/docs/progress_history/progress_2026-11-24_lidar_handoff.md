# LiDARアップサンプリングと自車位置推定の引き継ぎ

更新日: 2026-11-24 (JST)

清水氏の粗密探索を基準に、実測LiDARのライン数、TULIPによる補完、反射強度画像の作り方が自車位置推定へ与える影響を調べている。直近の課題は、実測16ラインの画素を保持したまま、局所3D平面と整合するTULIP生成点だけを採用すると、粗探索から最終位置推定まで改善するかの確認。

## 評価条件と計算環境

- Aは実測点から選んだ16ライン、Bは通常TULIP 16→32、Cは実測点から選んだ32ライン。観測画素優先は、Aの有効画素を優先し、その空白をBで補う投影画像。A+Bを足して48ラインにする処理ではない。観測8ラインは元の実測LiDARから選んだ8ライン。8→16のTULIP評価は別実験。
- 主評価は507 Query／754 DB。検証は別区間の374 Query／443 DB。Strict R@10は指定の正解アンカー、Soft 2 m・5 m R@10は各距離内のアンカーの検索成功。最終位置のXY誤差1 m・2 mはLoFTR＋PnP後の指標で、分母は全Query。対応数やPnP成立数は別に記録する。
- 基準の粗探索はResNet34＋NetVLAD、入力864×512、観測32ライン系の固定重み `/home/wakamatsu/ITS/waymo_outputs/old32_retrieval/train_wdrive_paper80_b16/checkpoints/best.ckpt` と [正規化設定](../../configs/paper80_effective_modality_stats.json)。密探索は `/home/wakamatsu/ITS/waymo_outputs/old32_dense/loftr_retrain_stage1_paper_wdrive/stage1_final_gstep715958.pth`、CUDA fp16、既定長辺840。引き継ぎ実装と再現用重みを使用した値で、修論公表値の完全再現は主張しない。
- Gitルートは `/home/wakamatsu/ITS`、研究実装は `TULIP_waymo_intensity_upsampling/`、清水氏の引き継ぎ実装は同階層の `handover_shimizu/src/`。前処理・Retrievalは `/home/wakamatsu/ITS/.venv_waymo/bin/python`、LoFTRは `/home/wakamatsu/ITS/.venv_tulip/bin/python`。ICPは `/mnt/w/32line/datasets/splits/cross/all/icp_startend_staticmaps.csv`。
- 実験データ・モデル重み・生成画像は W ドライブの `/mnt/w/its_outputs` または研究ディレクトリの `outputs`。再開前に該当パスを確認する。ツールの一部は保存済みrunのパスを定数として使う。環境移行時は参照先を更新する。

## 日付順の経過

### 2026-10-19〜10-22　粗密探索と観測画素優先

- A/B/Cの同一Query・同一アンカー集合を用意し、固定モデルで粗探索を評価した。Strict R@10は A 396/507、B 328/507、C 444/507。Top-10をLoFTR＋PnPへ接続し、最大インライア数で最終候補を選んだ。
- Bでは実測点が点群に残っていても、画像への投影で手前の生成点に置き換わる。観測画素優先ではAの有効画素を残し、空白にBを追加する。507 Queryの1 m以内は通常B 238件、観測画素優先344件、A 332件、C 365件。観測画素優先の画像には合成後のコントラスト処理があり、後でAの最終PNG値を保持する方式も比較した。
- 実装は [観測画素優先DB作成](../../tools/build_observed_priority_database.py)、[Top-K粗密探索](../../tools/evaluate_retrieval_topk_pipeline.py)、[観測画素優先の評価](../../tools/run_observed_priority_study.py)。全件出力は `outputs/observed_priority_full_20261022`。

### 2026-10-24〜11月上旬　観測8ラインと学習条件

- 観測8ラインに対しRetrievalを再学習した。正規化と検証Soft指標を点検し、独立検証374 Queryでepoch 52を選択。507 QueryのStrict R@10は75.94%、Soft 5 m R@10は92.90%。32ライン系固定重みでの観測8ラインは68.64%、87.77%。
- 8ライン用Retrieval＋固定Stage-1 LoFTRの連結評価は1 m以内313/507、2 m以内409/507、XY誤差中央値0.601 m。8ラインLoFTR再学習は小規模試行まで確認。374件の試行で1 m以内は固定273件、240ペアで適応した重み265件。8ラインに合わせた全量学習の最終結果は要確認。関連コードは [8ラインの粗探索](../../tools/run_observed8_localization.py)、[検証重みの再選択](../../tools/recheck_observed8_retrieval.py)、[8ラインLoFTR学習](../../tools/run_observed8_loftr_full.py)。
- 16ライン用Retrieval、通常B用Retrievalの再学習用データ作成も別系統で準備。固定32ライン系重みでのA/B比較と混同しない。 [16ライン](../../tools/run_observed16_retrieval.py) · [通常B](../../tools/run_original_tulip_retrieval.py)

### 2026-11-11頃〜11-13　最近傍縮小、Closing、投影の調査

- Aアンカーの通常評価は1920×1280から864×512へ縮小。最近傍条件も同じ最終サイズで、アンカー側の縮小方法を変更。AのStrict R@10／Soft 5 m R@10は78.11%／95.07%、最近傍は51.87%／77.91%。保存先は `/mnt/w/its_outputs/retrieval_nearest_A_864x512`。
- 11-12に最近傍画像へ3×3 Closingを追加した試行は50.10%／80.08%。別条件として**通常バイリニア縮小後**のAに3×3 Closingを適用。非ゼロの二値マスクの穴を選び、最寄りの有効画素から強度をコピーした。密探索は元のA画像と実測深度を使い、候補の選び方が変わる効果を測定した。 [再現コード](../../tools/prepare_observed16_bilinear_close3.py)
- 507件のバイリニア版ClosingはSoft 5 m Top-10が483件、XY誤差1 m以内334件、2 m以内426件。Aは482／332／422件。保存先は `/mnt/w/its_outputs/retrieval_A_bilinear_close3_qXriv3`。1920×1280の点インデックスを使う3×3・5×5・7×7の実験は別条件。全3条件の保存済み評価は確認されていない。
- 既存描画は点サイズ2で投影中心と周囲最大12画素に同じ点を描く。画素上の有効判定にはこの広がりが含まれる。3次元対応でコピーされた深度と画像座標を組み合わせる場合、特に遮蔽や深度境界で元の3D点からずれる可能性がある。

### 2026-11-16　観測画素優先のPnP内部の診断

- 同じ画像・同じLoFTR対応で、実測対応とTULIP生成対応を点番号から識別。検証374件で、標準の候補選択による1 m以内は全生成対応を許可すると268件、深度・局所平面でPnP対応を選別すると263件。僅差時に他候補の位置と比較する選択では269件と265件。
- この局所平面は**PnPへ渡す対応の選別**。生成画像を作る段階で点を選別する11-20の実験と区別する。 [PnPへの寄与](../../tools/audit_priority_pnp_sources.py) · [フィルタによる解き直し](../../tools/replay_priority_pnp_depth_filter.py)

### 2026-11-17　374件での画像・領域選択の切り分け

- 別区間374件で、AのStrict／Soft 2 m／Soft 5 m Top-10は277／324／343件、通常Bは230／291／330件。Aを生成点で置き換える条件は247／300／328件、置き換え＋空白追加は234／294／340件。後者で深度差2 m超の置き換えを抑えると266／310／337件だった。 [画素差し替えの実装](../../tools/run_tulip_pixel_ablation374.py)
- バイリニア＋Closingは282／329／343件。Aの最終PNG値を保持した全空白補完は278／323／341件。保存先は `/mnt/w/its_outputs/validation374_fill_retrieval_FFm1ii/retrieval_comparison.csv`。
- **2D深度境界での領域選択**は、Aの空白に入るTULIP画素の周囲11×11画素を調べ、実測深度幅が1 mを超える場合に補完を外す。画像443枚とmanifestを `outputs/validation374_region_prepare_RW1Qe0` に作成。補完候補画素の28.25%を除外。選んだ9 Queryでは、A値保持の全空白補完から領域制限を加えるとSoft 2 mが4→5件、Soft 5 mが7→6件。全374件での領域制限付き結果は要確認。 [2D領域選択の実装](../../tools/run_fill_followup374.py)

### 2026-11-18〜11-20　局所対応、深度の補完、3D平面による生成点選別

- 実測32ラインでは見えるが16ラインで失われる点を、同じアンカー・複数Queryで診断。27ペアの固定アンカー比較は1 m以内 A 20件、B 18件、C 21件。局所例では深度補完が有効だったが、374 Query・3,740候補へ広げるとAの1 m以内は276→271件、XY中央値は0.450→0.481 m。 [局所照合](../../tools/inspect_ACB_local_correspondences374.py) · [複数Query比較](../../tools/run_ACB_multiquery374.py) · [深度補完による解き直し](../../tools/replay_A_depth_fill374.py)
- **画像作成時に3D平面で空白を選別**。生成点の半径1 m内の実測点を最大16点取得し、6点以上かつ非退化、平面への当てはめRMS ≤ 5 cm、生成点と平面の距離 ≤ 10 cmの場合に採用。観測箇所の最終PNG値を保持し、採用した箇所は強度・深度・点番号を同じ生成点と結び付ける。
- 11-20の試行は9アンカーで候補画素の約76.6%を採用。 `outputs/tulip_plane_holes374_sXftRs/pixel_counts.csv` に9行、全件の `comparison.csv` は2026-11-24の確認時点で未作成。 [3D領域選択の実装](../../tools/select_tulip_holes_by_3d_plane374.py)

### 2026-11-24　8・16・32ラインの画像とLoFTR＋PnP可視化

- 3場面で同じカメラ画像と正解最近傍のアンカー位置を固定し、同じStage-1 LoFTRとPnP設定で9条件を実行。場面1の8／16／32ラインはLoFTR対応94／86／140、PnPインライア36／31／40、XY誤差0.979／0.657／0.620 m。場面3は8→32で対応570→832、位置誤差0.039→0.060 m。軒並み対応数が増える、または誤差が減るという結論は3場面からは出せない。
- `/mnt/w/its_outputs/observed_line_match_compare_pn1u6o40` に強度画像とマッチング図各6枚、`match_comparison.csv` を保存。緑の線は正解姿勢での再投影誤差4画素以内。PnPインライア数は推定姿勢に対する別指標。実測ラインの既存アンカー同士なので反射強度の正規化上限にも差がある。 [可視化の再実行](../../tools/render_observed_line_match_comparison.py)

## 再開方法

局所3D平面の9アンカー試行で使った設定のまま、同じrunに全443アンカーを追加して374 Queryを評価できる。既存の9アンカーは再利用される。保存先の `protocol.json` と入力側の参照パスを確認してから実行する。

```bash
PROJECT=/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
RUN="$PROJECT/outputs/tulip_plane_holes374_sXftRs"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/wakamatsu/ITS/.venv_waymo/bin/python -u \
  "$PROJECT/tools/select_tulip_holes_by_3d_plane374.py" \
  --output "$RUN" --limit 0 --evaluate
```

1. `pixel_counts.csv` でDBが443枚揃ったことを確認する。 `comparison.csv` と `transitions.csv` で実測A・全空白補完・3D選別補完を比較する。Soft 2 m Top-10の基準はA 324/374、A最終PNG値保持の全空白補完323/374。
2. 同じ374 Queryで境界除外と局所平面選別を比べる場合は、[run_fill_followup374.py](../../tools/run_fill_followup374.py) の `region-full --output <新しい出力先>` を使う。9件のパイロットや507件の主評価とは分けて記録する。
3. 3D選別後のTop-10と対応する反射強度・深度・点番号をLoFTR＋PnPへ渡し、追加点の寄与と最終位置の1 m・2 m以内件数を測る。検証セットで条件を決め、507 Queryは最後に評価する。

## 経過の出典

2026-11-12〜20は復元された会話 `LiDAR_chat_recovered_2026-11-12_to_20.txt` と対応する保存結果から整理。11-24は保存済み可視化と対応CSVを使用。復元ファイルには画像やツールログが含まれず、上の出力パスを実験の確認に使う。
