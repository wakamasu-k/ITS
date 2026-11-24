# 清水氏条件の確認とTULIP粗探索改善：再点検・実験計画

作成日：2026-10-10。既存の引き継ぎ書、診断レポート、現行コード、保存済みJSON/CSVを再確認した記録。
今回の作業は調査と計画作成であり、新しい推論・画像生成・Retrieval実験は実行していない。

## 1. 研究目的と編集範囲

実測16層から生成した点群が、実測32層に近い画像検索と最終位置推定に役立つかを検証する。
単なる点数増加や画像の見た目ではなく、同じQuery/DB条件でのRecallとPnPの位置・姿勢誤差を成果とする。

- A：native ring 0,4,...,60の実測16層。
- B：Aの実測値を保持し、native ring 2,6,...,62相当をTULIPで生成。
- C_generated：TULIP側builderで再生成した実測32層。
- C_reference：清水氏の既存32層DB。既存RetrievalのCはこちら。

編集・新規出力は本プロジェクト内。handover_shimizu、Waymo原本、checkpoint、参照DBは参照専用。
git statusには既存変更が多数あり、handover_shimizu側にも変更が存在する。今回の作業で発生したものではなく、復元や整理をしない。

## 2. 再確認した資料

- HANDOVER_SHIMIZU_CONDITION_REPRODUCTION.md
- HANDOVER_OPEN_ITEMS_ADDENDUM.md
- outputs/condition_audit/condition_audit_20261010_225726_716668.md（5,987行の診断資料。関連節を現行ファイルと照合）
- tools/build_tulip_16_to_32_static_map.py
- tulip_adapter/tulip_static_map.py
- tulip_adapter/waymo_16_to_32_tulip.py
- scripts/run_shimizu_507q_754db_abc_maps.sh
- scripts/run_shimizu_754db_render_ab.sh
- tools/prepare_shimizu_retrieval_protocol.py
- tools/make_shimizu_abc_retrieval_manifests.py
- tools/compare_anchor_reproduction_sample.py
- tools/evaluate_retrieval_spatial_tolerance.py
- docs/shimizu_preprocessing_spec.md、docs/dataset_role_policy.md
- handover_shimizu/src/data_make/train/make_static_map_txt.py（関連処理）
- handover_shimizu/src/data_make/valid_test/render_anchor_intensity_txt_se3_fix.py（投影・後処理）
- handover_shimizu/src/retrieval/evaluate_timing.py（Soft評価）
- 保存済みmaps/*/training/*/stats.json、A/B/C metrics、画像比較summary、manifest CSV

プロジェクト内の全バイナリ・全画像を読み終えたという意味ではない。

## 3. 今回確定できたこと

### プロトコルと画像生成

A/B/Cのpairs CSVは各507行、db CSVは各754行。行順を維持して列ごとに比較し、差分はanchor_pngだけだった。
A/Bのanchor.pngとanchor.jsonはそれぞれ754個。
5走行から各1アンカーを選び、A/BとC_referenceの姿勢およびrender_profileを比較した。
intensity_up_valを除いた投影設定とanchor_pose_T_wvは5例とも一致した。全754件の姿勢照合は今後の監査対象。

実行スクリプトは清水氏のvalid_testレンダラを直接呼び出している。設定はFRONT、global_map、percentile 99.5、gamma 1、range correction 0、point_size 2、histogram equalization＋CLAHE、crop_mode=none。
crop_radius_m=10という値は渡しているが、crop_mode=noneなので円形10 m切り出しは有効でない。

### 75 mは二つの用途を持つ

1. builderでrangeを75で割って入力し、推論後75を掛けてmへ戻す正規化。
2. tulip_static_map.pyのMAX_GENERATED_RANGE_M=75.0による、Bの生成ringだけの安全フィルタ。

生成距離は有限かつ0より大きく75以下の場合に残す。超過点は75 mへ丸めず除去する。
実測ringにはこの上限を適用していない。この差は現在の手法の一部であり、実験条件として明示する。
世界座標XYZのノルムや水平距離をセンサーrangeとして使ってはいけない。

### 5走行の保存統計

|項目|A|B|C_generated|
|---|---:|---:|---:|
|処理フレーム合計|992|992|992|
|失敗フレーム|0|0|0|
|地図の点数|33,464,418|67,225,297|67,179,815|
|生成点数|0|33,760,879|0|

Bの75 m超の生成候補除去は64,988画素、無効pixel poseによる生成候補除去は7,226,680画素。
これらはフィルタ段階の候補画素数であり、最終地図点数と同じ母数ではない。
Bの最終点数はC_generatedより45,482点多いが、同じ位置にあることや正確な距離であることは保証しない。

### 既存Retrievalを保存JSONから確認

|地図|Strict R@1|Strict R@5|Strict R@10|2 m R@1|2 m R@5|2 m R@10|Strict中央値|
|---|---:|---:|---:|---:|---:|---:|---:|
|A|5.72%|24.65%|42.01%|19.72%|44.38%|59.76%|14|
|B|4.93%|17.95%|28.01%|14.40%|29.98%|45.96%|23|
|C_reference|5.92%|24.26%|42.41%|20.51%|42.60%|56.80%|13|

Softの1/2/5/10 m評価は既存JSONにすべて存在し、各閾値のvalidは507、empty_gtは0。
2 mは「正解アンカーから2 m」ではない。清水氏評価コードはDBアンカーをICPでQuery座標へ変換し、Queryの水平位置から2 m以内の同一pair候補をSoft正解集合にしている。
tools/evaluate_retrieval_spatial_tolerance.pyは直接XYを比較する簡易コードなので、異走行データへICPなしで流用しない。

## 4. 過去の説明で修正が必要なこと

1. Bの劣化原因を反射強度と断定しない。距離予測、有効点選別、投影競合、正規化、後処理、モデルとの分布差を切り分ける必要がある。
2. 旧画像比較は画像値>0をmaskにしている。反射強度0の有効点を欠損扱いするため、純粋な幾何coverageではない。
3. 旧MAEはA/CまたはB/Cの非ゼロ領域の和集合で計算している。片方だけが点を持つ画素も含むため、強度誤差だけの指標ではない。
4. 旧サンプル数も揃っていない。現時点のsummaryはA=6枚（IoU 85.956%、MAE_union 31.421）、B=4枚（90.265%、43.151）、再生成C=3枚（99.830%、3.529）。同一アンカー集合で再比較する。
5. 同じ99.5 percentile設定でもup値は地図ごとに変化する。追加点の強度が元の実測画素の表示まで変え得る。
6. 基準CのRetrieval再現と、C_generated地図の完全一致は別の確認である。
7. すでに5組の異走行ペアを使っている。「まだ複数走行を使っていない」は誤り。次は未使用の走行ペアへ拡張する段階。

例：128443走行のup値はA=0.64453125、B=0.51953125、C_reference=0.609375。
この違い自体は同じ正規化アルゴリズムを適用した結果であり、設定ミスとは断定しない。

## 5. 解決の順番と成果物

### P0：評価条件の監査と指標の修正

目的：改善の数値を信用できる土台を固める。

- 全754アンカーの姿勢、intrinsics、画像処理設定を照合。データ依存up値を固定設定とは分けて記録。
- C_referenceとC_generatedを5走行の共通アンカーで比較し、差が大きいものだけ地図生成まで追う。
- 全754画像を対象にpoint_index>=0とfinite depthから幾何maskを作り、幾何IoU、共通mask上のdepth誤差、共通mask上の強度MAE、union MAEを別々に出す。
- 強度の物理誤差はraw intensityで計測し、8bit・CLAHE後の画像誤差は検索入力の差として報告する。
- Query/DB/GTのIDと順序、ICP方向、checkpoint/設定のハッシュを保存する。
- 学習segmentと今回の10走行の重複を監査する。評価データへの再学習・ハイパーパラメータ選択を最終汎化結果と混同しない。

成果物案：outputs/research_recovery_20261010/condition_audit/ 以下の一致表、指標定義、共通アンカー比較CSV、JSON。
完了条件：意図した差と未解決差が一覧化され、geometry/intensityを混同しない指標が揃う。

### P1：B生成点の診断

- 5走行から時間的に離れたフレームを固定選出し、先にフレーム一覧を保存する。
- モデル出力を一度保存し、全対照実験で同じ出力を使う。再推論の揺れをintensity変更と混同しない。
- frame_index、native ring、column、raw range、raw intensity、生成フラグ、各フィルタ通過を保存する。
- 実測保持の完全一致、log1p/expm1、非有限値、0以下/75超距離、無効pose除去、BBox除去を順番に集計する。
- 32層GT由来pixel poseの利用が、16層だけを入力する実運用にどこまで依存しているかを明記する。これは今後の実用性評価項目。

成果物案：frame_diagnostics.csv、point_provenance.npz、除去理由別の表。
完了条件：飛散を防ぐ既存処理の効果と、失われる候補の内訳を説明できる。

### P2：intensityだけを変える対照実験

|実験|生成点range/XYZ/mask|生成点intensity|用途|
|---|---|---|---|
|B_original|既存B|TULIP予測|基準|
|B_interp_I|既存Bと同一|隣接実測ringの補間|実用候補|
|B_oracle_I（任意）|既存Bと同一|対応GT intensity|診断のみ。性能主張に使わない|

補間は物理的なbeam角を確認して重みを決める。片側だけ有効ならその値を使う。
両側無効なら元予測を維持し、fallback数を明示する。末端ringも片側扱いを記録する。
GTなしで使えるB_interp_Iを主要候補とし、GTを利用した版は原因解析に限定する。
近いringでも異なる物体を観測するため、補間が必ず正しいとは仮定しない。

まず5走行各3アンカー程度の15アンカーで投影品質と不変条件を確認する。
小規模結果だけでRecall改善を主張せず、その後754 DBを生成し507 Queryで評価する。
最初は清水氏と同じデータ依存正規化・後処理を保つ。必要に応じて、up値固定や後処理の有無を別の追加実験として扱う。

完了条件：XYZ・mask・実測intensityの不変が確認され、追加intensityだけの差とその検索への影響が測定できる。

### P3：粗探索の採用判定

- Strictおよび1/2/5/10 mのR@1/5/10、順位中央値を全体とpair別に報告。
- A/B/C/B_interp_Iで同じQueryを比較し、成功から失敗・失敗から成功へ変わった件数も出す。
- 隣接フレームの相関があるため、507件を独立試行として過度な有意性を主張しない。5pairのばらつきも報告する。
- 当面の目標はB_originalより改善し、AおよびC_referenceのTop-5候補回収率へ近づくこと。閾値だけを緩めて改善としない。
- C_referenceの2 m R@5も42.60%なので、C相当というだけで高精度位置推定が完成したとはみなさない。
- 基準に届かなくても科学的に条件と原因が整理できれば、残る差を明示して限定的な密探索実験へ進める。完全一致を永続的な足止め条件にしない。

### P4：LoFTR＋PnP、未使用走行での検証

固定した10～20 QueryでA/B/C/B_interp_IのTop-5を用いて動作確認する。
対応点数、PnP成功率、インライア率、位置・姿勢誤差を測る。
候補に適切なDBがない失敗と、適切なDBがあっても対応付けに失敗した例を分ける。
必要に応じて正解候補を与える診断も行うが、通常のend-to-end性能と分けて報告する。
その後507 Query、さらに学習・調整に使っていない別走行へ進む。Kを変更する場合は全地図で統一する。

## 6. 直近の実装対象

第一の実装は全アンカーの条件監査とpoint_index/depthを使う比較レポート。
既存のcompare_anchor_reproduction_sample.pyは過去結果の意味を保つため、新しい評価スクリプトと新しい出力先を作る。
この結果を見てからbuilderのintensity対照版を追加する。
本計画では重い推論・全DB再生成はまだ開始していない。

## 7. 実行環境

このチャットの通常サンドボックス実行とファイル読取は2026-10-10に復旧した。
欠落していたcodex-linux-sandboxリンクを稼働中本体へ復元した一時的な復旧であり、アプリ再起動後の永続性は未確認。
研究データの欠落と実行ヘルパーの欠落は別々に記録する。
