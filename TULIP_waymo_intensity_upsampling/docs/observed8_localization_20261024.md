# 観測保持・領域補完・観測8ラインの実験手順

観測部分保持は、Aの有効画素ではAの反射強度と深度を採用し、Aの空白画素だけをBで補う画像上の診断処理です。合成後のヒストグラム平坦化とCLAHEで観測画素の最終輝度も変化します。点群の遮蔽をそのまま再現する物理レンダリングとは区別します。

既存の507 Queryでの位置推定（全Queryを分母とする1m以内）は、A 65.48%、B 46.94%、保持方式67.85%、C 71.99%でした。全体としてBから回復しましたが、Aに対して全指標で優位ではありません。領域ごとの補完はこの方式と別に、採用マスクを空白かつ指定領域に限定して検証できます。領域・閾値は374 Queryの独立validationで選び、507 Queryのtestで調整しません。

## 何を8ラインで学習するか

- Retrieval: 8ライン地図から描画した反射強度画像と、対応するカメラ画像を対照学習します。観測32ライン画像を復元するモデルではありません。
- LoFTR: 8ラインの反射強度画像とカメラ画像の正しい2D対応を、幾何情報から作り直して教師にします。Stage-1/Stage-2の学習対象です。
- PnP: 学習しません。LoFTR対応と深度・内部パラメータから位置姿勢を計算します。
- TULIP: 8→16または8→32の復元は別の学習です。今回追加したコードは、まず観測8ラインそのものの位置推定条件を準備します。既存16→32重みの8ライン対応は未確認です。

清水氏の研究は観測32ラインであり、8ラインはその方法の実験条件を変更したものです。学習手順を参考にしますが、修論に掲載された8ライン実験の再現とは呼びません。既存のStage-1のみの重みで8ラインを評価する場合も、8ラインで学習したとは呼びません。

## 追加したコード

- `tools/make_observed8_static_map.py`: 引継ぎコードをメモリ内で拡張し、TOP LiDARネイティブ64行の0,8,...,56を残す。元ソースは変更しない。`--max-frames`は一時出力での試験専用。
- `tools/run_observed8_localization.py`: 196学習segment、3 validationペア、5 testペアの役割を記録。元のカメラ画像とアンカー識別情報を使い、8ライン地図、画像、学習用ずらし画像、manifest、正規化統計、80epoch Retrieval学習、validationによる重み選択、test検索を段階実行する。

現在のplanは`outputs/observed8_localization_20261024/plan.json`です。prepareは実行済みです。地図・学習・推論はまだ実行していません。リング選択、構文、依存入力パスを検査済みです。全データ処理の完走を保証するものではないため、1segmentから確認します。テンプレートは既存のアンカー配置を保持します。既存学習用submapsには間隔が厳密に1mではないものも含まれ、アンカー配置まで修論と完全同一とは主張しません。

## 評価データから先に確認

```bash
PROJECT=/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
PY=/home/wakamatsu/ITS/.venv_waymo/bin/python
ROOT="$PROJECT/outputs/observed8_localization_20261024"
SCRIPT="$PROJECT/tools/run_observed8_localization.py"

# 最初のtest用DB segment。segment内は全フレーム・全アンカーを処理する。
"$PY" "$SCRIPT" data --root "$ROOT" --role test --limit 1

# test用DBの5segment全体。完了済み画像は引継ぎコードのresume処理でスキップする。
"$PY" "$SCRIPT" data --root "$ROOT" --role test
"$PY" "$SCRIPT" test-manifests --root "$ROOT"

# 従来モデルを固定した8ライン入力の影響。8ライン学習済みモデルの結果ではない。
"$PY" "$SCRIPT" evaluate-fixed --root "$ROOT"
```

出力は`retrieval_fixed/observed8_fixed_metrics.json`です。固定重みでのA/B/C比較に追加できる条件です。学習条件まで変更した後の8ライン結果とは分けて報告します。

## 8ラインでRetrievalを学習

```bash
PROJECT=/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
PY=/home/wakamatsu/ITS/.venv_waymo/bin/python
ROOT="$PROJECT/outputs/observed8_localization_20261024"
SCRIPT="$PROJECT/tools/run_observed8_localization.py"

# 新規196segmentの全フレーム重畳、全アンカー、ずらし画像の生成は大規模処理。
"$PY" "$SCRIPT" data --root "$ROOT" --role train --limit 1
"$PY" "$SCRIPT" data --root "$ROOT" --role train
"$PY" "$SCRIPT" data --root "$ROOT" --role val
"$PY" "$SCRIPT" manifests --root "$ROOT"
"$PY" "$SCRIPT" stats --root "$ROOT"
"$PY" "$SCRIPT" train --root "$ROOT"
"$PY" "$SCRIPT" select --root "$ROOT"
"$PY" "$SCRIPT" evaluate --root "$ROOT"
```

trainは学習開始時の`retrieval_train`既存ディレクトリを上書きしません。中断した学習のresumeは、記録されたcommand.jsonと引継ぎtrain.pyのresume設定を確認して別途行います。

モデルはResNet34、NetVLAD64、埋め込み256、512×864、80epoch、batch16、temperature0.12、AdamW LR3e-4/WD1e-4、cosine、gradient clipping1です。正規化は8ラインの学習画像から算出します。374validation QueryだけでSoft(5m) R@10を最大化するepochを選びます。元trainerの`best.ckpt`はStrict選択のため使用せず、全epochの記録から`selected_retrieval.json`に選択したcheckpointを保存します。

## LoFTR再学習の未解決項目

`handover_shimizu/src/LoFTR/tools/make_loftr_matches_gt_step2a_paper_final_fixed_cam_v7.py`は、同じフォルダのv6をimportしますが、現環境にはそのv6がありません。8ライン画像に既存32ラインのGTファイルだけを付け替えて学習する処理は用意していません。点の表示と有効深度が変わるためです。

8ラインLoFTRを再学習するには、ずらし画像・深度を生成し、この幾何GT生成器を復旧するか、新しい実装で座標系・遮蔽・粗密対応の検証を行う必要があります。Stage-1とStage-2 trainer自体は存在します。GT作成の確認前にその学習を開始しません。

## 8ライン検索からLoFTR/PnPへの接続

8ラインで学習したRetrievalのtest完了後は、明示的にLoFTR重みを指定して接続できます。以下の旧Stage-1重みで実行する場合、LoFTRは従来重み固定の参考評価です。

```bash
PROJECT=/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
PY=/home/wakamatsu/ITS/.venv_waymo/bin/python
ROOT="$PROJECT/outputs/observed8_localization_20261024"
SCRIPT="$PROJECT/tools/run_observed8_localization.py"
LOFTR=/home/wakamatsu/ITS/waymo_outputs/old32_dense/loftr_retrain_stage1_paper_wdrive/stage1_final_gstep715958.pth

"$PY" "$SCRIPT" dense-prepare --root "$ROOT" --loftr-checkpoint "$LOFTR"
"$PY" "$SCRIPT" dense-pilot --root "$ROOT"
# pilotの内容を確認した後に全507Queryへ広げる。
"$PY" "$SCRIPT" dense-all --root "$ROOT"
```

固定Retrievalの8ライン評価から接続する場合は、3行すべてに`--retrieval-source fixed`を付けます。出力は`dense_fixed`となります。8ラインで学習したLoFTRが用意できた場合は、別の実験保存先でそのcheckpointを指定します。

## 比較時の留意点

8ラインを作るために反射強度PNGの行を削ってはいけません。LiDARネイティブ走査線を選択してから、地図・アンカー画像・深度を再生成します。カメラ画像、アンカー位置、評価Query、DBアンカー集合は共通にします。固定モデル比較と各ラインで再学習したモデル比較は目的が異なります。

8ラインで再学習した結果と従来A/B/C結果の差には、ライン数だけでなく正規化・独立validation・重み選択も変わった影響が入ります。ライン数そのものの効果を比較する最終実験では、16・32ライン側も同じ学習・validation方針で再学習します。
