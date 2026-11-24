# 清水氏手法の再現調査：コード読解ガイド

## この資料の目的

別チャットで、清水一樹氏の修士論文に沿った粗探索・密探索の再現状況を、ソースコードと実行記録から段階的に確認するための案内です。

## 重要な作業ルール

- `handover_shimizu` は参照用です。コードを編集しないでください。
- `/mnt/w/waymo_tf`、`/mnt/w/32line/datasets` など入力データも変更しないでください。
- 変更が必要なら、まず内容・影響を説明し、ユーザーの了承を得てから `TULIP_waymo_intensity_upsampling` 内に複製または新規の確認コードを作ってください。
- 最初は読む・比較するだけにして、学習や評価を再実行しないでください。
- 結論では「確認済み」「推定」「未確認」を明確に分けてください。ファイル名だけで学習段階や由来を断定しないでください。
- CodexのシェルからWSLに入れない場合は、実行を装わず、ユーザーがWSLで実行できる読み取り専用コマンドを提示してください。

## 現時点で確認できた事実

### 論文の密探索条件

ユーザーがPDFから確認した記載では、密探索は507 Query、5セグメントペアで評価し、Strict（Queryごとの最近傍アンカー）と距離閾値1/2/5/10m以内の全候補ペア（Soft分析集合）を評価します。10mまでのペアは合計9,759件です。LoFTRの2D対応を反射強度画像の深度・indexで2D–3D対応にし、PnPを評価します。これはRetrieval Top-5だけをPnPへ渡す評価とは異なります。

### Stage-1ログ

`/home/wakamatsu/ITS/waymo_outputs/old32_dense/loftr_retrain_stage1_paper_wdrive/train_stage1.log` のNUL除去後のログから、以下を確認済みです。

- manifest: `/mnt/w/32line/datasets/loftr_pairs_shifted/training/manifest_matches_step2a_paper_final_wdrive.csv`
- usable rows: 715,958
- 初期重み: `./weights/mega_depth/outdoor_ds.ckpt`
- dataset mode: `fine_remap`（ログ説明ではStage2-like remap + 1-to-1 dedup）
- Stage-1終了時: epoch 1、global step 715,958、avg loss 5.343168
- このログはユーザー環境でのLoFTR再学習を示すが、論文の学習を完全再現した証拠とは限らない。

既存の密探索評価 `/home/wakamatsu/ITS/waymo_outputs/old32_dense/eval_dense_pose_retrain_stage1_final/summary.json` は `stage1_final_gstep715958.pth` を使っています。したがって、その評価はStage-1チェックポイントによる評価です。

### Stage-2関連コード

- `/home/wakamatsu/ITS/handover_shimizu/src/LoFTR/tools/train_stage2_step2a_fine.py` は `--init-ckpt` からロードし、fine部分を学習して通常は `state_dict`、`epoch`、`global_step` を保存する設計です。
- 同コードの `load_state()` は `ckpt['state_dict']` またはチェックポイント本体を `strict=False` でロードし、missing/unexpectedの件数だけ表示します。
- 学習ログ検索ではStage-2実行記録を発見できていません。記録が見つからなかったことは、実行されていない証明ではありません。

### Stage-2候補チェックポイントの構造

読み取り専用の `torch.load` で以下を確認済みです。

1. `/home/wakamatsu/ITS/handover_shimizu/src/LoFTR/runs/stageC_backbone_patchnce_safe/stage2_fine_best.pth`
   - top-level keys: `backbone`, `pos_encoding`, `loftr_coarse`, `loftr_fine`, `cfg`
   - backbone 214 entries、coarse 80 entries、fine 20 entries
   - cfgはResNetFPN、resolution (8,2)、fine window 5など
   - epoch/global_step等の学習履歴は今回の出力では見つかっていない。

2. `/home/wakamatsu/ITS/handover_shimizu/src/LoFTR/runs/merged_for_eval/b2_heads__stage2backbone.pth`
   - top-level keys: `pos_encoding`, `loftr_coarse`, `backbone`, `cfg`
   - `loftr_fine`は見当たらない。Stage-2 fineモデルの完成形と決めつけないこと。

この形式のため、先行テストで使った汎用ローダーの「10.6% key coverage」は、入れ子構造を展開できずに出た値であり、モデル全体との実際の互換性を表すとは限りません。StageC候補はfine重みを含みますが、学習履歴・正式な由来・正しいロード対応は未確認です。

## 推奨読解順と対象ファイル

### 1. 論文と再現対象の条件を固定する

PDF（ユーザー提供）:

- `/mnt/d/Users/wakamatsu.k/Desktop/shimizu.k.pdf`
- `/mnt/d/Users/wakamatsu.k/Desktop/shimizu.winf.pdf`

確認事項: 粗探索と密探索を区別する。密探索のStrict/Soft定義、ペア数、入力画像サイズ、LoFTR Stage-1/Stage-2説明、PnP成功条件・指標をページ番号付きで整理する。

### 2. Stage-1実行内容と評価結果

- ログ: `/home/wakamatsu/ITS/waymo_outputs/old32_dense/loftr_retrain_stage1_paper_wdrive/train_stage1.log`
- Stage-1スクリプト: `/home/wakamatsu/ITS/handover_shimizu/src/LoFTR/tools/train_stage1_step2a_backbone0_coarse_fast_bf16v2.py`
- 評価記録: `/home/wakamatsu/ITS/waymo_outputs/old32_dense/eval_dense_pose_retrain_stage1_final/summary.json`

確認事項: ログが示す学習設定と論文記載の差、学習スクリプトの実際の引数、stage1チェックポイントで評価した結果。

### 3. Stage-2訓練コードとチェックポイントの対応

- Stage-2学習: `/home/wakamatsu/ITS/handover_shimizu/src/LoFTR/tools/train_stage2_step2a_fine.py`
- Stage-2 smoke test: `/home/wakamatsu/ITS/handover_shimizu/src/LoFTR/tools/smoke_stage2_step2a_fine.py`
- GT対応データセット実装（必要箇所だけ読む）: `/home/wakamatsu/ITS/handover_shimizu/src/LoFTR/src/loftr/datasets/waymo_step2a_manifest_fine_dataset.py`
- モデル定義（fine/coarseのキー対応調査に必要な範囲）: `/home/wakamatsu/ITS/handover_shimizu/src/LoFTR/src/loftr/loftr_sfppr.py`
- 候補重み: `runs/stageC_backbone_patchnce_safe/stage2_fine_best.pth` と `runs/merged_for_eval/b2_heads__stage2backbone.pth`

確認事項: Stage-2候補の各サブ辞書をモデルのstate_dictへどう対応づけるか、tensor名・shape一致率、Stage-1からの継続学習か、fine重みが実際に更新・保存されたか。重みロード処理は評価実行前に専用の読み取り専用検査で確認する。

### 4. 評価ペアとGT対応の生成

- 主GT生成器: `/home/wakamatsu/ITS/handover_shimizu/src/LoFTR/tools/make_loftr_matches_gt_cross_eval_v3.py`
- v7ラッパー: `/home/wakamatsu/ITS/handover_shimizu/src/LoFTR/tools/make_loftr_matches_gt_step2a_paper_final_fixed_cam_v7.py`
- ペア選定元、ICP CSV、manifestの作成元をそれぞれ追う。

確認事項: 507 QueryとStrict/Soft全候補が入力manifestに存在するか、ICP変換方向、depth/indexの座標規約、生成GTのフィルタ条件。Top-5 manifestを密探索全ペア評価と混同しない。

### 5. LoFTR・PnP評価本体

- 評価器: `/home/wakamatsu/ITS/handover_shimizu/src/LoFTR/tools/eval_cross_match_to_pose_zalign_multi_visall_v2.py`
- LoFTRチェックポイント変換ラッパー: `/home/wakamatsu/ITS/handover_shimizu/src/LoFTR/tools/eval_cross_match_to_pose_zalign_multi_visall_v2_loftr.py`
- SuperGlue比較ラッパー: `/home/wakamatsu/ITS/handover_shimizu/src/LoFTR/tools/eval_cross_match_to_pose_zalign_multi_visall_v2_superglue.py`
- 時間計測派生: `eval_cross_match_to_pose_zalign_multi_visall_v2_timing.py` と `_timing_15.py`

確認事項: checkpoint configとstate_dictロード、LoFTRの画像順（RIがimage0、cameraがimage1）、深度/indexの参照、x-forward座標変換、PnP入力、ICP適用、query集約方法、Strict/Soft集計式。

`timing`と`timing_15`は本体の派生版です。精度評価に使う版を明示し、異なる版の結果を無条件に混ぜない。

## `tools/` の他ファイルについて

`recovered_stage2/` と `recovered_stage2_uncompyle/` は復元・逆コンパイル由来のコードが含まれています。正式な実行スクリプトと同一扱いせず、Stage-2候補の由来を探す補助資料として、ファイル先頭・コメント・呼び出し元・保存名を照合してください。`.pyc`しかないものは、実行の一次記録や正式なソースと断定しないでください。

## 次のチャットに最初に貼る指示文

```text
このプロジェクトで清水氏の修士論文に沿ったLoFTR密探索の再現状況を調査したいです。
最初に docs/code_reading_guide_shimizu_reproduction.md を読み、その手順と注意事項に従ってください。
特に handover_shimizu とWaymo入力データは編集せず、まずStage-2候補チェックポイントの由来と読み込み対応を読み取り専用で確認してください。
確認済み・推定・未確認を分け、論文の507 Query Strict/Soft全ペア条件とTop-5評価を区別してください。
最初に読む対象は train_stage2_step2a_fine.py、該当データセット、loftr_sfppr.py、評価器のcheckpointロード部分です。
作業を始める前に、今回読むファイル一覧と確認する問いを提示し、評価や学習の再実行・コード編集はまだ行わないでください。
```

