# 8ラインRetrievalの位置情報修正と重みの再選択

80epochの学習・checkpointは保存済み。元の検証ログはanchors.csv不足でSoftが0になっており、旧selected_retrieval.jsonは全epoch同点としてepoch1を選んでいる。旧ログ・重み・test結果は残し、新しい検証結果を別フォルダへ保存する。

## 実施順序

1. 元の32ライン用anchors.csvと8ラインのアンカーpose・配置が一致することを全gallery anchorで検査し、元CSVを8ラインのval/testフォルダへ追加する。
2. 検証374Query・443DBについて、全Queryの位置・ICP・Soft正解集合をチェックする。
3. epoch1と80を試験再評価し、Softが正しく計算されることを確認する。
4. 保存済み80個のcheckpointを374Queryだけで再評価する。FP32・batch32で元の学習中検証と合わせる。
5. validation Soft(5m) R@10で最大の重みを選ぶ。同値なら早いepochを選ぶ。全0なら停止する。
6. 選択が終わってから507Query・754DBのtestを実行する。

新しいスクリプトは既存のtrain.pyのeval_retrieval_metricsを呼び出す。画像・地図の作り直し、学習のやり直しはしない。元trainerと同じモデルを作り、checkpointをstrict=Trueでロードする。学習時に使った8ラインのmean/stdをそのまま使用する。

## コマンド

```bash
PROJECT=/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
PY=/home/wakamatsu/ITS/.venv_waymo/bin/python
ROOT="$PROJECT/outputs/observed8_localization_20261024"
SCRIPT="$PROJECT/tools/recheck_observed8_retrieval.py"

# コピー済みなら同じ内容を確認して再利用。異なるCSVは上書きしない。
"$PY" "$SCRIPT" repair --root "$ROOT"
"$PY" "$SCRIPT" check --root "$ROOT"

"$PY" "$SCRIPT" validate --root "$ROOT" --epochs 1,80
"$PY" "$SCRIPT" validate --root "$ROOT" --epochs all
"$PY" "$SCRIPT" select --root "$ROOT"
"$PY" "$SCRIPT" test --root "$ROOT"
```

全epochの評価は時間がかかる。既に再評価を完了したepochは再利用する。FP32でGPUメモリが足りなければ`--batch-size 16`と新しい`--output-dir`を全コマンドへ共通で指定する。異なるbatchや入力hashの結果は混在させない。

## 見る数値

- check: validation_queries=374、validation_db=443、query_positions_readable=374、全距離でempty_gt=0。
- validation: Strict_R@10、Soft_t5m_R@10、Strict_MedianRank。選択にはSoft_t5m_R@10のみを使う。選択基準をtest結果で変えない。
- Soft正解集合にはStrict正解も含む実装なので、再評価ではSoft R@10がStrict R@10以上になることも検査する。
- test: Strict R@1/5/10、Soft 2/5/10m R@10、順位中央値。374検証Queryの値と507test Queryの値を同一表の同一条件として混ぜない。

## 出力

`outputs/observed8_localization_20261024/validation_recheck_20261029/`

- protocol.json: 入力のhash、FP32、batch、選択データ数。
- position_check.json: 検証位置情報とSoft正解集合の検査結果。
- epoch_001.json ... epoch_080.json: 保存checkpointの再評価結果。
- validation_metrics.csv: 80epochの検証一覧。
- selected_retrieval.json: 修正後の選択epochとcheckpoint。
- test507/observed8_reselected_metrics.json: 選択後のtest評価。
- test507/observed8_reselected_topk_top50.jsonl: 今後LoFTRへ渡す候補。旧retrieval_testのepoch1候補とは分ける。
