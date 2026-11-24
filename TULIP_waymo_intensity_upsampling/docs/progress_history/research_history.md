# LiDAR研究履歴

更新 2026-11-25（JST）。Gitのコミット、保存済みの実験記録、参照できた関連Codexチャットを日付順に索引化した。実験を再開する場合は、[実行手順・保存先を含む引き継ぎ書](progress_2026-11-24_lidar_handoff.md)から始める。

## Gitからたどる

Gitには、コミットした版のコードと文書が残る。変更した理由や、重み・評価出力の所在は、実験記録やチャットを併せて読むと追いやすい。日付は作業記録を優先し、対応するコミットを併記した。たとえば11月の研究コードの一部は、それ以前の作業も含めて11月24日にコミットされた。

リポジトリのルートで実行する。

```bash
# 研究ディレクトリのコミットを古い順に表示
git log --reverse --date=short --format='%ad %h %s' -- TULIP_waymo_intensity_upsampling

# 特定コミットの変更ファイルと差分を確認
git show --stat 7802dc1
git show 7802dc1 -- TULIP_waymo_intensity_upsampling/tools/select_tulip_holes_by_3d_plane374.py

# 引き継ぎ書の変更履歴を確認
git log --follow -- TULIP_waymo_intensity_upsampling/docs/progress_history/progress_2026-11-24_lidar_handoff.md
```

## 日付順の記録

| 日付 | 作業と判断 | 参照先 |
|---|---|---|
| 2026-02-18のチャット | KITTI側の点群から反射強度画像を作れるか調査。既存PLYの赤・青はPred/GTの識別色で、反射強度を取得するには元の強度配列か強度属性が必要と整理。 | Codex「Confirm edits for reflection image」 |
| 2026-05-19開始のチャット | 清水氏のアンカー描画のcoverageを調査。coverageは投影された画素の割合で、LoFTRで使える対応の多さは別に計測する方針。対象の個別ターンの日付は取得できなかったため、チャット作成日を記載。 | Codex「プログラムを実行」 |
| 2026-07-24〜07-30 | 低解像度入力とGTの確認を起点に、Waymo用の独立した実装を追加。清水氏の前処理、32→64のペア生成とセグメント・実フレームの照合を整備。 | Codex「反射強度アップサンプリング改善」「Add manifest照合コードと文書」／ `cefadda` `f649101` `e35080e` `1cdbc79` ／ [7月の記録](../progress_2026-07-30.md) |
| 2026-08-03〜08-05 | セグメント単位の分割、TFRecord抽出、学習ペア、range-onlyの過学習・チェックポイント評価を整備。 | `0dce243` `969328e` `0c769c9` `a964599` `5c5c94f` ／ [8月3日の記録](../progress_2026-08-03.md) |
| 2026-08-20〜09-09 | 距離の正規化・逆変換、点群化とカメラ投影を確認。32→64の手順を記録し、高反射画素の不足と生成行だけの距離尺度を点検。反射強度の損失を比較。 | `d419f5f` `c9124f9` `da3c4df` `f643f08` `4ad509c` ／ [点群評価](../20260821_scaled_pointcloud_evaluation.md) ・ [反射強度の実験まとめ](progress_2026-09-09_tulip_method_evaluation_research_summary.md) |
| 2026-09-15〜09-16 | 既存の学習・評価用12セグメントと清水氏のLoFTR対象セグメントに重なりがないことを確認。清水氏対象の実フレームで16→32ペアを準備し、静的地図・FRONT投影・LoFTR/PnPへつなぐ方針を整理。 | Codex「調査 Waymo対応移植計画」／ [16→32の引き継ぎ](progress_2026-09-16_tulip_16to32_handoff.md) |
| 2026-09-29〜10-02 | A/B/Cの静的地図比較を準備。KITTIのGT・Low・Predで距離の疑似カラーと凡例を見直した。 | [A/B/C静的地図](../20260929_static_maps_abc_5frames.md) ／ Codex「反射強度アップサンプリング改善」／ `649f052` `738abac` |
| 2026-10-08〜10-19 | 清水氏の論文条件、Stage-1再学習重みとStage-2候補の由来を調査。BatchNorm固定のStage-2を1000 step試行し、Strict 20ペアでPnP 19/20。主評価ではStage-1重みを使用した。 | [再現調査](../../HANDOVER_SHIMIZU_CONDITION_REPRODUCTION.md) ・ [コード読解](../code_reading_guide_shimizu_reproduction.md) ・ [Stage-2記録](../shimizu_stage2_weight_audit_20261019.md) ／ Codex「Add manifest照合コードと文書」／ `30449aa` |
| 2026-10-19〜10-29 | A（観測16）、B（TULIP 16→32）、C（観測32）の507 Query・754 DBとTop-10連結評価を整備。観測画素優先を評価し、観測8ライン用Retrievalは検証でepoch 52を選択。 | [連結評価](../integrated_top10_protocol_20261019.md) ・ [観測8ラインの重み選択](../observed8_validation_recheck_20261029.md) |
| 2026-11-11〜11-20 | 最近傍縮小、3×3 Closing、観測画素優先、2Dの深度境界を避ける補完、3D局所平面で生成点を選ぶ補完を比較。3D局所平面は9アンカーの試行で候補画素の約76.6%を採用。 | Codex「LiDAR点群補間技術」の復元ログ／ [11月の実験・数値・保存先](progress_2026-11-24_lidar_handoff.md) |
| 2026-11-24 | 3場面で同じカメラ画像・アンカー位置を使い、観測8・16・32ラインのLoFTR対応とPnPを可視化。研究コードと引き継ぎをGitに保存。 | [可視化コード](../../tools/render_observed_line_match_comparison.py) ／ `7802dc1` `9eb489f` |

チャット欄は、Codexで確認できた関連履歴の要約。11月12〜20日は提示された復元テキストを参照した。取得できたターンと既存記録を対象にしており、チャットの全文やツールログのアーカイブは含まない。

## 再開する作業

局所3D平面で選んだ生成点を使い、443アンカー・374 Queryの全件で、Aと全空白補完に対して粗探索が改善するかを測る。次に、採用した画素の強度・深度・点番号からLoFTR＋PnPを評価する。入力の保存先と実行コマンドは[引き継ぎ書の再開方法](progress_2026-11-24_lidar_handoff.md#再開方法)にある。
