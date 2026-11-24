# 通常TULIP復元BのRetrieval再学習前後の比較

2026-11-05時点。再学習前の列は既存の評価結果。再学習後の列は評価完了後に記入する。

| 指標 | 従来のRetrieval重み | Bで再学習したRetrieval重み | 差（後−前） |
| --- | ---: | ---: | ---: |
| Strict Recall@10 | 64.69%（328/507） | — | — |
| Soft 5 m Recall@10 | 84.02%（426/507） | — | — |
| Top-10位置推定のXY誤差中央値（比較可能な成功例） | 1.0542 m（486件） | — | — |
| XY誤差1 m以内（全Query） | 46.94%（238/507） | — | — |
| XY誤差2 m以内（全Query） | 61.34%（311/507） | — | — |
| XY誤差5 m以内（全Query） | 70.81%（359/507） | — | — |

同じ通常Bのテスト画像、507 Query、754 DBを使用する。再学習後の重みは別の検証セット（374 Query、443 DB）で選ぶ。位置推定は各Retrieval結果の上位10候補を使い、LoFTR重み、PnP設定、候補選択ルールを揃える。率の差はパーセントポイント、誤差の差はメートルで記録する。評価前に画像リストと設定を照合し、正規化設定が異なる場合はその差も記録する。

基準値の出典

- [Retrieval評価JSON](../outputs/paperlike_retrieval_B_507q/paperlike_B_507q_metrics.json)
- [Top-10位置推定JSON](../outputs/B_retrieval_top10_stage1_fp16_20261020/all507_pnp/integrated_summary.json)
- [Top-10評価条件](../outputs/B_retrieval_top10_stage1_fp16_20261020/protocol.json)

画像の観察記録

- 先頭の学習用画像の目視確認で、左側の壁が白くつぶれて見えた（ユーザー報告）。現時点では1例の観察。壁の画素が255に集中しているか、他のアンカーでも同様か、どの工程で生じたかは未確認。
