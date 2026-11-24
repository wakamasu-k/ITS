# 清水氏条件再現・TULIP A/B/C 実験 引き継ぎ書

作成日: 2026-10-08  
対象: `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling`  
参照専用: `/home/wakamatsu/ITS/handover_shimizu`

## 1. この引き継ぎ書の目的

本研究では、清水氏の静的点群地図と反射強度画像DB作成条件を基準にして、TULIPによる16層から32層への点群アップサンプリングを評価した。

比較対象は次の3種類である。

| 地図 | 内容 |
|---|---|
| A | 実測16層点群のみ |
| B | 実測16層 + TULIPで生成した追加16層 |
| C | 実測32層。清水氏の32層地図に相当する基準 |

目的は、点群密度の増加が反射強度画像、粗探索Retrieval、将来のLoFTR/PnP位置推定に与える影響を確認することだった。

## 2. 編集方針と保護対象

編集可能な主対象:

`/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling`

参照のみで扱う対象:

`/home/wakamatsu/ITS/handover_shimizu`

Waymo TFRecordや既存の清水氏生成データも変更していない。

## 3. 参照した清水氏側の主要ファイル

静的地図作成:

`/home/wakamatsu/ITS/handover_shimizu/src/data_make/train/make_static_map_txt.py`

反射強度画像生成:

`/home/wakamatsu/ITS/handover_shimizu/src/data_make/train/render_anchor_intensity_txt.py`

カメラ画像前処理:

`/home/wakamatsu/ITS/handover_shimizu/src/data_make/train/export_cam_gray_txt.py`

これらは参照用であり、変更していない。

## 4. 追加・編集した主なTULIP側ファイル

清水氏条件のプロトコル作成:

`tools/prepare_shimizu_retrieval_protocol.py`

5組の走行ペアから、507 Query／754 DBの固定プロトコルを作る。

5組のDB地図生成:

`scripts/run_shimizu_507q_754db_abc_maps.sh`

A/B/Cの静的地図を作る。既存の完成地図を上書きしない設定で使う。

清水氏と同じ投影でA/Bをレンダリング:

`scripts/run_shimizu_754db_render_ab.sh`

A/Bの754枚ずつの反射強度画像、anchor.json、anchor_index.npzを生成した。

Retrieval用manifest作成:

`tools/make_shimizu_abc_retrieval_manifests.py`

A/B/Cそれぞれについて、同じ507 Queryと754 DBを使うmanifestを作る。

アンカー画像比較:

`tools/compare_anchor_reproduction_sample.py`

同一アンカーのcoverage、画素差、強度差を比較する。

## 5. 固定した清水氏条件

Retrieval条件:

- Query: 507枚
- DB: 754枚
- 走行ペア: 5組
- モデル: ResNet18系
- clusters: 8
- embedding dimension: 128
- freeze stages: 2
- 入力サイズ: 192 x 320
- checkpoint: `.../old32_retrieval/train_wdrive_epoch_cycle_16ep_b8/checkpoints/epoch_016.ckpt`
- batch size: 64
- AMP: false
- device: CUDA

入力正規化:

`configs/shimizu_retrieval_modality_stats.json`

- camera mean: 0.4401616253
- camera std: 0.2226330048
- anchor mean: 0.2048222792
- anchor std: 0.2914166740

反射強度画像条件:

- 解像度: 1920 x 1280
- 視点: FRONT
- 投影: `global_map`
- percentile: 99.5
- gamma: 1
- range correction: 0
- point size: 2
- histogram equalization: 有効
- CLAHE: 有効
- CLAHE clip limit: 2
- CLAHE tile: 8
- post mask: covered pixels only
- index output: 有効
- `crop_mode`: `none`

### 重要: サブマップ半径10 mについて

メタデータにはサブマップ半径10 mの設定があるが、実際の清水氏の反射強度画像投影では `crop_mode=none` であり、画像生成時に10 mで点群を切り取っていない。

したがって、現在の再現は「10 m範囲だけを投影する実験」ではなく、清水氏と同じく広い静的地図をアンカー視点へ投影する実験である。10 m crop版を作る場合は、清水氏条件とは別実験として名前を分ける必要がある。

## 6. 使用した走行ペア

清水氏のプロトコルに合わせ、次の5組を使用した。

| DB走行 | Query走行 | Query数 | DB数 |
|---|---|---:|---:|
| 641049 | 236730 | 186 | 209 |
| 128443 | 171355 | 55 | 100 |
| 138235 | 663760 | 111 | 215 |
| 126816 | 366934 | 68 | 106 |
| 848780 | 183339 | 87 | 124 |
| 合計 |  | 507 | 754 |

プロトコル出力:

`outputs/shimizu_retrieval_protocol_507q_754db`

## 7. 出力フォルダ

ABC静的地図:

`outputs/shimizu_protocol_507q_754db_abc/maps`

5走行 x A/B/C = 15個の `map_static.npz` を生成済み。全フレームの生成失敗は0件。

Aの反射強度画像:

`outputs/shimizu_protocol_507q_754db_abc/anchors_A`

Bの反射強度画像:

`outputs/shimizu_protocol_507q_754db_abc/anchors_B`

各々754枚のanchor.png、anchor.json、anchor_index.npzを生成済み。

比較結果:

- `outputs/shimizu_protocol_507q_754db_abc/A_vs_reference_sample/summary.json`
- `outputs/shimizu_protocol_507q_754db_abc/B_vs_reference_sample/summary.json`
- `outputs/shimizu_protocol_507q_754db_abc/C_reproduction_comparison_sample/summary.json`

Retrieval結果:

- A: `outputs/shimizu_protocol_507q_754db_abc/retrieval_A_retry1/shimizu_507q_754db_A_metrics.json`
- B: `outputs/shimizu_protocol_507q_754db_abc/retrieval_B/shimizu_507q_754db_B_metrics.json`
- C再現: `outputs/shimizu_retrieval_reproduction_epoch16_retry1/shimizu_test4_reproduced_metrics.json`

## 8. 清水氏結果の再現確認

Cを使ったRetrievalは、清水氏の保存済み結果と一致した。

| 評価 | 保存結果 | 再現結果 |
|---|---:|---:|
| Strict R@1 | 5.92% | 5.92% |
| Strict R@5 | 24.26% | 24.26% |
| Strict R@10 | 42.41% | 42.41% |
| Strict median | 13位 | 13位 |
| 2 m R@1 | 20.51% | 20.51% |
| 2 m R@5 | 42.60% | 42.60% |
| 2 m R@10 | 56.80% | 56.80% |
| 2 m median | 7位 | 7位 |

この一致により、少なくともQuery、DB、checkpoint、前処理、評価ラベル、ランキング評価の主要条件は再現できている。

## 9. A/B/C Retrieval結果

### Strict評価

| 地図 | R@1 | R@5 | R@10 | R@30 | R@50 | 中央順位 |
|---|---:|---:|---:|---:|---:|---:|
| A: 実測16層 | 5.72% | 24.65% | 42.01% | 70.41% | 84.22% | 14位 |
| B: TULIP32層 | 4.93% | 17.95% | 28.01% | 58.38% | 73.37% | 23位 |
| C: 実測32層 | 5.92% | 24.26% | 42.41% | 71.01% | 88.36% | 13位 |

### 2 m許容評価

| 地図 | R@1 | R@5 | R@10 | 中央順位 |
|---|---:|---:|---:|---:|
| A: 実測16層 | 19.72% | 44.38% | 59.76% | 7位 |
| B: TULIP32層 | 14.40% | 29.98% | 45.96% | 12位 |
| C: 実測32層 | 20.51% | 42.60% | 56.80% | 7位 |

### 10 m許容評価

| 地図 | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| A | 54.83% | 68.24% | 79.49% |
| B | 40.43% | 56.61% | 64.69% |
| C | 51.87% | 65.09% | 76.92% |

## 10. A/B/Cの比較解釈

### AとC

Aは16層で点数が少ないが、道路、建物境界、白線などの大きな構造は残る。そのため、Retrievalの大域特徴ではCに近い結果になった。

AとCの厳密なペア比較では、勝ち負けがほぼ均衡している。AのStrict R@5がCよりわずかに高いが、507 Query中の数件差であり、16層が32層より本質的に優れるという意味ではない。

### BとC

Bは点の位置・coverageはCに近づいたが、生成点の反射強度がCから離れた。

画像比較のサンプル値:

| 比較 | Cとのcoverage IoU | Cとの非ゼロ領域内MAE |
|---|---:|---:|
| A | 約86.0% | 約31.6/255 |
| B | 約90.3% | 約43.2/255 |
| 再生成C | 約99.83% | 約3.53/255 |

つまりBは「点がある場所」は改善したが、「その点の明るさ」は悪化した。Retrievalモデルは輪郭だけでなく、画像全体の濃淡分布やテクスチャも使うため、画素数が増えただけでは性能が上がらない。

TULIPで生成した反射強度の誤差が、percentile正規化、ヒストグラム平坦化、CLAHEで強調され、学習時の実測強度分布から離れた可能性が高い。

## 11. 発生したエラーと扱い

### TensorFlowのExecutorメッセージ

次のようなメッセージが出た。

`INVALID_ARGUMENT: You must feed a value for placeholder tensor ...`

これは読み込み時のDEBUG出力であり、実際の処理は継続した。A/B/C地図の生成数とフレーム失敗数を確認し、全フレーム失敗0であることを確認した。

### CUDA/NUMA/TensorRT警告

GPUライブラリ、NUMA、TensorRTに関する警告が出た。RetrievalはCPUまたは利用可能なCUDA構成で実行でき、結果生成を停止するエラーではなかった。

### 清水氏レンダラの直接実行

清水氏側のレンダラを、今回作成した新しいアンカー形式に直接与えたとき、仮想アンカー姿勢リスト形式の不一致で画像書き出し前に停止した。そのため、清水氏側コードは変更せず、必要なメタデータを清水氏の投影条件に合わせたTULIP側のラッパーでA/Bを生成した。

### modality stats JSON

元のstats JSONには、シェルコマンド文字列が混入していた。清水氏の値を記したクリーンなファイルをTULIP側に追加した。

## 12. 現在残っている課題

1. TULIP追加ringの反射強度が実測32層と異なる原因を切り分ける。
2. TULIP点の位置・奥行きは維持し、追加ringの強度だけを隣接実測ringから補間した対照版を作る。
3. その対照版を507 Query／754 DBでA/B/Cと同じ条件でRetrievalする。
4. TULIP強度の分布、生成点と実測点の平均・分散・percentileを比較する。
5. 10 m cropを使う場合は、清水氏条件とは別実験としてC/A/Bすべてで同じcropを適用する。
6. 距離許容を1 m、2 m、5 m、10 mで統一的に報告する。
7. 粗探索の品質を確認してから、Top-5候補をLoFTRへ渡す。
8. LoFTR＋PnPを10～20 Queryで動作確認し、その後507 Query全体へ拡張する。
9. A/B/CのPnP位置誤差を比較する。

## 13. 次に実施する推奨手順

### 第1段階: 粗探索の原因切り分け

追加ring強度を実測ringから補間したB補正版を作る。点の位置、姿勢、投影、画像処理、Query、DB数は現行Bと完全に同じにする。

比較する値:

- coverage IoU
- 非ゼロ領域内MAE
- 強度分布の平均、標準偏差、percentile
- Strict R@1/R@5/R@10
- 2 m R@1/R@5/R@10
- 10 m R@1/R@5/R@10

### 第2段階: 密探索

粗探索のTop-5候補をmanifest化し、カメラ画像と反射強度画像の対応を作る。その後、まず10～20 QueryでLoFTRとPnPを動かし、対応点数、インライア率、位置誤差を確認する。

### 第3段階: 全Queryと複数シーン

507 Query全体で実行し、A/B/Cを同じ条件で比較する。その後、別走行・複数シーンへ拡張する。

## 14. 現時点の結論

- 清水氏のRetrieval条件はCで再現できている。
- Aは実測16層だが、主要な大域構造を保持するためCと近い性能になった。
- Bは点のcoverageを改善したが、TULIP追加点の反射強度差によりRetrieval性能が低下した。
- 点を追加するだけでは、学習済みRetrieval特徴との一致は保証されない。
- 現時点では、密探索へ進む前にBの強度補間版を評価するのが最も情報量の多い作業である。
- その対照実験後、条件を固定したままLoFTR＋PnPへ進む。

## 15. 端末復旧後の再開手順

最初に作業ディレクトリを移動する。

```bash
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
```

次に、既存変更を確認する。ここでは削除、リセット、上書きを行わない。

```bash
git status --short
sed -n '1,260p' HANDOVER_SHIMIZU_CONDITION_REPRODUCTION.md
```

既存成果物の個数を確認する。

```bash
find outputs/shimizu_protocol_507q_754db_abc/anchors_A -name anchor.png | wc -l
find outputs/shimizu_protocol_507q_754db_abc/anchors_B -name anchor.png | wc -l
find outputs/shimizu_protocol_507q_754db_abc/retrieval_manifests -type f | sort
```

期待値はA画像754枚、B画像754枚である。既存画像がある場合は、同じ出力先へ再生成しない。検証版は必ず新しい名前のフォルダへ出力する。

## 16. 入力から結果までのデータ系統

```text
Waymo TFRecord
  ↓
清水氏の走行ペア・ICP対応・Query/DB選択
  ↓
TULIP側のA/B/C map_static.npz
  ↓
清水氏と同じ投影条件でanchor.png生成
  ↓
507 Query / 754 DB manifest
  ↓
epoch 16 checkpointでRetrieval
  ↓
R@1/R@5/R@10、順位、距離許容評価
  ↓
Top-5候補をLoFTRへ入力
  ↓
PnPで最終姿勢推定
```

Cだけは清水氏の既存DB画像を基準に使った。これにより、Cの結果が地図再生成の差ではなく、評価コードと設定の再現になっているかを確認できた。再生成Cの画像比較も別途実施し、coverage IoU約99.83%、union MAE約3.53/255だった。

## 17. 数値を読むときの注意

R@1は正解アンカーが1位になったQueryの割合、R@5とR@10は正解が上位5件、10件に入った割合である。Strict評価は正解アンカーIDを完全一致させるため、隣接する正しい場所が2位でも失敗になる。

2 m許容評価は、候補位置が正解位置から2 m以内なら正解とする。したがって、Strict値と2 m値は用途が異なる。粗探索からLoFTRへ候補を渡す場合は、R@1よりR@5と距離許容R@5を重視する。

coverage IoUは点の有無の一致、MAEは強度値の一致である。coverageが高くてもMAEが大きければ、モデル入力の濃淡パターンは一致しない。今回のBはこの状態に該当する。

## 18. Bの追加ringを調べるときの実験条件

新しい検証版は、次の条件を変えない。

- Query 507枚
- DB 754枚
- アンカー姿勢
- ICP変換
- FRONT投影
- point size 2
- percentile 99.5
- histogram equalization
- CLAHE
- checkpoint
- 評価コード

変更するのは、TULIPが生成した追加ringのintensity値だけにする。追加ringのXYZや深度を変更すると、強度の影響と点位置の影響を同時に変えてしまうため、原因切り分けにならない。

比較対象は少なくとも次の4つにする。

| 版 | XYZ | 追加ring intensity |
|---|---|---|
| A | 実測16層 | 実測値のみ |
| B | TULIP生成 | TULIP推定値 |
| B-int補間 | TULIP生成 | 隣接実測ringから補間 |
| C | 実測32層 | 実測値 |

## 19. 75 m超の点・外れ値の扱い

地図生成や投影時に、センサ仕様や清水氏条件で想定していない遠距離点を黙って混ぜない。75 m超の点が存在する場合は、まず件数、距離分布、画像内画素数を記録する。

確認項目:

- XYZから計算した水平距離
- 75 m超の点数と割合
- A/B/Cごとの点数差
- 75 m超の点を除いた場合のcoverage差
- 75 m超の点を除いた場合のRetrieval差

外れ値除去を行う場合も、A/B/Cの3地図すべてへ同じ条件を適用し、出力先を別フォルダにする。Bだけに適用すると比較条件が崩れる。

## 20. まだ確認できていない事項

以下は、引き継ぎ時点で未完了である。

1. 追加ring intensityを隣接実測ringから補間したB-int補間版の生成。
2. B-int補間版の507 Query／754 DB Retrieval。
3. 生成点と実測点を分けた強度分布統計。
4. 75 m超点の全走行集計。
5. A/B/CのTop-5候補を使うLoFTRの10～20 Query動作確認。
6. 507 Query全体のLoFTR＋PnP。
7. 複数シーン・別走行での汎化確認。

## 21. 作業を再開するときの判断基準

- 端末が起動しない場合は、ファイルの破損と判断せず、実行環境の問題として扱う。
- 清水氏側コードを修正して解決しようとしない。
- 既存A/B/C画像やRetrieval結果を上書きしない。
- B-int補間版の結果がCまたはAへ近づくかを最初の品質判定にする。
- B-int補間版でも低い場合は、intensityだけでなくTULIPの深度・投影位置・モデル分布差を調べる。
- 粗探索の候補をLoFTRへ渡す際は、A/B/Cで同じTop-K、同じQuery、同じPnP設定を使う。

