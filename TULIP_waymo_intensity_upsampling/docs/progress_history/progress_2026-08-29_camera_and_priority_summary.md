# Waymo TULIP 16→32 本日の作業記録

作成日: 2026-08-29

## 対象

Waymo 16→32 TULIPモデルについて、生成行評価、点群復元、複数カメラ投影、GT・Low・Pred画像、FRONT_RIGHTの原因分析を実施した。

## 共通条件

- checkpoint: outputs/waymo_16_to_32_full_e100/checkpoint.pth
- checkpoint epoch: 99
- index: outputs/training_pairs_16_to_32_full/index.csv
- range scale: 75.0 m
- intensity transform: log1p
- device: CUDA
- validation: 40フレーム
- test: 40フレーム
- 評価対象: 32ライン上のTULIP生成奇数行

## 生成奇数行の評価

| split | 有効画素数 | range MAE | intensity MAE |
|---|---:|---:|---:|
| validation | 1,537,686 | 1.1071 m | 6.2651 |
| test | 1,491,970 | 1.3284 m | 17.8708 |

target・predictionのNaN/Infは0件だった。testではintensityが1000を超えるtarget画素が1,768個、約0.1185%存在し、最大値は372,736.06だった。

## 点群復元

validation 40フレーム:

- TULIP coordinate MAE: 0.4844 m
- TULIP Euclidean MAE: 1.1070 m
- TULIP intensity MAE: 6.2638
- nearest Euclidean MAE: 1.3373 m
- linear Euclidean MAE: 1.8544 m

TULIPの点群幾何誤差は、最近傍・線形補間より小さかった。

## 複数カメラ投影

### validation

| camera | oracle points | predicted points | mean shift | p95 shift |
|---|---:|---:|---:|---:|
| FRONT | 197,082 | 197,066 | 0.285 px | 0.828 px |
| FRONT_LEFT | 218,917 | 218,918 | 0.375 px | 1.334 px |
| FRONT_RIGHT | 219,978 | 219,796 | 0.964 px | 3.669 px |

### test

| camera | oracle points | predicted points | mean shift | p95 shift |
|---|---:|---:|---:|---:|
| FRONT_LEFT | 207,394 | 207,394 | 0.787 px | 3.143 px |
| FRONT_RIGHT | 219,820 | 219,789 | 0.922 px | 3.528 px |

FRONT_RIGHTはvalidation・testの両方でFRONTよりpixel shiftが大きかった。一方、oracleとpredictedの投影点数・投影率はほぼ一致しており、投影処理の破綻は確認されなかった。

## GT・Low・Pred画像

validation sample 0について、FRONT、FRONT_LEFT、FRONT_RIGHTの画像を生成した。

出力先:

outputs/waymo_16_to_32_full_e100/guide_validation_sample0/

Lowは32グリッドの偶数行に16ライン観測値を配置した。Pred fusedは偶数行をLow、奇数行をTULIP予測値とした。

| camera | GT points | Low points | Pred fused points |
|---|---:|---:|---:|
| FRONT | 11,398 | 5,786 | 11,399 |
| FRONT_LEFT | 11,191 | 5,655 | 11,189 |
| FRONT_RIGHT | 10,829 | 5,514 | 10,822 |

Pred fusedの点数はGTとほぼ同数であり、16ラインから32グリッドへの点密度復元は確認できた。

## FRONT_RIGHTの距離・強度帯分析

### validation

- common projected points: 219,752
- pixel shift mean: 0.9766 px
- range absolute error: 1.3104 m
- intensity MAE: 4.7155
- 30〜50 m帯のintensity MAE: 11.643
- intensity 100〜1000のMAE: 320.452
- intensity 1000以上のMAE: 7,031.749

### test

- common projected points: 219,668
- pixel shift mean: 0.9192 px
- range absolute error: 1.0449 m
- intensity MAE: 21.5424
- 10〜20 m帯のintensity MAE: 46.524
- intensity 100〜1000のMAE: 335.046
- intensity 1000以上のMAE: 14,871.530

FRONT_RIGHTの主要課題は、カメラ投影数の不足ではなく、距離帯によるrange誤差と、少数の高intensity外れ値によるraw MAEの増大である。validation・testの両方で同じ傾向が確認された。

## 保存した診断結果

- outputs/waymo_16_to_32_full_e100/priority_generated_sanity.json
- outputs/waymo_16_to_32_full_e100/front_right_validation_distance_intensity.json
- outputs/waymo_16_to_32_full_e100/front_right_test_distance_intensity.json
- outputs/waymo_16_to_32_full_e100/camera_validation_front_40/
- outputs/waymo_16_to_32_full_e100/camera_test_front_left_40/
- outputs/waymo_16_to_32_full_e100/camera_test_front_right_40/
- outputs/waymo_16_to_32_full_e100/guide_validation_sample0/

## 発生した実行上の問題

一時的なインライン解析で、2688幅から2650幅へのcrop指定ミスと投影配列のアンパックミスが発生した。どちらも正式コードには影響せず、修正後に解析を完了した。結果を残さない失敗出力は確認後に残っていない。一時スクリプトファイルは作成していない。

## コード変更範囲

- TULIP model source: 変更なし
- Transformer、Encoder、Decoder、Attention: 変更なし
- Dataset、既存評価スクリプト: 変更なし
- 正式な新規コード: なし
- 本日追加したMarkdown: 本ファイル

## 本日の結論

Waymo 16→32 TULIPは、forward、range復元、点群復元、複数カメラ投影まで正常に動作している。FRONT_RIGHTでも投影処理は成立しているが、高intensity外れ値と一部距離帯の誤差が残っているため、全Waymoデータへの本格適用はまだ保留する。

## 次の作業

1. intensity通常域と外れ値を分離した主評価指標を確定する。
2. FRONT_RIGHTのtest結果を基準値として保存する。
3. 必要なら既存モデル構造を変更せず、正規化・loss重み条件を比較する。
4. 複数seedまたは追加フレームで再現性を確認する。
5. 条件が安定した後に全Waymoデータへ適用する。
## 複数フレームのGT・Low・Pred比較画像（2026-08-29）

検証用にvalidationの代表5フレーム（sample index 0, 10, 20, 30, 39）について、FRONT、FRONT_LEFT、FRONT_RIGHTの各カメラでRGB画像とGT・Low・Predの反射強度投影画像、overlay画像を生成した。保存先は outputs/waymo_16_to_32_full_e100/guide_validation_samples_5/ である。PNGは確認用であり、評価値や学習データの代替には使用していない。

投影点数（GT / Low / Pred）は次のとおりで、全15ケースでPredはGTとほぼ同数、Lowはおおむね半数となった。これは16ライン入力が32ライン相当へ疎になること、またPredの生成行をGTと比較できることを視覚確認するための結果である。

| sample | FRONT | FRONT_LEFT | FRONT_RIGHT |
|---:|---:|---:|---:|
| 0 | 11398 / 5786 / 11399 | 11191 / 5655 / 11189 | 10829 / 5514 / 10822 |
| 10 | 9560 / 4788 / 9559 | 11078 / 5640 / 11078 | 11543 / 5865 / 11537 |
| 20 | 9455 / 4695 / 9455 | 11129 / 5561 / 11128 | 11082 / 5603 / 11080 |
| 30 | 9386 / 4627 / 9389 | 11142 / 5581 / 11142 | 10999 / 5569 / 10998 |
| 39 | 9517 / 4787 / 9517 | 11389 / 5740 / 11386 | 11210 / 5607 / 11207 |

生成されたPNGは90枚で、全ファイルを読み込み検査した。RGB画像45枚は [1280, 1920, 3]、投影・overlay画像45枚は [1280, 1920] の8-bit画像で、読み込みエラーは0件だった。

画像は、(1) RGB上でLiDARの視野・位置が合っているか、(2) Lowで欠けるリングがPredで補われるか、(3) Predの強度分布がGTに近いか、の順で確認する。疑似カラーは表示用のパーセンタイル正規化であり、数値比較は projection_raw.npz とJSON統計を使用する。

今回の追加確認では、FRONT_RIGHTで既に確認されているp95画素シフトおよび高強度外れ値の課題は解決したとは判定しない。今後はこの5フレーム画像を入口に、強度帯・距離帯別の数値評価と併せて、学習条件を固定した比較を行う。

**コード変更:** なし。追加されたのは検証出力と本記録のみで、TULIP本体・Dataset・既存前処理は変更していない。
## test側3カメラ出力の監査（2026-08-29）

既存のtest 40フレーム出力について、3カメラのsummary.json、per-frame出力、PNGを監査した。各カメラでPNGは200枚（RGB 120枚、投影・overlay 80枚）、全画像の読み込みエラーは0件で、画像サイズはすべて [1280, 1920] または [1280, 1920, 3] だった。

| camera | common LiDAR pixels | oracle points | predicted points | projection rate (oracle / predicted) | pixel shift mean / p95 |
|---|---:|---:|---:|---:|---:|
| FRONT | 1,491,962 | 184,239 | 184,230 | 0.123494 / 0.123488 | 0.6732 / 2.5000 |
| FRONT_LEFT | 1,491,962 | 207,394 | 207,394 | 0.139005 / 0.139005 | 0.7872 / 3.1432 |
| FRONT_RIGHT | 1,491,962 | 219,820 | 219,789 | 0.147324 / 0.147303 | 0.9221 / 3.5280 |

test側でもPredの投影点数と投影率はGT（oracle）とほぼ一致した。一方、FRONT_RIGHTの画素シフトは3カメラ中で最大であり、既存の課題として残る。これは投影処理の完走確認であり、カメラ画像上の特徴マッチング性能を証明するものではない。

**今回の判断:** 画像生成・投影経路は3カメラで実行可能。次の学習判断では、FRONT_RIGHTのずれと高強度intensity帯を含む評価を必須とする。TULIP本体および既存Datasetの変更は行っていない。
## 全240学習ペアの最終監査（2026-08-29）

本格学習前の安全確認として、training 160、validation 40、test 40の合計240ペアを読み取り監査した。全件でNPZが存在し、入力は [16,2650,2]、targetは [32,2650,2]、maskはそれぞれ [16,2650] / [32,2650]、dtypeはデータが float32、maskが bool であった。NaN/Infはなく、ring IDも入力 0,4,...,60、target 0,2,...,62 と一致した。

現在の学習ペアはraw zero-fillではなく、mask付き学習用テンソルとして無効画素を両チャンネル -1 sentinelで保持している。監査ではmask=Trueの画素に -1 がなく、mask=Falseの画素は両チャンネルとも -1 だった。無効画素数はinput合計1,293,312、target合計2,586,882である。

**結果:** 240/240件 PASS、欠損0件、shape/dtype/有限性/mask/ring IDの不一致0件。途中の試行でDataset返却名をNPZキーとして参照したこと、およびring IDを0..31と比較したことによる一時的な誤判定は、既存データを変更せず、正しいNPZ仕様で再監査して解消した。
## geometry分離評価（2026-08-29）

カメラ投影のgeometryとintensity信号を分離して整理した。既存のcamera summary.jsonを再利用し、追加推論なしで次の3条件を定義した。

| 条件 | geometry | signal | 投影への意味 |
|---|---|---|---|
| GT range + GT intensity | GT range | GT intensity | 基準。GT rangeと同一の投影座標 |
| GT range + Pred intensity | GT range | TULIP intensity | intensity復元だけを評価する条件。GT range + GT intensityと投影座標は同じ |
| Pred range + Pred intensity | TULIP range | TULIP intensity | 実運用条件。GTとの差がrange復元による画素ずれを含む |

validation/testのFRONT、FRONT_LEFT、FRONT_RIGHTについて、GT rangeを使う2条件の投影点数・座標は一致し、両者の相対画素ずれは0となる。Pred range + Pred intensityでは、既存評価のoracle-vs-predicted shiftがrange geometryの影響を表す。

| split | camera | GT-range points | Pred-range points | pixel shift mean / p95 |
|---|---|---:|---:|---:|
| validation | FRONT | 197,082 | 197,066 | 0.2845 / 0.8283 |
| validation | FRONT_LEFT | 218,917 | 218,918 | 0.3750 / 1.3337 |
| validation | FRONT_RIGHT | 219,978 | 219,796 | 0.9637 / 3.6687 |
| test | FRONT | 184,239 | 184,230 | 0.6732 / 2.5000 |
| test | FRONT_LEFT | 207,394 | 207,394 | 0.7872 / 3.1432 |
| test | FRONT_RIGHT | 219,820 | 219,789 | 0.9221 / 3.5280 |

**結論:** 投影処理そのものがintensity値によってずれている証拠はなく、現在観測されている画素ずれは主にTULIP rangeをPred geometryへ使った場合に発生する。GT range + Pred intensityでのintensity精度はgeometry誤差を含めずに別評価すべきである。

追加推論を行う一時スクリプトは長時間化したため停止したが、既存の検証済みcamera出力から導出したレポートを保存した。TULIP本体、Dataset、既存評価コードの変更はない。
