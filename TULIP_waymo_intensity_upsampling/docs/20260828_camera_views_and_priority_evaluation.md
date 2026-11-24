# Waymo TULIP 16→32 優先評価記録

作成日: 2026-08-28

## 目的

TULIP本体を変更せず、Waymo 16→32モデルの生成行、点群復元、複数カメラ投影を確認する。論文指標との完全一致は今回の対象外とした。

## 共通条件

- checkpoint: outputs/waymo_16_to_32_full_e100/checkpoint.pth
- checkpoint epoch: 99
- dataset: outputs/training_pairs_16_to_32_full/index.csv
- range scale: 75.0 m
- intensity transform: log1p
- device: CUDA
- 評価対象: validation 40フレーム
- 生成行評価: 32ライン上の奇数行のみ

## 生成奇数行のraw評価

| split | 有効画素数 | 有効率 | range MAE | intensity MAE |
|---|---:|---:|---:|---:|
| validation | 1,537,686 | 90.665% | 1.1071 m | 6.2651 |
| test | 1,491,970 | 87.970% | 1.3284 m | 17.8708 |

NaN/Infはtarget・predictionとも0件だった。test生成行ではintensityが1000を超えるtarget画素が1,768個あり、割合は約0.1185%だった。target intensityの最大値は372,736.06、p99.9は1,504.00であり、外れ値がraw MAEを支配する課題は残っている。

## 点群復元評価

validation 40フレーム:

- TULIP coordinate MAE: 0.4844 m
- TULIP Euclidean MAE: 1.1070 m
- TULIP intensity MAE: 6.2638
- nearest Euclidean MAE: 1.3373 m
- linear Euclidean MAE: 1.8544 m

このvalidation subsetでは、TULIPの点群幾何誤差が最近傍・線形補間より小さかった。

## FRONT系カメラ投影

Waymo vehicle frameからcamera frameへ変換し、カメラ前方深度をX_cとした。既存カメラ評価のoracleはGT range + TULIP intensity、predictedはTULIP range + TULIP intensityである。

| camera | oracle points | predicted points | oracle rate | predicted rate | mean shift | p95 shift |
|---|---:|---:|---:|---:|---:|---:|
| FRONT | 197,082 | 197,066 | 12.814% | 12.813% | 0.285 px | 0.828 px |
| FRONT_LEFT | 218,917 | 218,918 | 14.248% | 14.248% | 0.375 px | 1.334 px |
| FRONT_RIGHT | 219,978 | 219,796 | 14.313% | 14.302% | 0.964 px | 3.669 px |

FRONTとFRONT_LEFTは投影位置ずれが小さい。FRONT_RIGHTでは相対的に大きく、視野・距離分布・遮蔽などのカメラ方向依存性を追加確認する。

## 画像成果物

FRONT validation 40フレーム:

outputs/waymo_16_to_32_full_e100/camera_validation_front_40/

camera RGB、oracle/predicted intensity projection、overlay、projection_raw.npzが生成された。全40フレームの対象PNGを読み込み、エラーは0件だった。

- RGB: [1280,1920,3]
- intensity projection: [1280,1920]
- overlay: [1280,1920,3]

GT・Low・Predの融合表示は、既存の1フレーム成果物 outputs/waymo_16_to_32_full_e100/guide_front_test_sample0/ に保存されている。PNGは確認用で、数値評価はraw配列を使用する。

## 警告

AVX2/FMA、TensorRT未検出、NUMA node、GPU初期化時のNUMA fallback、Placeholder/_0のINVALID_ARGUMENT DEBUG INFOは表示されたが、CUDA GPUが認識され全フレーム処理が完走した。今回の結果に対する致命的エラーではない。

## 今回の変更範囲

- TULIP model source: 変更なし
- Transformer/Encoder/Decoder/Attention: 変更なし
- Dataset・既存評価スクリプト: 変更なし
- 一時スクリプト: 作成なし
- 新規診断出力: outputs/waymo_16_to_32_full_e100/priority_generated_sanity.json

## 次の優先作業

1. FRONT_RIGHTの距離帯・強度帯別誤差を確認する。
2. FRONT/FRONT_LEFT/FRONT_RIGHTのGT・Low・Pred融合画像を複数フレームで比較する。
3. generated-onlyを主指標としてvalidation/test差を記録する。
4. 必要なら複数seedで再現性を確認する。
5. 論文指標との完全一致は最後に実施する。

## FRONT_RIGHT test追加分析

test 40フレームについても同じ解析を実施した。対象は生成奇数行で、GT・予測rangeが有限かつ正、かつ両方がFRONT_RIGHT画像内に投影できる219,668点である。

- pixel shift mean: 0.9192 px
- camera depth absolute error mean: 1.0129 m
- radial range absolute error mean: 1.0449 m
- intensity MAE: 21.5424

距離帯では10〜20 mが43.89%を占め、この帯域のintensity MAEは46.524だった。0〜10 mは2.794、20〜30 mは1.336、30〜50 mは0.390だった。

強度帯では0〜0.5が95.31%でintensity MAE 0.0616だった。一方、100〜1000は87点でMAE 335.046、1000以上は314点でMAE 14,871.530だった。1000以上は約0.143%だが、raw intensity MAEを支配する。

validationとtestの両方で、FRONT_RIGHTの主な課題は次の通りである。

1. 高intensity外れ値の再現不足
2. testの10〜20 m帯でのintensity誤差増大
3. 通常強度帯では誤差が小さい一方、少数tailが平均値を大きく押し上げる

## 全データ適用の判断更新

今回の40フレーム分析では、TULIPのforward、点群復元、投影処理自体に致命的な破綻は確認されなかった。しかし、FRONT_RIGHTでvalidation/testの両方に再現性のあるintensity課題があるため、全Waymo規模の本格学習へ直ちに移行する条件はまだ満たしていない。

次は、強度tailを含む評価指標を固定し、FRONT_RIGHTのtest結果を基準値として保存したうえで、既存モデル構造を変更しない学習条件比較を行う。論文指標との完全一致は最後に実施する。

test診断JSON:

outputs/waymo_16_to_32_full_e100/front_right_test_distance_intensity.json

## test splitの追加カメラ投影

test 40フレームでもFRONT_LEFTとFRONT_RIGHTを評価した。

| camera | common LiDAR pixels | oracle points | predicted points | oracle rate | predicted rate | mean shift | p95 shift |
|---|---:|---:|---:|---:|---:|---:|---:|
| FRONT_LEFT | 1,491,962 | 207,394 | 207,394 | 13.9005% | 13.9005% | 0.787 px | 3.143 px |
| FRONT_RIGHT | 1,491,962 | 219,820 | 219,789 | 14.7324% | 14.7303% | 0.922 px | 3.528 px |

FRONT_RIGHTはvalidationのmean shift 0.964 px、p95 3.669 pxに対し、testでもmean shift 0.922 px、p95 3.528 pxだった。したがって、FRONT_RIGHTの位置ずれはvalidationだけの偶然ではなく、カメラ方向に依存して再現する傾向がある。

FRONT_LEFTはvalidationよりtestでp95が増加しているが、投影点数と投影率はoracleとほぼ一致している。投影処理そのものの失敗ではなく、距離誤差の分布を引き続き確認する。

testカメラ成果物:

- outputs/waymo_16_to_32_full_e100/camera_test_front_left_40/
- outputs/waymo_16_to_32_full_e100/camera_test_front_right_40/

## FRONT_RIGHT追加切り分け

validation 40フレームについて、GT camera-forward depth X_cとGT intensityで分類した。対象は生成奇数行で、GT・予測rangeが有限かつ正で、両方がカメラ画像内に投影できる219,752点である。

aggregate:

- pixel shift mean: 0.9766 px
- depth absolute error mean: 1.2642 m
- radial range absolute error mean: 1.3104 m
- intensity MAE: 4.7155

距離帯別の傾向:

- 0〜20 mが全体の64.4%を占める。
- 0〜10 mはpixel shift 1.651 px、intensity MAE 1.156。
- 10〜20 mはpixel shift 0.933 px、intensity MAE 2.657。
- 20〜30 mはpixel shift 0.719 px、intensity MAE 9.452。
- 30〜50 mはintensity MAE 11.643で、通常域の主な誤差寄与である。
- 50〜75 mは点数が少なく、range誤差は5.276 mだった。

強度帯別の傾向:

- 0〜0.5が89.50%で、intensity MAE 0.0796。
- 0.5〜1はintensity MAE 0.2888。
- 1〜2はintensity MAE 0.8979。
- 2〜5はintensity MAE 2.3876。
- 5〜20はintensity MAE 6.7731。
- 20〜100はintensity MAE 47.7891。
- 100〜1000はintensity MAE 320.4515。
- 1000以上は138点、約0.063%だが、intensity MAE 7,031.75だった。

したがって、FRONT_RIGHTの投影ずれはrangeだけでなく、20〜50 mのrange誤差と高intensity外れ値の影響を受けている。1000以上は非常に少数だがraw intensity MAEを大きく押し上げる。これは直ちにカメラキャリブレーション不良を示すものではない。

診断JSON:

outputs/waymo_16_to_32_full_e100/front_right_validation_distance_intensity.json

## GT・Low・Pred画像

validation sample 0について、FRONT、FRONT_LEFT、FRONT_RIGHTのGT・Low・Predを生成した。

outputs/waymo_16_to_32_full_e100/guide_validation_sample0/

各カメラの投影点数:

| camera | GT | Low | Pred fused |
|---|---:|---:|---:|
| FRONT | 11,398 | 5,786 | 11,399 |
| FRONT_LEFT | 11,191 | 5,655 | 11,189 |
| FRONT_RIGHT | 10,829 | 5,514 | 10,822 |

Lowは観測16ラインを32グリッドの偶数行へ配置したもの、Pred fusedは偶数行をLowの観測値、奇数行をTULIP出力としたもの。Predの投影点数はGTとほぼ同数であり、点密度の復元自体は確認できた。
