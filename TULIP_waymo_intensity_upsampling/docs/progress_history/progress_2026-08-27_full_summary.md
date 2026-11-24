# TULIP Waymo 16→32 これまでの実施内容と現状

作成日: 2026-08-27

## 1. 目的

KITTI向けTULIPの構造を維持したまま、Waymo Open DatasetのTOP LiDARについて、16ライン入力から32ラインのrange/intensity画像を復元する流れを構築・検証してきた。

最終的な目的は、TULIPで復元した反射強度画像を可視光カメラ画像と対応付けることである。距離疑似カラー画像は確認用であり、主なマッチング対象はintensity画像とする。

## 2. 守っている構成

- TULIP本体のTransformer構造は変更していない。
- Encoder、Decoder、Attention、Patch Merging、Patch Unmerging、skip connectionは変更していない。
- Waymo向け処理はTULIP_waymo_intensity_upsampling内に分離している。
- TULIPの既存KITTIコード、TULIP_backup、handover_shimizu、/mnt/w/32lineは参照対象として保持している。
- Waymoの入力幅2650は2688へcircular paddingし、出力後に2650へcropする構成である。

## 3. Phase 1: Waymo Range Imageの確認

### 実施内容

Waymo TFRecordからTOP LiDARの第1returnを読み込み、native Range Imageを確認した。

- TFRecord: segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord
- native shape: [64, 2650, 4]
- return index: 0
- channel 0: range
- channel 1: intensity
- range/intensity dtype: float32
- mask dtype: bool

native 64ラインから、次の規則でデータを作成した。

```text
range_32     = range_64[0::2]
intensity_32 = intensity_64[0::2]
range_16     = range_32[0::2]
intensity_16 = intensity_32[0::2]
```

- ring_ids_32 = [0,2,4,...,62]
- ring_ids_16 = [0,4,8,...,60]
- 無効条件: rangeが0以下、またはrange/intensityが非有限
- 無効画素のrange/intensityは0

### 検証結果

- range_64: [64,2650]
- range_32: [32,2650]
- range_16: [16,2650]
- NaN/Inf: 0
- 行選択の一致: PASS
- NPZ、metadata JSON、確認PNGを生成
- Phase 1 pytest: 4 passed

## 4. Phase 2: intensity統計

複数TFRecord・複数フレームについて、32ラインGTのvalid画素を逐次集計した。

実施した内容:

- min/max/count/sum/sum of squaresの逐次集計
- 固定seedによるフレーム単位のランダムサンプリング
- rangeとintensityで同じvalid画素位置を使用
- 失敗TFRecord・失敗フレームの記録
- maximum_range、intensity_scaleはコードへ固定していない

intensityは大半が低い値だが、1000以上の極端な値が少数存在することを確認した。

このため、raw MAEだけでは外れ値が全体値を支配することが分かった。以降はraw domainだけでなく、log1p domain、強度帯別MAE、1000以上の再現率を併用して評価している。

## 5. Phase 3: raw 32ラインGT出力

Waymo TFRecordから、学習前のraw 32ラインGTを.npyとして保存する処理を実装した。

出力仕様:

- shape: [32,2650,2]
- dtype: float32
- array[...,0]: raw range
- array[...,1]: raw intensity
- 正規化なし
- clip/log1p/gamma補正なし
- 2688へのpaddingなし
- 無効画素は両チャンネル0

sidecar JSON、manifest、failures JSONも出力した。5フレーム確認後、同一TFRecordの全フレームを出力し、240フレームの学習・validation・test用indexを作成した。

## 6. Datasetと入力変換

生成したsplitは次のとおりである。

- train: 160フレーム
- validation: 40フレーム
- test: 40フレーム
- 合計: 240フレーム

Datasetの確認結果:

- 1サンプルのinput: [2,16,2688]
- 1サンプルのtarget: [2,32,2688]
- batch input: [B,2,16,2688]
- batch target: [B,2,32,2688]
- dtype: torch.float32
- range: range / 75.0
- intensity: rawまたはlog1pを選択可能
- 2650→2688: 水平方向circular padding
- 出力評価時: 2688→2650 crop

Dataset pytest:

```text
4 passed
```

## 7. TULIP forward確認

Waymo用モデルは、既存TULIPのモデル構造を再利用している。

forward確認済みのshape:

```text
input:  [1,2,16,2688]
output: [1,2,32,2688]
crop:   [1,2,32,2650]
target: [1,2,32,2688]
```

loss計算も有限値で実行できた。

Transformer本体を変更せず、Waymo用のpatch設定と最終アップサンプリング条件で16→32を通過できることを確認した。

## 8. 学習実験

### overfit確認

1サンプル・20 epochでlossが、

```text
5.6853 → 4.0874
```

となり、学習が進むことを確認した。

### 5フレーム学習

5サンプル・5 epoch:

```text
8.8670 → 7.5435
```

### 20フレーム学習

20サンプル・10 epoch:

```text
9.5221 → 7.1971
```

### 基準モデル e100

現在の主な基準モデル:

```text
outputs/waymo_16_to_32_fixed_e100/checkpoint.pth
```

設定:

- train: 160フレーム
- epoch: 100
- range scale: 75.0 m
- intensity transform: log1p
- range weight: 0.05
- intensity weight: 1.0
- model: TULIP 2ch
- Transformer構造: 変更なし

学習loss:

```text
0.06898 → 0.01379
```

## 9. 画像上の評価

e100モデルのtest 40フレームでは、raw値へ戻した評価で次の結果を得た。

```text
range MAE:     0.7688 m
intensity MAE: 13.0783
```

validationでは、

```text
range MAE:     0.6404 m
intensity MAE:  6.6166
```

nearest・linear補間と比較して、rangeはTULIPが改善した。

## 10. intensity帯別評価

e100 test 40フレーム、raw intensity domainの結果:

| 強度帯 | 割合 | TULIP MAE | nearest MAE | linear MAE |
|---|---:|---:|---:|---:|
| 0–0.5 | 95.17% | 0.101 | 1.100 | 1.506 |
| 0.5–1 | 3.77% | 0.522 | 3.758 | 5.728 |
| 1–2 | 0.38% | 1.214 | 16.093 | 24.555 |
| 2–5 | 0.20% | 2.987 | 78.440 | 73.707 |
| 5–20 | 0.16% | 9.335 | 33.791 | 84.018 |
| 20–100 | 0.11% | 46.854 | 211.165 | 311.334 |
| 100–1000 | 0.10% | 386.471 | 417.032 | 437.600 |
| 1000以上 | 0.11% | 12105.420 | 6715.288 | 7468.497 |

結論:

- 0〜1000ではTULIPがほぼすべての帯域で補間を上回る。
- 1000以上ではTULIPが補間に負ける。
- 1000以上の外れ値は、現在のモデルでは十分に再現できていない。

## 11. tail-weighted実験

raw domainで、target intensityが1000を超える画素へ追加重みを付けた。

試した条件:

- tail threshold: 1000.0
- tail weight: 2、5、10
- train subset: 20フレーム
- epoch: 20
- loss rows: generated

weight 5のvalidation結果:

```text
1000以上 TULIP MAE: 16602.857
target平均:         16603.171
```

weight 10でも、1000以上の改善は確認できなかった。

この実験から、単純なraw-domain tail weightだけでは極端値の復元に不十分であることが分かった。なお、raw tail-weightedモデルとlog1p e100モデルは学習domainが異なるため、直接比較してはいけない。

## 12. 点群復元評価

e100モデルをWaymo vehicle frameの点群へ復元し、test 40フレームで評価した。

| 指標 | TULIP | nearest | linear |
|---|---:|---:|---:|
| coordinate MAE [m] | 0.6075 | 0.6342 | 0.9873 |
| Euclidean MAE [m] | 1.3725 | 1.4127 | 2.2385 |
| intensity MAE | 17.8234 | 19.1956 | 18.3393 |

TULIPはnearestより、

- coordinate MAE: 約4.2%改善
- Euclidean MAE: 約2.8%改善
- intensity MAE: 約7.1%改善

した。

これはrange image上だけでなく、3次元点群位置でもTULIPの復元が有効であることを示す。

## 13. カメラ投影評価

e100モデルについて、Waymoの5カメラへの投影を確認した。

評価したもの:

- GT range + GT intensity
- GT range + TULIP intensity
- TULIP range + TULIP intensity

GT rangeを固定したA-B比較はintensity復元のみを評価し、B-C比較はrange誤差による投影位置変化を評価する。

log1p domainのA-B MAE:

| Camera | log1p MAE | Pearson相関 |
|---|---:|---:|
| FRONT | 0.0965 | 0.6060 |
| FRONT_LEFT | 0.1107 | 0.7363 |
| FRONT_RIGHT | 0.0704 | 0.7879 |
| SIDE_LEFT | 0.0779 | 0.6358 |
| SIDE_RIGHT | 0.0577 | 0.6175 |

GT rangeを使ったため、ここではrange geometryの誤差を分離している。

## 14. 可視光RGBとの対応確認

投影位置でintensityとRGB画素を直接比較した。

log1p intensityとRGB輝度の相関は、

- FRONT: -0.040
- FRONT_LEFT: 0.051
- FRONT_RIGHT: 0.086
- SIDE_LEFT: 0.275
- SIDE_RIGHT: 0.187

であり、強い相関は確認できなかった。

これは直ちに投影失敗を意味しない。LiDAR intensityとRGB輝度は、照明、材質、色、露出、反射特性が異なる物理量だからである。

ORBによる特徴点対応は探索的に実施したが、幾何学的な正解対応ラベルを使用していない。そのため、マッチ数はLoFTRやPnPの性能指標として扱っていない。

## 15. 実行環境の問題

環境は完全には統合されていない。

### .venv_tulip

- TULIP/PyTorch forward・学習に適している。
- TensorFlowがなく、TFRecordを直接読む点群評価では不足する場合がある。

### .venv_waymo

- TensorFlow/Waymo TFRecord処理に適している。
- timm、chamfer_distanceが不足する場合がある。

一部の検証では、実行時stubを使用した。したがって、stub-assisted evaluationと正式依存関係による最終評価は区別する必要がある。

TensorFlowのGPUライブラリ警告が出た場合、TFRecord処理はCPUへフォールバックするが、処理が完走していれば結果自体は生成できる。PyTorch側は--device cudaで実行できているかを別途確認する必要がある。

## 16. 現在できていること

- Waymo TFRecordからTOP LiDARを読み込める。
- native 64ラインから32ラインGTを生成できる。
- 32ラインから16ライン入力を生成できる。
- 2ch range/intensity Datasetを作成できる。
- [1,2,16,2688]から[1,2,32,2688]へforwardできる。
- 16→32のTULIP学習ができる。
- range/intensityの画像評価ができる。
- intensity帯別評価ができる。
- 3次元点群評価ができる。
- 5カメラへの投影評価ができる。
- RGBとの単純な画素対応を探索できる。

## 17. 未解決課題

1. intensity 1000以上の再現率が0%に近い。
2. raw MAEは極端値に支配される。
3. log1p tail-weighted学習はまだ正式比較していない。
4. 点群評価の手法間で有効点数が異なるため、共通valid mask比較が必要である。
5. timmとchamfer_distanceを含む正式環境での最終forward確認が必要である。
6. 240フレームはWaymo全体ではなく、現段階の検証用subsetである。
7. RGBとの対応は単純相関では不十分で、LoFTR/PnPなどの幾何評価が必要である。
8. 複数seedによる再現性確認はまだ行っていない。

## 18. 次に行う推奨順序

### Step 1: e100のgenerated-only評価

観測済みの16ラインをloss・評価へ混ぜていないことを確認する。

### Step 2: e100の点群評価結果を基準として保存

range geometry、Euclidean誤差、intensity誤差を基準値として固定する。

### Step 3: log1p条件のtail-aware比較

raw tail-weightedではなく、e100と同じrange scale 75・log1p条件で、target raw intensity >1000の画素へ重みを付ける。Transformer本体は変更しない。

### Step 4: 5カメラで再評価

投影率、pixel shift、A-B intensity誤差、RGB対応を比較する。

### Step 5: 正式依存関係で再実行

stubを使わず、TULIPの実依存関係を整えた環境でforward、点群、カメラ評価を再確認する。

## 19. 現時点の総合判定

### TULIPのWaymo 16→32適用

小規模技術検証としてはPASS。

### TULIP論文の中心部分

range imageを入力し、Transformerで高解像度range imageを復元し、点群へ戻す流れは再現できている。

### intensity復元

通常強度0〜1000では有効だが、1000以上の外れ値復元は未解決。

### 最終的な可視光RGBとのマッチング

投影処理の準備段階まで進んでいる。RGBとの特徴対応性能を主張するには、幾何学的な正解対応を用いた追加評価が必要である。

作成日: 2026-08-27

