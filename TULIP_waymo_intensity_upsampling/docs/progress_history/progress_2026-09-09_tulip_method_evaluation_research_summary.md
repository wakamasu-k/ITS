# TULIPを用いたWaymo 16→32アップサンプリング研究まとめ

作成日：2026-09-09  
作業ルート：`/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling`

## 1. 研究目的

Waymo TOP LiDARの16ライン観測から、32ラインの距離（range）と反射強度
（intensity）を同時に復元する。復元結果をRange Imageだけで評価せず、三次元
点群とカメラ画像へ戻し、将来の画像対応付けや自己位置推定へ接続できる品質かを
確認する。

本研究はTULIP論文の中心概念である「LiDARをRange Imageとして扱い、Swin
Transformerで垂直解像度を高め、点群へ戻す」をWaymoへ適用したものである。
一方、センサ解像度と目的倍率がKITTI実験とは異なるため、完全な同条件再現では
なく、TULIP本体を維持したWaymo向け適用である。

## 2. TULIP原手法の中心

TULIPはLiDAR点群を二次元Range Imageへ変換し、LiDAR向けの異方的なpatchと
windowを持つSwin Transformer encoder-decoderで欠けた走査線を復元する。

主要要素は次の通りである。

- Range Image表現
- 距離と反射強度の2チャンネル入力・出力
- 水平方向の周期性を考慮したcircular padding
- Swin Transformer BlockとWindow Attention
- PatchEmbedding、PatchMerging、encoder、decoder、skip connection
- PatchUnmergingとPixelShuffleによる高解像度化
- 復元Range Imageから三次元点群への逆投影

既存KITTI設定では、入力`[B,2,16,1024]`から出力`[B,2,64,1024]`を作る。
主要設定は`patch_size=(1,4)`、`window_size=(2,8)`、`embed_dim=96`、
`depths=(2,2,2,2)`、`num_heads=(3,6,12,24)`である。

## 3. 現在のWaymo提案手法

### 3.1 全体の流れ

```text
Waymo TFRecord
  ↓ TOP LiDAR・第1 returnを抽出
native Range Image [64,2650,4]
  ↓ channel 0=range、channel 1=intensityを使用
32ラインGT [32,2650,2]
  │ native ring 0,2,4,...,62
  ↓ 32ラインGTの偶数行を選択
16ライン入力 [16,2650,2]
  │ native ring 0,4,8,...,60
  ↓ range/反射強度変換、valid mask適用
range / 75、intensity=log1p
  ↓ 水平方向circular padding
[2,16,2650] → [2,16,2688]
  ↓ Waymo adapter + TULIP Transformer本体
予測 [2,32,2688]
  ↓ 2650列へcrop
予測 [2,32,2650]
  ↓ 偶数行=観測値、奇数行=モデル予測として評価・融合
復元32ラインRange Image
  ↓ beam inclination・方位角・外部パラメータを使用
Waymo vehicle座標の三次元点群
  ↓ カメラ外部・内部パラメータ
FRONTカメラ投影・画像対応評価
```

32ライン上の偶数行は実際に観測された16ラインであり、奇数行がTULIPの生成対象
である。補完性能の主評価は必ず奇数行だけで行う。

### 3.2 TULIPと同じ部分

- Swin Transformer Block
- Window Attention
- encoder / decoder
- PatchMerging
- skip connection
- `embed_dim=96`
- `depths=(2,2,2,2)`
- `num_heads=(3,6,12,24)`
- `window_size=(2,8)`
- `in_chans=2`
- rangeとintensityを同時に予測する考え方

### 3.3 Waymo向けに変更した部分

| 項目 | KITTI実験 | 現在のWaymo 16→32 |
|---|---|---|
| 入出力 | 16→64 | 16→32 |
| native幅 | 1024 | 2650 |
| モデル入力幅 | 1024 | circular padding後2688 |
| patch size | `(1,4)` | `(1,2)` |
| PatchEmbedding | KITTI既存 | kernel `(1,8)`、stride `(1,2)`、左右3列circular padding |
| 最終拡大 | 既存PixelShuffle系 | adapterの縦2倍・横2倍展開 |
| range scale | 80 m基準 | 75 mを実験条件として使用 |
| intensity | 既存log処理 | `log1p`を明示的に使用 |
| loss mask |明示maskなし | 有効な生成奇数行のみ |

したがって、Transformer本体は変更していないが、入出力stem/head、patch size、
padding、mask、正規化はWaymo向けに変更している。既存KITTI checkpointをそのまま
利用せず、Waymo用2chモデルを新規学習している。

## 4. 学習データと条件

- データ数：240ペア
- train：160
- validation：40
- test：40
- 入力：`[2,16,2688]`
- 教師：`[2,32,2688]`
- native評価幅：2650
- range scale：75 m
- intensity transform：`log1p`
- 主loss範囲：有効な生成奇数行
- 比較baseline：nearest、linear

trainで学習し、validationで条件を選択する。testは条件決定後の最終確認にのみ
使用する。

## 5. 現在の評価手法

### 5.1 Range Image評価

有効な生成奇数行だけについて次を計算する。

- range MAE [m]
- raw intensity MAE
- log1p intensity MAE
- 負の予測値数・割合
- NaN/Inf

### 5.2 単純補間との比較

TULIP、nearest、linearを同一GT・同一mask・同一生成行で比較する。観測済み偶数行
を評価へ混ぜると誤差が不当に小さくなるため、主指標には使用しない。

### 5.3 反射強度帯別評価

raw GT intensityで次の帯に分ける。

```text
0–0.5 / 0.5–1 / 1–2 / 2–5 / 5–20 /
20–100 / 100–1000 / 1000以上
```

さらに閾値20、100、1000についてPrecision、Recall、F1を評価する。全体MAEだけ
では約98%を占める0–1帯の影響が強く、高反射点の失敗を見落とすためである。

### 5.4 三次元点群評価

復元rangeをWaymo vehicle座標へ逆投影し、次を計算する。

- coordinate MAE：X/Y/Zの絶対誤差平均
- Euclidean MAE：対応点間の三次元距離誤差
- 点群上のintensity MAE

### 5.5 カメラ投影評価

GT rangeを使用したoracle投影とTULIP予測rangeを使用した投影を比較する。

- カメラ内投影点数・投影率
- oracleとpredictionの画素ずれ平均
- 画素ずれp95
- RGB画像上での距離・反射強度overlay

## 6. これまでの主要結果

### 6.1 Controlled baseline：validation 40フレーム、生成行限定

| 手法 | range MAE [m] | raw intensity MAE | log1p intensity MAE |
|---|---:|---:|---:|
| TULIP | **0.8217** | **6.3311** | **0.06970** |
| Nearest | 2.3369 | 10.6640 | 0.09540 |
| Linear | 1.9594 | 9.8877 | 0.08830 |

TULIPはrangeと全体intensityで単純補間を下回った。ただし、高反射帯だけを見ると
単純補間より悪い帯が残る。

Controlled baselineの高反射検出は次の通りである。

| 閾値 | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 20以上 | 0.706 | 0.244 | 0.362 |
| 100以上 | 0.722 | 0.232 | 0.351 |
| 1000以上 | 0.513 | 0.145 | 0.227 |

高反射予測は比較的慎重でPrecisionは高いが、見逃しが多くRecallが低い。

### 6.2 低強度専用モデル

- 0–1帯 raw MAE：`0.06670`
- 0–1帯は生成有効画素の約`98.09%`
- 20/100/1000以上のRecall：すべて`0`

大部分を占める低強度帯は良くなるが、高反射点を一切出さなくなるため、このモデル
だけを最終結果にはできない。

### 6.3 Moderate高強度重みモデル：validation 40フレーム

| 指標 | 結果 |
|---|---:|
| range MAE | 0.8426 m |
| raw intensity MAE | 6.2277 |
| log1p intensity MAE | 0.07006 |
| F1（20以上） | 0.331 |
| F1（100以上） | 0.369 |
| F1（1000以上） | 0.307 |

1000以上F1はbaselineの0.227から0.307へ改善したが、rangeと低強度帯が悪化した。
単純に高強度重みを強くするだけでは、低強度・距離・高強度の同時維持が難しい。

### 6.4 既存full_e300 checkpointのtest点群評価

| 手法 | coordinate MAE [m] | Euclidean MAE [m] | 点群intensity MAE |
|---|---:|---:|---:|
| TULIP | **0.5422** | **1.2236** | **17.8362** |
| Nearest | 0.6342 | 1.4126 | 19.2183 |
| Linear | 0.9873 | 2.2384 | 18.3574 |

40 testフレーム、生成行の約149万点で評価した。TULIPは三次元幾何ではNearestと
Linearを下回り、点群補完として有効性が確認できた。一方、約1.22 mの三次元誤差
が後段マッチングで許容できるかは今後の検証対象である。

### 6.5 FRONTカメラ投影：full_e300、test 40フレーム

- oracle投影率：`0.12350`
- predicted投影率：`0.12353`
- 平均画素ずれ：`0.6167 px`
- p95画素ずれ：`2.2769 px`

点数・投影率はGT rangeと予測rangeでほぼ一致している。平均ずれは1画素未満だが、
誤差上位5%では約2.28画素のずれがある。

## 7. 新しい帯別バランス提案

低強度専用fine-tuneと高強度fine-tuneを交互に行うと、一方を改善した際に他方を
忘れる現象が確認された。そのため、モデル構造を変えず、1回のlossで全帯域を同時
に学習する方式を実装した。

```text
L_intensity = 0.90 L(0–1)
            + 0.05 L(1–20)
            + 0.02 L(20–100)
            + 0.02 L(100–1000)
            + 0.01 L(1000以上)
```

各項は画素総和ではなく帯内平均である。バッチに存在しない帯は除外し、存在する帯
の係数で再正規化する。これにより少数の高反射画素にも勾配を与えつつ、0–1帯を主
目的として維持する。

この方式は単体テスト21件と1フレームCUDA smoke testをPASSした。ただし、160件・
10 epochのvalidation性能はまだ未取得であり、改善したと結論付ける段階ではない。

採用判定の暫定目標は次の通りである。

- 0–1 raw MAE：0.080以下
- range MAE：0.825 m以下
- F1（20以上）：0.35以上
- F1（100以上）：0.35以上
- F1（1000以上）：0.25以上

## 8. 発表で表示する画像

### 8.1 Range ImageのGT・Low・Pred比較

最初に表示する画像：

`/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/pure_test_sample0/range_gt_low_pred.png`

別々に表示する場合：

- `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/separate_test_sample0/range_gt.png`
- `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/separate_test_sample0/range_low.png`
- `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/separate_test_sample0/range_pred.png`

見る点は、Lowの走査線間隔、Predで生成行が埋まっているか、物体境界がGTに近いか、
遠距離が平滑化されすぎていないかである。

### 8.2 反射強度のGT・Low・Pred比較

通常帯域の構造確認：

`/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/pure_test_sample0/intensity_gt_low_pred_linear.png`

広いダイナミックレンジの確認：

`/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/pure_test_sample0/intensity_gt_low_pred_log1p.png`

別々に表示する場合：

- `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/separate_test_sample0/intensity_gt_log1p.png`
- `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/separate_test_sample0/intensity_low_log1p.png`
- `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/separate_test_sample0/intensity_pred_log1p.png`

log1p画像は高強度まで表示しやすいが、raw誤差を小さくする処理ではない。色だけで
性能を判断せず、帯別MAEとPrecision/Recallを併記する。

### 8.3 FRONTカメラ上の距離overlay

- GT：`/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/final_front_sample0/gt/depth_overlay.png`
- Low：`/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/final_front_sample0/low/depth_overlay.png`
- Pred：`/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/final_front_sample0/pred/depth_overlay.png`

### 8.4 FRONTカメラ上の反射強度overlay

- GT：`/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/final_front_sample0/gt/intensity_log1p_overlay.png`
- Low：`/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/final_front_sample0/low/intensity_log1p_overlay.png`
- Pred：`/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/final_front_sample0/pred/intensity_log1p_overlay.png`

1枚で比較する場合：

`/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/pure_camera_test_sample0_front/intensity_overlay_gt_low_pred_log1p.png`

## 9. 画像を用いた考察

### 良好な点

- LowよりPredの走査線密度が増え、32ライン相当の出力を得られている。
- rangeの物体形状は単純補間より良く、三次元誤差でも改善している。
- FRONT投影では予測点数がGT相当で、平均画素ずれは1画素未満である。
- 0–1の通常反射強度帯は比較的高精度に復元できる。

### 課題

- 高反射点が極端に少なく、通常の画素平均lossでは学習への寄与が弱い。
- log1pはダイナミックレンジを圧縮するため、非常に大きい値の差も圧縮される。
- 低強度を優先すると高反射Recallが0になり、高反射を強くすると低強度とrangeが
  悪化する。
- 反射板や標識のような細い高反射構造は、縦方向補完で平均化されやすい。
- full_e300画像とcontrolled実験数値は異なるcheckpointである。同じスライドへ載せる
  場合は、画像側に`full_e300`と明記する。最新モデルの数値と画像を直接対応させる
  には、同じcheckpointから画像を再生成する必要がある。

## 10. 現在地と次の作業

完了していること：

1. Waymo TFRecordから64/32/16ラインデータ生成
2. 16→32の2ch Dataset
3. TULIP Transformerを再利用したWaymo adapter
4. CUDA forward・学習・checkpoint保存
5. train/validation/test分離
6. 生成奇数行限定の統一評価
7. nearest・linear比較
8. Range Image、点群、カメラ投影評価
9. GT・Low・Pred画像生成
10. 低強度・高強度のトレードオフ確認
11. 帯別バランスlossの実装とsmoke test

次に行うこと：

1. 帯別バランスモデルを10 epoch学習する。
2. validation 40フレームをcontrolled条件で評価する。
3. 0–1 MAE、range MAE、20/100/1000 F1をbaselineと比較する。
4. 条件を変更せず同一checkpointからGT・Low・Pred画像を再生成する。
5. validationで合格した場合のみtest評価と点群・FRONT画像評価へ進む。
6. その後、清水手法のLoFTR/PnPへ渡す。LoFTR/PnPはTULIP評価とは分離して記録する。

## 11. 発表での短い結論例

「TULIPのSwin Transformer本体を維持し、Waymoの幅と16→32倍率に対応する入出力
adapterを追加した。生成走査線に限定した評価では、TULIPはnearest・linear補間より
range誤差と三次元点群誤差を低減した。一方、Waymo intensityは極端なロングテール
分布を持ち、低強度と高反射の同時復元が課題である。現在はモデル構造を増やさず、
強度帯ごとの平均損失をバランスして学習する方法を検証している。」

## 12. 主要な数値・画像の出所

- Controlled baseline：`outputs/waymo_16_to_32_controlled_baseline_s0/validation40/summary.json`
- 低強度モデル：`outputs/waymo_16_to_32_controlled_low_0_1_s0/validation40/summary.json`
- Moderateモデル：`outputs/waymo_16_to_32_controlled_moderate_s0/validation40/summary.json`
- 三次元点群：`outputs/waymo_16_to_32_full_e300/final_pointcloud_test40/summary.json`
- FRONT投影：`outputs/waymo_16_to_32_full_e300/camera_test_front_40/summary.json`
- 帯別バランス設計：`docs/20260909_band_balanced_intensity_experiment.md`

