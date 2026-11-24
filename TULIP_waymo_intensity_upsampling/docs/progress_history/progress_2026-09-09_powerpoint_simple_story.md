# Waymo 16→32 TULIP研究：PowerPoint用の簡潔な説明構成

作成日：2026-09-09

## 発表全体で伝える一文

「16ラインのWaymo LiDARから、TULIPを使って32ラインの距離と反射強度を復元し、
単純補間より良い三次元形状を得られることを確認した。現在は、数が非常に少ない
高反射点の復元改善に取り組んでいる。」

## スライド1：研究目的

### タイトル

TULIPによるWaymo LiDARの16→32ラインアップサンプリング

### スライドに書く内容

```text
入力：16ラインの低密度LiDAR
        ↓ TULIP
出力：32ラインの距離＋反射強度
        ↓
高密度点群・カメラ画像との対応へ利用
```

### 話す内容

「本研究では、低密度な16ラインLiDARから32ライン相当の情報を復元します。
距離だけでなく反射強度も同時に復元し、最終的には画像照合や自己位置推定へ利用する
ことを目指しています。」

### 載せる画像

`/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/pure_test_sample0/range_gt_low_pred.png`

画像上に`GT 32 lines / Low 16 lines / TULIP Pred 32 lines`と追記する。

## スライド2：データの作り方

### タイトル

Waymo 64ラインから学習ペアを作成

### スライドに書く内容

```text
Waymo TOP LiDAR 64ライン
  ↓ 偶数ringを選択
32ラインGT
  ↓ さらに偶数行を選択
16ライン入力
```

```text
32ライン上の偶数行：観測済み
32ライン上の奇数行：TULIPが生成する行
```

### 話す内容

「Waymoの64ラインRange Imageから32ラインの正解データを作り、その半分を取り出して
16ライン入力としました。評価では、モデルが実際に生成した奇数行だけを使います。
観測済みの行を混ぜて精度を良く見せないことが重要です。」

## スライド3：提案手法

### タイトル

TULIP本体を維持したWaymo向け適用

### スライドに書く内容

```text
[2,16,2650]
  ↓ range / 75、intensity=log1p
  ↓ circular padding：2650→2688
  ↓ Waymo入出力adapter
  ↓ TULIP Swin Transformer
[2,32,2688]
  ↓ crop
[2,32,2650]
```

```text
維持：Transformer、Attention、Encoder、Decoder、skip connection
変更：入出力倍率、patch、padding、最終拡大部、valid mask
```

### 話す内容

「TULIPのSwin Transformer本体は変更していません。Waymoの幅2650と16→32倍率へ
合わせるため、入口と出口のadapter、padding、mask処理だけをWaymo向けにしました。
2チャンネルはrangeとintensityです。」

## スライド4：評価方法

### タイトル

二次元・三次元・カメラ投影の3段階で評価

### スライドに書く内容

| 評価段階 | 確認内容 |
|---|---|
| Range Image | range MAE、raw/log1p intensity MAE |
| 3D点群 | 座標MAE、三次元距離MAE |
| カメラ投影 | 投影率、GTとの画素ずれ |

比較対象：`TULIP / Nearest / Linear`

評価範囲：`有効な生成奇数行のみ`

### 話す内容

「見た目だけではなく、Range Image上の誤差、点群へ戻した三次元誤差、カメラへ投影
した画素ずれの3段階で確認します。すべて同じ生成行maskで単純補間と比較します。」

## スライド5：主要結果

### タイトル

距離・三次元形状は単純補間より改善

### Range Image結果

Controlled baseline、validation 40フレーム、生成行限定：

| 手法 | range MAE | log1p intensity MAE |
|---|---:|---:|
| TULIP | **0.822 m** | **0.0697** |
| Nearest | 2.337 m | 0.0954 |
| Linear | 1.959 m | 0.0883 |

### 三次元点群結果

full_e300、test 40フレーム：

| 手法 | 座標MAE | 三次元距離MAE |
|---|---:|---:|
| TULIP | **0.542 m** | **1.224 m** |
| Nearest | 0.634 m | 1.413 m |
| Linear | 0.987 m | 2.238 m |

### 話す内容

「距離画像と三次元点群の両方で、TULIPはNearestとLinearを下回りました。この結果
から、TULIPは単に行数を増やすだけでなく、三次元形状を単純補間より良く復元して
いると判断できます。」

### 注意

上段はcontrolled baseline、下段は既存full_e300であり、checkpointが異なる。
発表資料では表の横に実験名を必ず表示する。

## スライド6：反射強度の課題

### タイトル

通常反射は復元できるが、高反射点の見逃しが残る

### スライドに書く内容

```text
0～1の反射強度：約98%
高反射点：全体のごく一部
```

Controlled baseline：

| 高反射閾値 | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 20以上 | 0.706 | 0.244 | 0.362 |
| 100以上 | 0.722 | 0.232 | 0.351 |
| 1000以上 | 0.513 | 0.145 | 0.227 |

### 話す内容

「高反射点を出したときの正しさを示すPrecisionは比較的高い一方、Recallが低く、
実際の高反射点を多く見逃しています。全体MAEだけでは、約98%を占める0～1帯に
隠れてこの問題が見えません。」

### 載せる画像

`/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/pure_test_sample0/intensity_gt_low_pred_log1p.png`

画像の高反射位置へ丸印を付け、GTとPredの差を示す。

## スライド7：現在の改善案

### タイトル

モデルを変えず、強度帯ごとに学習をバランス化

### スライドに書く内容

```text
L_intensity = 0.90 L(0–1)
            + 0.05 L(1–20)
            + 0.02 L(20–100)
            + 0.02 L(100–1000)
            + 0.01 L(1000以上)
```

```text
目的：
・0～1帯の精度を維持
・少数の高反射点にも学習勾配を与える
・Transformer本体は変更しない
```

### 話す内容

「低強度だけを学習すると高反射Recallが0になり、高反射の重みを強くすると距離と
低強度が悪化しました。そこで、各強度帯の平均誤差を1回のlossで同時に扱う方式を
実装しました。現在は動作確認まで完了し、次にvalidationで効果を判定します。」

## スライド8：結論と次の作業

### タイトル

現在地と今後

### スライドに書く内容

```text
確認済み
✓ 16→32・2ch forwardと学習
✓ 距離・三次元点群で単純補間を改善
✓ FRONTカメラ投影
✓ 高反射問題の定量化

次の作業
1. 帯別バランスlossをvalidation評価
2. 同じcheckpointでGT・Low・Pred画像を再生成
3. 点群・FRONT投影を再評価
4. その後にLoFTR/PnPへ接続
```

### 話す内容

「TULIPによる16→32点群補完の基本経路は完成しています。距離復元は有効ですが、
高反射点のRecall改善が残っています。次はモデル構造を増やさずに帯別バランス学習
を評価し、同じcheckpointで数値と画像をそろえます。」

## 発表で優先して使う画像パス

1. Range Image GT/Low/Pred  
   `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/pure_test_sample0/range_gt_low_pred.png`

2. Intensity GT/Low/Pred  
   `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/pure_test_sample0/intensity_gt_low_pred_log1p.png`

3. FRONT距離GT  
   `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/final_front_sample0/gt/depth_overlay.png`

4. FRONT距離Low  
   `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/final_front_sample0/low/depth_overlay.png`

5. FRONT距離Pred  
   `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/final_front_sample0/pred/depth_overlay.png`

6. FRONT反射強度GT/Low/Pred  
   `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/outputs/waymo_16_to_32_full_e300/pure_camera_test_sample0_front/intensity_overlay_gt_low_pred_log1p.png`

## 作成時のルール

- 1枚のスライドでは1つの結論だけを伝える。
- GTは左、Lowは中央、Predは右に統一する。
- GT/Low/Predで同じカラーマップと表示範囲を使う。
- `full_e300`と`controlled`の結果を同じcheckpointとして扱わない。
- 画像の見た目と、生成行限定の数値を必ずセットで説明する。
- raw intensity MAEとlog1p intensity MAEは単位・意味が違うため同じ列で比較しない。

