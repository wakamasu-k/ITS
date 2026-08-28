# KITTI GT・Low・Pred 疑似カラー画像生成手引書

## 1. 対象

`sequence 00 / frame 000000`のGT・Low・Predを同一条件でカメラへ投影し、
距離・反射強度の疑似カラー画像を再現するための手引書である。

- GT: 64ライン正解Range Imageから再構成した点群
- Low: 低解像度入力から再構成した点群
- Pred: 観測行を復元したTULIPの`pred_fused`点群

使用コード:

```text
/home/wakamatsu/ITS/TULIP_backup/tools/make_intensity_depth_overlays.py
```

## 2. 最後に生成した設定

### 距離

```text
表示量        : Camera Z（カメラ前方方向の奥行き）
表示範囲      : 4～40 m
正規化        : log
gamma         : 0.75
カラーマップ  : OpenCV Turbo（近距離=赤、遠距離=青紫）
色段階        : 256（連続表示）
点の描画半径  : 3 px
overlay alpha : 0.65
```

4 m未満と40 m超の点は削除されず、それぞれ端の色へクリップされる。
点群や距離値そのものは変更しない。

### 反射強度

最後に実際に生成した画像では次を使用した。

```text
mode          : fixed
表示範囲      : 0～1
カラーマップ  : OpenCV Turbo（低強度=青紫、高強度=赤）
色段階        : 256（連続表示）
```

車体などの低反射点は濃い青紫となり、黒背景や黒い車体に溶ける場合がある。

## 3. 入力ファイル

```text
GT:
debug_outputs/one_frame_000000/diagnostic/
  seq00_frame000000_gt_reconstructed_xyzi.ply

Low:
debug_outputs/one_frame_000000/diagnostic/
  seq00_frame000000_low_reconstructed_xyzi.ply

Pred:
debug_outputs/one_frame_000000/tulip_inference/
  seq00_frame000000_pred_fused_xyzi.ply

Camera:
/mnt/e/data_odometry_gray/dataset/sequences/00/image_0/000000.png

Calibration:
/mnt/e/data_odometry_calib/dataset/sequences/00/calib.txt
```

## 4. 実行前確認

```bash
source /home/wakamatsu/ITS/.venv_tulip/bin/activate
cd /home/wakamatsu/ITS/TULIP_backup

test -f debug_outputs/one_frame_000000/diagnostic/seq00_frame000000_gt_reconstructed_xyzi.ply
test -f debug_outputs/one_frame_000000/diagnostic/seq00_frame000000_low_reconstructed_xyzi.ply
test -f debug_outputs/one_frame_000000/tulip_inference/seq00_frame000000_pred_fused_xyzi.ply
test -f /mnt/e/data_odometry_gray/dataset/sequences/00/image_0/000000.png
test -f /mnt/e/data_odometry_calib/dataset/sequences/00/calib.txt
```

すべて終了コード0なら生成へ進む。

## 5. GT生成

```bash
python tools/make_intensity_depth_overlays.py \
  --points debug_outputs/one_frame_000000/diagnostic/seq00_frame000000_gt_reconstructed_xyzi.ply \
  --calib /mnt/e/data_odometry_calib/dataset/sequences/00/calib.txt \
  --image /mnt/e/data_odometry_gray/dataset/sequences/00/image_0/000000.png \
  --camera P0 \
  --out_dir debug_outputs/one_frame_000000/camera_compare_turbo_near_4_40m_point3/gt \
  --prefix seq00_frame000000_gt_turbo_near_4_40m_point3 \
  --depth_min 4 --depth_max 40 \
  --depth_scale log --depth_gamma 0.75 --depth_bins 256 \
  --intensity_mode fixed --intensity_min 0 --intensity_max 1 \
  --intensity_bins 256 --point_size 3 --alpha 0.65
```

## 6. Low生成

```bash
python tools/make_intensity_depth_overlays.py \
  --points debug_outputs/one_frame_000000/diagnostic/seq00_frame000000_low_reconstructed_xyzi.ply \
  --calib /mnt/e/data_odometry_calib/dataset/sequences/00/calib.txt \
  --image /mnt/e/data_odometry_gray/dataset/sequences/00/image_0/000000.png \
  --camera P0 \
  --out_dir debug_outputs/one_frame_000000/camera_compare_turbo_near_4_40m_point3/low \
  --prefix seq00_frame000000_low_turbo_near_4_40m_point3 \
  --depth_min 4 --depth_max 40 \
  --depth_scale log --depth_gamma 0.75 --depth_bins 256 \
  --intensity_mode fixed --intensity_min 0 --intensity_max 1 \
  --intensity_bins 256 --point_size 3 --alpha 0.65
```

## 7. Pred生成

```bash
python tools/make_intensity_depth_overlays.py \
  --points debug_outputs/one_frame_000000/tulip_inference/seq00_frame000000_pred_fused_xyzi.ply \
  --calib /mnt/e/data_odometry_calib/dataset/sequences/00/calib.txt \
  --image /mnt/e/data_odometry_gray/dataset/sequences/00/image_0/000000.png \
  --camera P0 \
  --out_dir debug_outputs/one_frame_000000/camera_compare_turbo_near_4_40m_point3/pred \
  --prefix seq00_frame000000_pred_turbo_near_4_40m_point3 \
  --depth_min 4 --depth_max 40 \
  --depth_scale log --depth_gamma 0.75 --depth_bins 256 \
  --intensity_mode fixed --intensity_min 0 --intensity_max 1 \
  --intensity_bins 256 --point_size 3 --alpha 0.65
```

## 8. 保存先と生成物

```text
debug_outputs/one_frame_000000/camera_compare_turbo_near_4_40m_point3/
├── gt/
├── low/
└── pred/
```

| 末尾 | 内容 |
|---|---|
| `_depth_color.png` | 黒背景のCamera Z疑似カラー |
| `_depth_overlay.png` | カメラ画像と距離疑似カラーの重畳 |
| `_depth_legend.png` | 距離色の凡例 |
| `_intensity_color.png` | 黒背景の反射強度疑似カラー |
| `_intensity_overlay.png` | カメラ画像と反射強度の重畳 |
| `_intensity_legend.png` | 反射強度色の凡例 |
| `_mask.png` | 投影点の描画領域 |

一覧確認:

```bash
find debug_outputs/one_frame_000000/camera_compare_turbo_near_4_40m_point3 \
  -type f -name "*.png" | sort
```

## 9. 最後の実行結果

| 種類 | 読み込んだ点数 | カメラ上の可視画素 | Camera Z最小/最大 |
|---|---:|---:|---:|
| GT | 51,396 | 7,686 | 4.066 / 75.930 m |
| Low | 12,640 | 1,855 | 4.099 / 74.389 m |
| Pred | 51,487 | 7,804 | 1.737 / 75.327 m |

GTの7,686画素は以前の描画と一致しており、表示設定変更で点が削除されて
いない。Predの4 m未満の点は4 m側の色になるため、物理的妥当性は別途評価する。

## 10. 注意事項

1. 距離色はLiDARの三次元rangeではなく`Camera Z`を表す。
2. log表示は近距離差を広げる一方、遠距離差を圧縮する。
3. `point_size=3`は描画だけを拡大し、新しい点を生成しない。
4. 点を大きくすると近い点が遠い点を覆い、輪郭が太く見える。
5. GT・Low・Pred比較では全表示オプションを同じにする。
6. 見た目だけでモデル性能を判断せず、MAE・RMSEも確認する。
7. 正確な投影確認用には`point_size=0`または`1`も保存する。
8. Pred評価では`pred_fused`と`pred_raw`を区別する。

## 11. 車両の反射強度が見えにくい理由

右側車両付近には約1,070個の可視点があり、点自体は欠落していない。
同領域では強度0が約40.1%、強度0.1未満が約42.8%だった。低強度はTurboの
濃い青紫となり、黒背景や黒い車体に溶ける。ナンバープレートなど高反射部のみ
橙～赤で目立つ。

低反射確認専用の候補設定:

```bash
--intensity_mode fixed \
--intensity_min -0.15 \
--intensity_max 0.6 \
--intensity_bins 256
```

これは最後に生成した設定ではない。使用時は別フォルダへ保存し、GT・Low・Pred
すべてを同じ設定で生成する。

## 12. 別フレームへの変更

次の組を同時に揃える。

- sequence番号
- frame番号
- GT・Low・Pred点群
- カメラ画像
- calibration
- 出力prefix

同じフレーム番号でもsequenceが異なれば別場面なので注意する。
