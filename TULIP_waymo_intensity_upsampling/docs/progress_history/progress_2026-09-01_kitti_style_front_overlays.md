# Waymo KITTI-style FRONT overlay visualization — 2026-09-01

## 目的

既存のWaymo GT・Low・Pred車両座標点群を、KITTIの確認手引書に近い形式で
FRONTカメラへ投影する。点が薄くRGB重畳で見えない問題に対し、表示専用の
塗りつぶし円、黒背景疑似カラー、RGB重畳、凡例、マスクを保存する。

## 実装範囲

新規ファイルのみを追加した。

- `tools/make_waymo_front_kitti_style_overlays.py`

既存のTULIP、Phase 1〜4A、点群出力、カメラ投影コードは変更しない。
点群NPZを再利用するため、モデル推論・学習は実行しない。

## 入力

- 点群: `outputs/waymo_16_to_32_full_e300/gt_low_pred_pointcloud_test40/sample_000000/gt_low_pred_pointcloud.npz`
- メタデータ: 同じディレクトリの `metadata.json`
- カメラ: `FRONT`
- 描画点サイズ: `3`
- 重畳 alpha: `0.65`

## 座標・距離の注意

KITTIの `Z_cam` と異なり、今回のWaymo投影規約は `X_c` がカメラ前方である。
したがって距離画像は「camera-forward depth Xc (m)」として保存する。LiDAR原点
からの放射状rangeではない。

## 保存画像

`gt/`, `low/`, `pred/` の各ディレクトリに以下を保存する。

- `depth_color.png`: 黒背景・camera-forward XcのTurbo疑似カラー
- `depth_overlay.png`: RGB画像へのXc重畳
- `depth_legend.png`: Xc凡例
- `intensity_color.png`: raw intensityの線形表示
- `intensity_overlay.png`: raw intensityのRGB重畳
- `intensity_log1p_color.png`: raw intensityのlog1p表示
- `intensity_log1p_overlay.png`: raw intensityのlog1p RGB重畳
- `intensity_legend.png`, `intensity_log1p_legend.png`
- `mask.png`: 描画領域
- `projection_raw.npz`: 投影座標、Xc、raw intensity、nearest-pixel map/mask

PNGの正規化は表示専用であり、NPZのraw値は変更しない。

通常強度を見やすくする追加出力を次に生成した。

`outputs/waymo_16_to_32_full_e300/camera_compare_front_sample0_kitti_style_intensity_normal/`

- 線形intensity表示: `0..1`
- log1p intensity表示: raw値 `0..20`
- 点サイズ: `3`
- RGB重畳alpha: `0.65`

GT・Low・Predの全`intensity_color.png`、`intensity_overlay.png`、
`intensity_log1p_color.png`、`intensity_log1p_overlay.png`を読み込み確認した。
画像サイズはすべて`1280x1920x3`、maskは`0,255`の二値である。

`0..1`は通常強度の色差比較用で、1を超える外れ値は飽和する。一方、外れ値を
含む全体確認には元の`0..20000` log1p出力を使う。両者は用途が異なるため、
同じ画像として解釈しない。

## 反射強度カラーマップ変更 — 2026-09-01

反射強度の色差を見やすくするため、距離画像のTurboは維持し、強度画像だけ
`plasma`へ変更できるようにした。`plasma`は低強度の紫から中強度の赤・橙、
高強度の黄色へ変化するため、黒背景およびRGB重畳で強度差を確認しやすい。

別出力:

`outputs/waymo_16_to_32_full_e300/camera_compare_front_sample0_kitti_style_plasma/`

設定:

- intensity線形表示: `0..1`
- intensity log1p表示: `0..20`
- intensity colormap: `plasma`
- depth colormap: `turbo`
- point size: `3`
- overlay alpha: `0.65`

GT・Low・Predの強度color/overlay/log1p画像を読み込み確認した。代表画像の
色数は単一ではなく、GTの`intensity_color.png`だけでも12,800色が存在する。
raw投影値は変更していない。

## 遠距離の色分離追加 — 2026-09-01

従来の距離表示は`depth-max=40 m`だったため、40 m超の点が同じ色に飽和して
いた。新規ツールに`piecewise`距離マッピングを追加し、別出力を生成した。

出力先:

`outputs/waymo_16_to_32_full_e300/camera_compare_front_sample0_kitti_style_depth_piecewise/`

設定:

- 距離範囲: `4..75 m`
- 近距離区間: `4..20 m`、既存と同じlog1p+gamma 0.75
- 遠距離区間: `20..75 m`、別の色域へ線形展開
- 近距離用色域: Turboの65%
- 遠距離用色域: Turboの35%
- intensity線形表示: `0..1`
- intensity log1p表示: `0..20`

この設定では、近距離の色変化を維持しつつ、40 m超の壁・柱・遠景を同じ最大色
へ飽和させずに比較できる。全画像を再読込でき、距離・強度画像は1280x1920x3、
maskは0/255の二値である。既存の40 m表示フォルダは保持している。

## 実行コマンド

```bash
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
source /home/wakamatsu/ITS/.venv_waymo/bin/activate
python tools/make_waymo_front_kitti_style_overlays.py \
  --pointcloud-npz outputs/waymo_16_to_32_full_e300/gt_low_pred_pointcloud_test40/sample_000000/gt_low_pred_pointcloud.npz \
  --metadata outputs/waymo_16_to_32_full_e300/gt_low_pred_pointcloud_test40/sample_000000/metadata.json \
  --output-dir outputs/waymo_16_to_32_full_e300/camera_compare_front_sample0_kitti_style \
  --camera FRONT \
  --depth-min 4 --depth-max 40 \
  --depth-gamma 0.75 \
  --intensity-min 0 --intensity-max 20 \
  --intensity-log-max 20000 \
  --point-size 3 --alpha 0.65
```

The `0..20` intensity range is a Waymo-adapted display range. KITTI's `0..1`
setting must not be copied blindly because Waymo raw intensity has a long tail.

## 実行結果

出力先:

`outputs/waymo_16_to_32_full_e300/camera_compare_front_sample0_kitti_style/`

| mode | カメラ内投影点 | 描画領域画素 | Xc最小 | Xc最大 | intensity最小 | intensity最大 |
|---|---:|---:|---:|---:|---:|---:|
| GT | 9,248 | 401,435 | 6.194 m | 74.335 m | 0.000220 | 11,584.003 |
| Low | 4,654 | 202,522 | 6.300 m | 74.335 m | 0.000220 | 6,079.997 |
| Pred | 9,248 | 401,416 | 4.110 m | 74.181 m | 0.000000 | 427.644 |

全PNGをOpenCVで再読込できた。color/overlay/legend/maskに読み込みエラーは
なく、color画像の点領域は黒一色ではない。maskはGT/Low/Predとも値が
`0,255`のみの二値画像である。

なお、描画領域画素は点サイズ3の円を塗った表示上の画素数であり、新しい
LiDAR点を生成した数ではない。LowがGTより少ないのは16ライン観測点が少ない
ためで、Predの投影点数はこの入力NPZの評価条件に従う。

TensorFlowからはAVX2/FMA、TensorRT未検出、NUMA、GPUライブラリ不足による
CPUフォールバック、PlaceholderのDEBUG INFOが表示された。今回はTFRecordの
読み込みと画像生成が完走しており、画像生成結果を中断する致命的エラーではない。

## 確認上の結論

「点が薄い」問題には `--point-size 3` が反映され、RGB重畳は元のカメラ画像を
背景として色点を描画する形式になった。Waymo raw intensityの長い外れ値は、
線形画像では `0..20`、別画像では `log1p` と `0..20000` の表示スケールを使う。
これはPNGだけの表示設定であり、`projection_raw.npz`のraw値や学習データは
変更していない。
