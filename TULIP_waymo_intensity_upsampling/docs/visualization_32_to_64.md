# Waymo 32->64ペアの固定スケール可視化

## 実行

```bash
source /home/wakamatsu/ITS/.venv_tulip/bin/activate
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
bash scripts/20_visualize_one_frame_32_to_64.sh
```

## 生成先

`outputs/one_frame_000000/visualization`

## 画像の意味

- `range_64_gt.png`: 加工前64ラインGT距離
- `range_32_input.png`: packedされた32ライン入力距離
- `range_32_on_64_rows.png`: 32ラインを元の偶数行へ戻した距離
- `intensity_64_gt_fixed.png`: 64ラインGT反射強度、固定0～0.75表示
- `intensity_32_input_fixed.png`: 32ライン入力反射強度
- `intensity_32_on_64_rows_fixed.png`: 元の偶数行へ戻した反射強度
- `intensity_64_gt_log.png`: 22,016級の外れ値を確認するlog1p表示
- `valid_mask_64.png`: 64ライン有効マスク
- `valid_mask_32.png`: packedされた32ライン有効マスク
- `comparison_32_to_64.png`: 距離・反射強度・maskの比較一覧
- `visualization_metadata.json`: スケール、行番号、統計、検証結果

## 表示規則

- 距離: 全画像で0～75m
- 固定反射強度: 全画像で0～0.75
- invalid: 黒
- row 0: 上
- 横方向: Waymo native azimuth column order
- 補間: nearest（画像表示時に行間を補間しない）
- 画像ごとのmin-max正規化: 使用しない

## 対応検証

可視化前に、32ラインの距離・反射強度が64ラインGTの
`0,2,...,62`行と完全一致することを検証する。一致しなければ画像を保存せず
エラー終了する。黒い奇数行`1,3,...,63`がTULIPの生成対象である。
