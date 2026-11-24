# Waymo 16→32 点群評価結果: validation 40フレーム

## 条件

- checkpoint: `outputs/waymo_16_to_32_tail_weighted_all_t20_w2/checkpoint.pth`
- dataset role: validation
- evaluated frames: 40
- intensity transform: raw
- coordinate frame: Waymo vehicle frame
- selected native rings: `0,2,...,62`
- evaluated rows: generated odd rows `1,3,...,31`
- metric: same range-image pixelのpaired vehicle-frame point error
- Chamfer距離ではない

## 集計結果

| 方法 | 有効点数 | coordinate MAE [m] | Euclidean MAE [m] | intensity MAE |
|---|---:|---:|---:|---:|
| TULIP | 1,537,681 | 1.3997 | 3.2528 | **6.2660** |
| nearest | 1,485,117 | 0.5929 | 1.3374 | 8.3412 |
| linear | 1,530,812 | 0.8052 | 1.8544 | 9.8479 |

## 解釈

TULIPはrangeから復元した3D位置ではnearest/linearより悪い。
これは現段階のrange復元誤差が点群座標へ直接伝播しているためである。

一方、raw intensityのMAEはTULIPが最も小さい。

- nearest比: 約24.9%低い
- linear比: 約36.4%低い

したがって、現在のモデルは「rangeの幾何復元」よりも「欠損ringの反射強度推定」に有効性が現れている。
これは可視光画像とLiDAR反射強度画像をマッチングする目的と整合する。

## 注意点

この評価は5フレーム評価の延長として行った40フレーム評価であり、全Waymoデータの最終結果ではない。
また、TULIP点群は同一画素対応のpaired metricで評価しており、Chamfer距離・点群密度・物体別性能は未評価である。

出力:

- `outputs/waymo_16_to_32_tail_weighted_all_t20_w2/validation_pointcloud_40/summary.json`
- `outputs/waymo_16_to_32_tail_weighted_all_t20_w2/validation_pointcloud_40/per_frame.csv`

## 次段階

1. 同じcheckpointでtest 40フレームを点群評価する。
2. intensity帯別に点群intensity誤差を集計する。
3. TULIP出力を反射強度画像として可視光カメラ画像との対応付けへ接続する。
4. range改善は、intensityマッチングに必要かを別評価する。

---

作成日（ファイル更新日時から推定）: 2026-08-20
