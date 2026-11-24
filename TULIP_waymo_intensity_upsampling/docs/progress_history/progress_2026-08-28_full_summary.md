# TULIP Waymo 16→32 進捗まとめ

作成日: 2026-08-28

## 1. 目的

KITTI向けTULIPの中心構造を維持し、Waymo TOP LiDARの16ライン入力から32ラインのrange/intensity画像を復元する。将来的には、復元した反射強度画像を可視光カメラ画像との対応付けに利用する。

## 2. 維持している方針

- TULIP_backupの既存KITTIコードは変更していない。
- Transformer Block、Encoder、Decoder、Attention、Patch Merging、Patch Unmerging、skip connectionは変更していない。
- Waymo固有処理はTULIP_waymo_intensity_upsampling側に分離している。
- Waymo用モデルは2chの新規学習であり、既存1ch checkpointは流用していない。
- native幅2650はDatasetで2688へcircular paddingし、評価時に2650へcropする。
- /mnt/w/32lineおよびhandover_shimizuは変更していない。

## 3. Phase 1: Waymo Range Image

Waymo TFRecordからTOP LiDAR第1returnを取得した。

- native shape: [64,2650,4]
- return index: 0
- channel 0: range
- channel 1: intensity
- range/intensity: float32
- 無効条件: range <= 0、またはrange/intensityが非有限
- 無効画素のrange/intensity: 0
- ring_ids_32: [0,2,4,...,62]
- ring_ids_16: [0,4,8,...,60]
- Phase 1 pytest: 4 passed

## 4. Phase 2: intensity統計

複数TFRecord・複数フレームで、32ラインGTのvalid画素を逐次集計した。

- 無効値、NaN、Infは統計から除外
- rangeとintensityは同じvalid位置を使用
- intensityの大半は低値だが、1000以上の極端値が少数存在
- 1000以上は少数でもraw MAEを大きく支配する
- maximum_rangeとintensity_scaleは自動固定していない

主条件はrange scale 75.0 m、intensity log1pとし、評価時はraw domainへ戻す方式にした。

## 5. Phase 3: raw 32ラインGT

Waymo TFRecordからraw 32ラインGTを保存する処理を実装・検証した。

- 出力shape: [32,2650,2]
- dtype: float32
- channel 0: raw range
- channel 1: raw intensity
- 保存時の正規化、clip、log1p、gamma補正、2688 paddingはなし
- sidecar JSON、manifest、failuresを生成

## 6. Datasetと入力変換

検証用indexは240フレームで構成される。

- train: 160フレーム、8 segment
- validation: 40フレーム、2 segment
- test: 40フレーム、2 segment
- train/validation/test間のsegment重複: 0

Dataset出力:

- input: [2,16,2688], torch.float32
- target: [2,32,2688], torch.float32
- batch input: [B,2,16,2688]
- batch target: [B,2,32,2688]
- range: range / 75.0
- intensity: log1p
- 2650→2688: 水平方向circular padding
- Dataset pytest: 4 passed

## 7. TULIP forwardとadapter

Waymo adapterから既存TULIP factoryを再利用している。

- patch_size=(1,2)
- window_size=(2,8)
- depths=(2,2,2,2)
- num_heads=(3,6,12,24)
- embed_dim=96
- in_chans=2
- input: [B,2,16,2688]
- output: [B,2,32,2688]
- crop後: [B,2,32,2650]

入力patch embeddingと最終の異方的アップサンプリングはWaymo adapter側で調整している。Transformer本体は変更していない。

TULIP backboneのmc_drop=True経路ではtarget値を推論入力として使用せず、adapter側でtarget maskを用いてlossを計算している。通常学習lossはtargetの全valid行を対象にする。未観測奇数行のみのlossは別実験として扱う。

## 8. 学習確認

小規模確認:

- 1サンプル20 epoch: loss 5.6853 → 4.0874
- 5サンプル5 epoch: loss 8.8670 → 7.5435
- 20サンプル10 epoch: loss 9.5221 → 7.1971

主checkpoint:

outputs/waymo_16_to_32_fixed_e100/checkpoint.pth

- train: 160フレーム
- epoch: 100
- batch size: 1
- learning rate: 1e-4
- range scale: 75.0 m
- intensity transform: log1p
- range weight: 0.05
- intensity weight: 1.0
- 学習loss: 0.06898 → 0.01379

追加の本格学習相当checkpoint:

outputs/waymo_16_to_32_full_e100/checkpoint.pth

- 学習loss: 0.0645843 → 0.0135599
- 100 epochを正常終了
- checkpoint保存・再読み込み成功

## 9. 画像評価

waymo_16_to_32_full_e100のraw値評価:

| split | range MAE | intensity MAE |
|---|---:|---:|
| validation 40 | 0.6016 m | 6.6226 |
| test 40 | 0.7146 m | 13.0841 |

rangeは良好だが、validationからtestでintensity誤差が増えている。

## 10. intensity帯別評価

0〜1000の範囲ではTULIPが最近傍・線形補間より良い帯域が多い。一方、1000以上の極端値は十分に復元できていない。

- 低強度帯: TULIPが良好
- 1〜1000: TULIPが補間より良い帯域が多い
- 1000以上: raw MAEが大きく、外れ値再現は未解決
- raw tail weightだけでは1000以上を十分改善できなかった

## 11. 点群復元評価

### validation 40フレーム

出力:

outputs/waymo_16_to_32_fixed_e100/pointcloud_validation40_integrated/summary.json

- TULIP coordinate MAE: 0.4904 m
- TULIP Euclidean MAE: 1.1214 m
- TULIP intensity MAE: 6.2029
- nearest Euclidean MAE: 1.3373 m
- linear Euclidean MAE: 1.8544 m

### test 40フレーム

出力:

outputs/waymo_16_to_32_full_e100/pointcloud_test40/summary.json

- TULIP coordinate MAE: 0.5885 m
- TULIP Euclidean MAE: 1.3284 m
- TULIP intensity MAE: 17.8709
- nearest Euclidean MAE: 1.4126 m
- linear Euclidean MAE: 2.2384 m

統合環境・stubなしでvalidation/testとも40フレーム完走した。TULIPは点群座標、3D距離、intensityで比較した補間手法を上回った。

## 12. カメラ投影

FRONT、FRONT_LEFT、FRONT_RIGHT、SIDE_LEFT、SIDE_RIGHTへの投影を確認した。

- GT rangeによるoracle投影
- TULIP rangeによる予測投影
- projected point数、projection rate
- oracleと予測のpixel shift
- raw projection NPZ
- percentile正規化PNG

RGBとの単純な画素相関は強くなかった。LiDAR intensityとRGB輝度は異なる物理量なので、これは直ちに投影失敗を意味しない。LoFTR・PnPは清水手法側で実施するため後回しとしている。

## 13. 統合実行環境

現在の主環境:

/home/wakamatsu/ITS/.venv_tulip

同一環境で確認済み:

- PyTorch 2.0.0+cu118
- CUDA利用可能
- TensorFlow 2.12.0
- timm 0.6.13
- chamfer_distance
- Waymo frame_utils
- TULIP本体
- TensorFlowによるRTX 3060認識
- stubなしの点群評価40フレーム

pip checkにはJAXやWaymo補助依存に関する警告が残っているが、今回のTFRecord読み込み、TULIP学習、推論、点群評価は正常完走している。

## 14. 現在できていること

- Waymo TFRecord読み込み
- TOP LiDAR第1return取得
- 64→32→16ライン生成
- raw 32ラインGT保存
- 2ch PyTorch Dataset
- 2688幅paddingと2650幅crop
- TULIP 16→32 forward
- 2ch新規学習
- range/intensity画像評価
- 未観測奇数行評価
- intensity帯別評価
- 点群復元評価
- カメラ投影評価
- 統合環境でのstubなし評価

## 15. 残っている課題

1. 現在の240フレームはWaymo全データではなく検証用subsetである。
2. intensity 1000以上の外れ値再現は未解決である。
3. validationとtestでintensity誤差に差がある。
4. 複数seedによる再現性は未確認である。
5. validation最良epochの自動選択は未実装で、現状は最終epochを保存している。
6. pip checkは完全cleanではないが、対象処理は動作している。
7. RGBとの本格的な特徴対応、LoFTR、PnPは未実施である。

## 16. 総合判定

以下のTULIP中心経路はWaymo 16→32・2ch条件で動作している。

LiDAR → Range Image → TULIPによる高解像度復元 → 点群復元

本格学習、validation、test、点群評価まで技術的に完了した。現時点でTransformer本体を追加変更する必要はない。

次は、現在のcheckpointに対するgenerated-only評価、intensity帯別評価、複数seed確認を行い、その後に必要ならWaymo全体規模の学習へ進む。LoFTR・PnPは清水手法側で実施する。


