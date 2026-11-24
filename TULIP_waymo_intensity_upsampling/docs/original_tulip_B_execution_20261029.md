# 最初の通常TULIP復元BでRetrievalを再学習する実行手順

## 今回固定するもの

- TULIP: `outputs/waymo_16_to_32_baseline_to_moderate_s0_e20/checkpoint.pth`。このモデル自体は再学習しない。
- 原観測16ラインを保持し、追加走査線へ予測値を入れる既存の点群融合。画像の観測部分保持合成は行わない。
- 距離75mスケール、強度log1p、予測距離0<r<=75mの既存フィルタ。
- 元のカメラ画像とアンカー配置。学習196segment、検証374Query/443DB、test507Query/754DBを分離。
- testには`shimizu_protocol_507q_754db_abc`内の最初のB画像をそのまま使う。新しいtest画像の都合のよい選択は行わない。
- TULIP学習8segmentと、位置推定の検証・testに使うsegmentとの重複0をprepareで検査済み。

再学習するのはRetrieval。従来重み固定でSoft5m R@10=84.02%だった通常Bについて、画像に対応した学習で検索が改善するかを見る。観測部分保持方式はこの実験の後に扱う。

## 実行コード

共通の起動スクリプト:

```bash
LAUNCH=/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/scripts/run_original_tulip_retrieval.sh
```

1. 準備確認と最初の1segment（prepareは今回実行済み、再度呼んでも同じ計画なら再利用）:

```bash
bash "$LAUNCH" prepare
bash "$LAUNCH" data --role train --limit 1
bash "$LAUNCH" status
```

1segmentでも全フレーム、全アンカーと学習用ずらしRIを生成する。地図はPyTorch TULIP/CUDAで復元し、TensorFlowの幾何処理はCPU。描画とずらしRI生成はNumPy/OpenCVのCPU処理。GPU使用率が描画中に下がるのは正常。

2. 全データの生成と学習前検査:

```bash
(
set -euo pipefail
LAUNCH=/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/scripts/run_original_tulip_retrieval.sh
bash "$LAUNCH" data --role train
bash "$LAUNCH" data --role val
bash "$LAUNCH" manifests
bash "$LAUNCH" stats
bash "$LAUNCH" check
)
```

学習196＋検証3segmentの新しいB地図と画像が必要なため大規模処理。既存の地図は完了記録とcheckpointのSHAを確認して再利用し、画像も元rendererのresume処理を使う。不完全な地図は上書きせず停止する。学習用の各anchorから最大8枚のずらしRIを作り、epochごとに巡回する。

検査で見る値: train=28772、val_pairs=374、val_db=443、test_pairs=507、test_db=754。checkでは全距離でempty_gt=0、Soft5m候補数中央値10程度を確認する。

3. Retrieval学習・重み選択・test:

```bash
(
set -euo pipefail
LAUNCH=/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/scripts/run_original_tulip_retrieval.sh
bash "$LAUNCH" train
bash "$LAUNCH" select
bash "$LAUNCH" evaluate
)
```

モデルはResNet34、NetVLAD64、埋め込み256、512x864、80epoch、batch16。学習画像のmean/stdを用い、374validation QueryのみのSoft5m R@10で選ぶ。元trainerのbest.ckptはStrict監視なので、この実験ではselected_retrieval.jsonに記録されたepochを使う。Soft全0なら選択を停止する。

中断時のtrainの自動resumeは未実装。データ生成や既存結果を上書きせず、保存されたcommand.jsonとcheckpointからresume条件を確認する。

4. 通常Bで学習した検索結果からLoFTR/PnPへ:

```bash
(
set -euo pipefail
LAUNCH=/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/scripts/run_original_tulip_retrieval.sh
bash "$LAUNCH" dense-prepare
bash "$LAUNCH" dense-pilot
)
```

pilotの10Query・100候補を確認してから:

```bash
bash /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/scripts/run_original_tulip_retrieval.sh dense-all
```

LoFTRは既存Stage-1重み固定。BでLoFTRを再学習したとは呼ばない。選択はPnP成功・最大インライア数、同点時は再投影誤差・3D対応数・検索順位。距離や正解シーンによる候補の事前除外はしない。

## 見る結果

- `outputs/original_tulip_B_retraining_20261029/retrieval_test/original_B_retrained_metrics.json`
- `outputs/original_tulip_B_retraining_20261029/dense/all507_pnp/integrated_summary.json`
- Retrieval主指標: test Soft5m R@10。Strict R@1/5/10、Soft2m/10m R@10、順位中央値も確認。
- 位置推定: 全候補失敗数、別シーン選択数、全507を分母とする1/2/5m以内成功率、成功・比較可能例の中央値とその対象数。

この実験と過去通常B固定重みの差は、Retrievalの重みだけでなく正規化と検証選択方針も変わった影響を含む。検索が回復しただけでTULIPの幾何誤差が改善したとは言わない。

## 検証済み範囲

prepareで元データパス、アンカー配置、分割、固定TULIP重み、TULIP学習と位置推定heldoutの重複を確認。新規コードの構文とCLIを検査。学習・検証用Bの全地図生成とRetrieval再学習はまだ実行していない。最初の1segmentから実データ処理を確認する。
