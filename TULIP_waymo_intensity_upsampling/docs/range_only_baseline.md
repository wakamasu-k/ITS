# ベースラインA: Waymo range-only TULIP

## 目的

TULIP論文の中心であるrange imageの距離復元を、Waymoの32→64ラインへ適用する。
intensityは入力・教師に使用せず、後続の提案手法Bと分離する。

## 既存TULIPへの最小変更

既存実装の論文設定は1×4 row patchと16→64を組み合わせる。token mapを元画像へ
戻す最終headは縦横とも4倍するため、32→64へ倍率だけ変更すると幅が半分になる。

ベースラインAでは次を維持する。

- TULIP Swin U-Net encoder/decoder
- 1×4 row-based patch
- 非正方形window (2×8)
- 水平方向circular padding
- skip connection
- range 1 channel

最終headだけを縦2倍・横4倍へ変更する。これは、32→64の垂直2倍と、1×4 patchで
圧縮した水平幅の復元を両立するために必要である。

Waymo幅2650はpatch width 4と3段のpatch mergingに必要な32の倍数ではないため、
右端を6列だけ0 paddingして2656とし、forward後に2650へcropする。lossはcrop後の
GT有効画素だけで計算するのでpadding列は学習へ影響しない。

## Dataset

Waymo32To64Datasetへsignal_modeを追加する。

- range_intensity: 従来の2 channel
- range_only: ベースラインA用の1 channel

range_onlyのshape:

- input: [1,32,2650]
- target: [1,64,2650]

## 1 batch forward

このコマンドは未学習モデルを構築して1 batchだけforwardする。学習は行わない。

```bash
source /home/wakamatsu/ITS/.venv_tulip/bin/activate
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
bash scripts/75_check_range_only_tulip_forward.sh \
  --dataset-role train \
  --device cuda
```

GPUメモリ不足の場合は--device cpuを指定する。出力shapeは[1,1,64,2650]となるべきである。
表示されるlossは未学習のランダム初期値に対するmasked L1であり、性能値ではない。
