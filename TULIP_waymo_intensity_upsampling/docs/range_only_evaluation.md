# Range-only TULIP checkpointの正式評価

保存checkpointを読み込み、`model.eval()` と `torch.no_grad()` で重みを固定します。
同じフレーム、同じ64-line GT、同じ `generated_mask`（入力にない奇数行の有効画素）で
TULIP、nearest-neighbor、linear補間を比較します。追加学習は行いません。

```bash
source /home/wakamatsu/ITS/.venv_tulip/bin/activate
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
bash scripts/90_evaluate_range_only_tulip.sh \
  --dataset-role validation --max-frames 2 --device cuda
```

`outputs/range_only_evaluation/` に以下を保存します。

- `summary.json`: checkpoint epoch、各手法の生成行L1、TULIP/linear比、負値総数
- `per_frame.csv`: segment、frame、timestamp、全L1、予測rangeの最小・最大・平均・負値数
- `images/frame_XXXX.png`: GT、TULIP予測、TULIP誤差、linear誤差

主判定値は `generated_l1` です。まず2フレームで動作と画像を確認し、その後
`--max-frames 40` でvalidation全体を評価します。正式評価が終わるまで追加学習は開始しません。
