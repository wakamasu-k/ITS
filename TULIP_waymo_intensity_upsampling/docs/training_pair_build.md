# role付き32→64学習ペア

抽出済みのoutputs/training_frames_64_32/index.csvを読み、各フレームから
input_32、target_64、valid maskを持つTULIP学習ペアを生成する。

入力indexのdataset_roleを検証してそのまま継承し、roleの上書きや推測は行わない。
許可するroleはtrain、validation、testのみである。

## dry-run

```bash
source /home/wakamatsu/ITS/.venv_tulip/bin/activate
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
bash scripts/70_build_training_32_to_64_pairs.sh --dry-run
```

## 最小生成

```bash
bash scripts/70_build_training_32_to_64_pairs.sh \
  --max-frames 2 \
  --resume
```

## 240件生成

```bash
bash scripts/70_build_training_32_to_64_pairs.sh --resume
```

生成物は各フレームディレクトリのtulip_pair_32_to_64.npz/jsonと、
outputs/training_pairs_32_to_64/index.csv、failures.csvである。

resume判定ではrole、元split、segment、frame index、timestamp、task、偶数ringの
完全一致検証結果を照合する。
