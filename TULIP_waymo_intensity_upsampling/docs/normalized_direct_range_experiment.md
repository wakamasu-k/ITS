# 正規化付き直接range回帰

TULIPの直接range回帰を維持し、公式実装に合わせて入力とGTを120 mで
正規化します。checkpointには `range_scale=120` を保存し、正式評価では
自動的に読み取って予測をメートルへ戻します。補間ベースラインは最初から
メートル単位なので、CSVと画像はすべて同じ単位で比較されます。

## 1. 2フレームoverfit

```bash
bash scripts/82_overfit_normalized_range_tulip.sh \
  --max-train-samples 2 --epochs 20 --batch-size 1 --device cuda
```

続行する場合は最終epochを指定して `--resume` を付けます。

```bash
bash scripts/82_overfit_normalized_range_tulip.sh \
  --max-train-samples 2 --epochs 300 --batch-size 1 --device cuda --resume
```

旧メートル学習checkpointとは出力先を分離しています。

## 2. 保存checkpointのtrain 2フレーム評価

```bash
bash scripts/92_evaluate_normalized_range_tulip.sh \
  --dataset-role train --max-frames 2 --device cuda
```

`summary.json` の `range_scale_m` が120、`model_training` がfalseであることを
確認します。この結果を確認するまでは8フレームや160フレームへ進みません。
