# GT rangeとTULIP rangeの分離評価

次のカメラ投影へ進む前に、range誤差とintensity誤差を分離する。

比較する2条件:

- `oracle_range`: GT range + TULIP intensity
- `predicted_range`: TULIP range + TULIP intensity

両条件で同じ生成odd rowと共通valid pixelを使用する。

`oracle_range`の3D誤差は理論上ほぼ0となるため、intensity推定そのものを確認できる。
`predicted_range`との差分は、range推定が点群位置・カメラ投影へ与えるgeometry penaltyとして解釈する。

出力:

- `summary.json`
- `per_frame.csv`

この診断の後、range誤差の影響を把握した上でWaymo camera calibrationを使った投影へ進む。

作成日: 2026-08-20
