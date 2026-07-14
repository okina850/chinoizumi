# ニューラルネットワークの活性化関数に有理関数を用いる手法の検証
## Abstract
ニューラルネットワーク(NN)の活性化関数は通常ReLU, tanh, GELUなどが用いられるが、近年、有理関数近似を用いるrational activationが話題である。rational activationは、多層パーセプトロンのパラメータ数を大幅に削減することが期待されている。そこで、既存の活性化関数とrational activationの比較を行うとともに、NNモデルの学習に適切なrational activationの設計について考察する. 
## 項目一覧
- [Rational activationと既存活性化関数の比較実験ノート](./NN_ActHikaku_4sou.ipynb): Franke関数の関数近似をベンチマークに、rational activationと既存の活性化関数で、近似精度の比較を行った. 
- [Rational activationの極の発生回避手法](./RatAct_avoid_poles.ipynb): [Rational activationと既存活性化関数の比較実験ノート](./NN_ActHikaku_4sou.ipynb) の結果に沿って、極の発生を回避する方法を考察し、実験を行った




## 参考文献
- 