# ニューラルネットワークの活性化関数に有理関数を用いる手法の検証
## Abstract
ニューラルネットワーク(NN)の活性化関数は通常ReLU, tanh, GELUなどが用いられるが、近年、有理関数近似を用いるrational activationが話題である。rational activationは、多層パーセプトロンのパラメータ数を大幅に削減することが期待されている。そこで、既存の活性化関数とrational activationの比較を行うとともに、NNモデルの学習に適切なrational activationの設計について考察する. Franke関数の近似をタスクとする.
## 項目一覧
- [Rational activationと既存活性化関数の比較実験ノート](./NN_ActHikaku_4sou.ipynb):
    Franke関数の関数近似をベンチマークに、rational activationと既存の活性化関数で、近似精度の比較を行った. 
- [Rational activationの極の発生回避手法](./RatAct_avoid_poles.ipynb): 
    [Rational activationと既存活性化関数の比較実験ノート](./NN_ActHikaku_4sou.ipynb) の結果に沿って、極の発生を回避する方法を考察し、実験を行った
- [初期値設定法の比較](./defaultAct_ReLU_vs_LReLU.ipynb):
    有理関数による活性化関数の初期値を、ReLU関数に近似する手法とLeaky ReLUに近似する手法を試し、比較を行った
- [深層へのスケール](./Nak_3k_deep.ipynb)
   Nakatsukasa etal. に準拠し、有理関数の次数を$(3^k, 3^k - 1)$にした場合に深層NNにスケールする様子を確認した
- [浅く広い学習の検証](./shallow_wide.ipynb)
    [深層へのスケール](./Nak_3k_deep.ipynb)への結果を受けて、Franke関数の関数近似に浅く広いNNを利用する実験を行った. Optunaによるハイパーパラメータ最適化を利用した.





## 参考文献
- [Ratinal neural networks](https://proceedings.neurips.cc/paper/2020/file/a3f390d88e4c41f2747bfa2f1b5f87db-Paper.pdf)
- [Padé Activation Units: End-to-end Learning of Flexible Activation Functions in Deep Networks](https://ml-research.github.io/papers/molina2020iclr_pau.pdf)