## 機械学習・AI
- [感情分析](https://github.com/okina850/sentiment_analysis_stat)
  - **概要:** Hugging Face (Transformers) を用いた感情分析推論パイプラインと、推論結果に対する基本統計解析（t検定）の実装サンプル
  - **使用技術:** 自然言語処理, Hugging Face(Transformer/BERT), 統計学,Pandas
- [2変数関数近似](https://github.com/okina850/kan_siren_rbf__vs__fh)
  - **概要:** 2変数関数の近似タスクに対して、KAN, MLP, ガウス過程回帰, の3手法を用いた手法の比較, およびそれら3手法と有理関数補間法の比較実験
  - **使用技術:** KAN,MLP,ガウス過程回帰,Pytorch, Optuna
  - **成果/知見:** ガウス過程回帰は有理関数補間に迫る精度を達成したが、KANおよびMLPは2変数関数の近似タスクに関しては限界があることを実証
- [有理関数を用いた活性化関数の設計](https://github.com/okina850/rational_activation_small-note)
  - **概要:** 活性化関数に有理関数を用いた「Rational activation」によるニューラルネットワークモデルの実験ノート
  - **使用技術:** Pytorch, Optuna
- [NNを用いた2変数関数の近似](https://github.com/okina850/NN_inverse_transform_sampling_test)
  - **概要:** 2変数関数の近似タスクにおける最適なNNモデルの設計について調査・実験
  - **使用技術:** Pytorch
- [ブラック=ショールズ方程式をPyTorchで解く(工事中)](./bs-pytorch-pricing)
## 関数近似
- [ルンゲ関数の近似における3種の補間法の比較](https://github.com/okina850/interpolation-benchmark)
  -  **概要:** 急勾配を持つルンゲ関数に対する補間法（Floater-Hormann法, Bicubic Spline, Chebyshev polynomials）を用いた近似モデルの評価プロファイリング
  -  **使用技術:** Python, NumPy, SciPy, Matplotlib
  -  **成果/知見:** Floater-Hormann法により、Bicubic Spline と比較して補間ノード数（$\sim$モデルパラメータ数）を約74%削減できることを実証
- [Franke関数の近似精度の比較](https://github.com/okina850/gpr_bicubicspline_chebypoly__vs__fh)
  -  **概要:** Franke関数に対する各種関数近似手法、Floater-Hormann法, Bicubic Spline, Chebyshev polynomials、ガウス過程回帰、を用いた近似モデルの精度比較
  -  **使用技術:** Python, NumPy, SciPy, Pytorch, Optuna
  -  **成果/知見:** Chebyshev点配置を用いた場合、Chebyshev多項式が最大絶対誤差$\sim 10^{-12}$という圧倒的な精度を達成

