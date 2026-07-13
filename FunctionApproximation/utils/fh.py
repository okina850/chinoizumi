import numpy as np

class MultivariateFloaterHormannInterpolator:
    """
    多変数（N次元）テンソル積グリッド上の Floater-Hormann 有理補間
    """
    def __init__(self, points, values, d=3):
        """
        パラメータ
        ----------
        points : tuple of 1D array_like
            各次元のグリッド座標のタプル。例: (x_grid, y_grid)
        values : N-D array_like
            グリッド上の関数値。次元数は points の長さと一致する必要があります。
        d : int or tuple of ints, default: 3
            FH補間の次数パラメータ。次元ごとに異なる次数を指定する場合はタプルを渡します。
        """
        self.points = [np.asarray(p, dtype=float) for p in points]
        self.values = np.asarray(values, dtype=float)
        self.ndim = len(self.points)

        if isinstance(d, int):
            self.d = [d] * self.ndim
        else:
            self.d = d

        # 入力チェック
        if self.values.ndim != self.ndim:
            raise ValueError("valuesの次元数はpointsのタプルの長さと一致する必要があります。")

        for i in range(self.ndim):
            if len(self.points[i]) != self.values.shape[i]:
                raise ValueError(f"次元 {i} のグリッドサイズとvaluesの形状が一致しません。")
            if not (0 <= self.d[i] < len(self.points[i])):
                raise ValueError(f"次数 d[{i}] は 0 <= d < n の範囲である必要があります。")

        # 各次元ごとに1次元のFH重みを事前計算する
        self.weights = [self._compute_fh_weights(self.points[i], self.d[i]) 
                        for i in range(self.ndim)]

    def _compute_fh_weights(self, z, d):
        """SciPyの実装に基づく1次元FH重みの計算"""
        w = np.zeros_like(z)
        n = w.size
        for k in range(n):
            # 重みの計算式（Floater and Hormann 2007）
            for i in range(max(k - d, 0), min(k + 1, n - d)):
                w[k] += 1.0 / np.prod(np.abs(np.delete(z[k] - z[i : i + d + 1], k - i)))
        w *= (-1.0)**(np.arange(n) - d)
        return w

    def _compute_basis(self, Z_eval, x_nodes_interp, w):
        """
        評価点 Z_eval に対する、特定の次元の1次元重心基底関数 Φ_i(Z) を計算する
        """
        # Z_eval: (M,), x_nodes_interp: (N,), w: (N,)
        # 差分行列 (M, N) を作成
        diff = Z_eval[:, np.newaxis] - x_nodes_interp[np.newaxis, :]

        # ゼロ割り警告を一時的に無視して計算
        with np.errstate(divide='ignore', invalid='ignore'):
            C = w[np.newaxis, :] / diff         # (M, N)
            S = np.sum(C, axis=1, keepdims=True) # (M, 1)
            Phi = C / S                         # 基底関数の値 (M, N)

        # 評価点がノードと完全に一致した場合（ゼロ割り発生箇所）の処理
        # ノードと一致した場合は、そのノードの基底関数を1、それ以外を0にする
        matches = (diff == 0)
        has_match = np.any(matches, axis=1)

        if np.any(has_match):
            Phi[has_match] = matches[has_match].astype(Phi.dtype)

        return Phi

    def __call__(self, xi):
        """
        補間値を評価する
        
        パラメータ
        ----------
        xi : tuple of array_like
            評価したい座標。np.meshgrid 等で作成した配列を渡すことができます。
        """
        if len(xi) != self.ndim:
            raise ValueError(f"評価点の次元数が異なります。{self.ndim} 次元必要です。")

        # 評価点を共通の形状にブロードキャスト（meshgrid等に対応するため）
        xi_b = np.broadcast_arrays(*xi)
        target_shape = xi_b[0].shape

        # 全ての評価点を1次元配列 (長さM) に平坦化
        xi_flat = [x.ravel() for x in xi_b]

        # 各次元の基底関数行列 Phi (M, N_k) を計算
        Phis = []
        for i in range(self.ndim):
            Phi = self._compute_basis(xi_flat[i], self.points[i], self.weights[i])
            Phis.append(Phi)

        # ====== ここから下を書き換える ======
        if self.ndim == 2:
            # 2次元の場合は einsum を使わず、明示的な行列積 (Matrix Multiplication) で高速かつ省メモリに計算する
            Phi_x = Phis[0]  # shape: (M, Nx)
            Phi_y = Phis[1]  # shape: (M, Ny)
            
            # self.values (Nx, Ny) を使って計算
            # 1. Phi_x と values を掛け合わせる -> shape: (M, Ny)
            temp = Phi_x @ self.values 
            
            # 2. その結果と Phi_y を要素ごとに掛け合わせて横方向に足し合わせる -> shape: (M,)
            result_flat = np.sum(temp * Phi_y, axis=1)
            
        else:
            # 3次元以上の場合は従来の einsum を使用 (念のため optimize=True を付与)
            phi_subscripts = [f"m{chr(97 + i)}" for i in range(self.ndim)]
            val_subscript = "".join(chr(97 + i) for i in range(self.ndim))
            einsum_str = f"{','.join(phi_subscripts)},{val_subscript}->m"
            result_flat = np.einsum(einsum_str, *Phis, self.values, optimize=True)

        # 元の評価点の形状にリシェイプして返す
        return result_flat.reshape(target_shape)

    
   
if __name__ == "__main__":
    # ==========================================
    # 1. 既知のデータ点（補間の元になる粗いグリッド）を作成
    # ==========================================
    Nx = Ny = 7
    x_nodes = np.linspace(-3, 3, Nx) # X軸の7点
    y_nodes = np.linspace(-3, 3, Ny) # Y軸の7点
    
    
    # meshgridを作成 (indexing='ij' で (x, y) 順の形状にするのがポイント)
    X_nodes, Y_nodes = np.meshgrid(x_nodes, y_nodes, indexing='ij')
    
    # 補間元となる関数値（Z値）を計算
    def true_func(x,y): return np.sin(x) * np.cos(y)
    Z_values = true_func(X_nodes,Y_nodes) 
    
    # ==========================================
    # 2. FH補間器のインスタンスを作成
    # ==========================================
    # points はタプルとして渡す
    points = (x_nodes, y_nodes)
    
    # 今回は各次元の要素数が7なので、次数 d は 0 <= d < 7 の範囲で指定（例: d=3）
    interpolator = MultivariateFloaterHormannInterpolator(points, Z_values, d=3)
    
    # ==========================================
    # 3. 補間したい新しい評価点（細かいグリッド）を作成
    # ==========================================
    x_eval = np.linspace(-3, 3, 50) # 細かく50点
    y_eval = np.linspace(-3, 3, 50)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval, indexing='ij')
    
    # 真の値
    Z_true = true_func(X_eval, Y_eval) # 真の値
    Z_eval = interpolator((X_eval, Y_eval)) # 近似値
    abs_err = np.abs(Z_true - Z_eval)
    max_abs_err = np.max(abs_err)
    
    print("元のデータの形状:", Z_values.shape)
    print("補間後のデータの形状:", Z_eval.shape)
    print(f"最大絶対誤差:{max_abs_err}")
    print(f"最大相対誤差:{max_abs_err/np.max(Z_true)}")
    

    

    

    # ==========================================
    # 5. 3Dプロットによる可視化
    # ==========================================
    import matplotlib.pyplot as plt
    from matplotlib import cm
    fig = plt.figure(figsize=(12, 8))
    ax1= fig.add_subplot(121, projection='3d')

    # 補間された曲面を描画
    surf1 = ax1.plot_surface(X_eval, Y_eval, Z_eval, cmap=cm.viridis,
                           alpha=0.7, antialiased=True, label='FH Interpolation')
    # 元のデータ点を赤い点としてプロット
    ax1.scatter(X_nodes, Y_nodes, Z_values, color='red', s=20, label='Original Nodes', depthshade=False)

    # ラベルとタイトルの設定
    ax1.set_title('3D Floater-Hormann Rational Interpolation')
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z value')
    
    # カラーバーの追加
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    ax1.legend()

    ax2= fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X_eval, Y_eval, abs_err, cmap="viridis",alpha=0.7, antialiased=True, label='Abs err')
    ax2.set_title('Abs err')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('f-r')
    ax2.legend()
    
    plt.show()

