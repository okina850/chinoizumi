
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import FloaterHormannInterpolator

class FloaterHormannMinimax1D:
    def __init__(self, f, x_min, x_max,n_nodes, init_nodes=None, d=3, tol=1e-6, max_iter=100):

        self.f = f
        self.x_min = x_min
        self.x_max = x_max
        self.n_nodes = n_nodes
        self.d = d
        self.tol = tol
        self.max_iter = max_iter
        
        # 初期ノード (チェビシェフ配置)
        k = np.arange(1, n_nodes + 1)
        u_k = np.cos((2 * (n_nodes - k) + 1) / (2 * n_nodes) * np.pi)
        self.nodes = init_nodes if init_nodes is not None else 0.5 * (x_min + x_max) + 0.5 * (x_max - x_min) * u_k

        self.approx = None

        self.local_extrema = []
        self.max_history = [] 
        self.min_history = [] 


    def _find_extrema(self, r):
        extrema_x = np.zeros(self.n_nodes + 1)
        extrema_y = np.zeros(self.n_nodes + 1)
        
        # 区間の設定: [x_min, x_1], (x_1, x_2), ..., [x_N, x_max]
        bounds = [self.x_min] + list(self.nodes) + [self.x_max]
        
        for i in range(self.n_nodes + 1):
            # 各区間で |r(x) - f(x)| の最大化 (= -|r(x) - f(x)| の最小化)
            res = minimize_scalar(
                lambda x: -np.abs(r(x) - self.f(x)),
                bounds=(bounds[i], bounds[i+1]),
                method='bounded'
            )
            extrema_x[i] = res.x
            extrema_y[i] = r(res.x) - self.f(res.x) # 符号付き誤差を保存
            
        return extrema_x, extrema_y

    def optimize(self, verbose=True):
        eps = 1e-14 # ゼロ除算防止用の微小値

        for iteration in range(self.max_iter):
            # 1. 現在のノードでFH補間
            r = FloaterHormannInterpolator(self.nodes, self.f(self.nodes), d=self.d)
            
            # 2. 誤差の極値を探索
            ex_x, ex_y = self._find_extrema(r)
            
            # 3. ダミー区間（幅0）を除外して有効な極値だけを抽出
            bounds = [self.x_min] + list(self.nodes) + [self.x_max]
            valid_indices = [i for i in range(self.n_nodes + 1) if bounds[i+1] - bounds[i] > 1e-12]
            valid_ex_y = [ex_y[i] for i in valid_indices]
            
            # 4. ログと履歴の記録（有効な極値のみを使用）
            max_f_abs = np.max(np.abs(self.f(ex_x)))
            abs_valid_ex_y = np.abs(valid_ex_y)
            
            max_abs_err = np.max(abs_valid_ex_y)
            min_abs_err = np.min(abs_valid_ex_y)
            
            max_rel_err = max_abs_err / max_f_abs
            min_rel_err = min_abs_err / max_f_abs
            
            self.max_history.append(max_rel_err)
            self.min_history.append(min_rel_err)

            if verbose:
                print(f"iter {iteration}: Max Rel Err={max_rel_err}")
            
            # 5. 収束判定
            if max_abs_err - min_abs_err < self.tol:
                print(f"Converged at iteration {iteration}")
                return r  # 収束した時点の関数を返す
                
            # 6. ノードの更新 (Acklamの更新則)
            new_nodes = np.copy(self.nodes)
            for k in range(self.n_nodes):
                # 両端に固定されたノードは動かさない（ピン留め）
                if np.isclose(self.nodes[k], self.x_min) or np.isclose(self.nodes[k], self.x_max):
                    continue
                
                # 内側のノードのみ移動
                c1 = (self.nodes[k] - ex_x[k]) / (np.abs(ex_y[k]) + eps)
                c2 = (ex_x[k+1] - self.nodes[k]) / (np.abs(ex_y[k+1]) + eps)
                new_nodes[k] = ex_x[k] + c1 * (ex_x[k+1] - ex_x[k]) / (c1 + c2)
                
            self.nodes = new_nodes
            
        # 最大ループ到達時は、最後のイテレーションで評価・ログ出力した関数を返す
        return r
    
