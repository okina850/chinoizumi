from utils.fh import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from joblib import Parallel, delayed
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class FHMinimax2D:
    def __init__(self, target_func, x_bounds, y_bounds, Nx, Ny, d=3, init_nodes=None, error_metric='absolute', search_method='grid', grid_res=15):
        """
        Floater-Hormann法に基づく2次元有理ミニマックス最適化クラス
        p, q (次数)の代わりに d (FHブレンド次数)を使用します。
        """
        self.search_method = search_method
        self.grid_res = grid_res
            
        self.target_func = target_func
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.Nx = Nx
        self.Ny = Ny
        self.d = d  # FH法の次数パラメータ（通常 3 程度が安定します）
        self.error_metric = error_metric
        
        print(f"Nodes: {Nx * Ny} (Nx={Nx}, Ny={Ny}), FH Degree: d={d}")
        print("Mode: Barycentric Floater-Hormann Interpolation (No Linear Solvers)")

        if init_nodes is not None:
            self.x_nodes, self.y_nodes = init_nodes
        else:
            # デフォルトは等間隔
            self.x_nodes = np.linspace(x_bounds[0], x_bounds[1], Nx + 2)[1:-1]
            self.y_nodes = np.linspace(y_bounds[0], y_bounds[1], Ny + 2)[1:-1]
        
        self.fh_interpolator = None
        self.approx_func = None 
        
        self.ideal_dx = np.zeros((Nx, Ny))
        self.ideal_dy = np.zeros((Nx, Ny))
        self.actual_dx = np.zeros(Nx)
        self.actual_dy = np.zeros(Ny)

        self.local_extrema = []
        self.max_history = [] 
        self.min_history = [] 
        self.rel_history = []
        self.max_loc_history = []
        self.min_loc_history = []
        self.rel_linf_history = []
        self.naive_rel_history = []
        
        

    def fit_rational_function(self):
        """
        FH法で直接補間器を構築する（爆速・クラッシュなし）
        """
        # 現在のノード位置での真値 (Z) を計算
        XX, YY = np.meshgrid(self.x_nodes, self.y_nodes, indexing='ij')
        Z_values = self.target_func(XX, YY)
        
        # FH補間器のインスタンスを作成（これ自体が補間関数になります）
        self.fh_interpolator = MultivariateFloaterHormannInterpolator(
            points=(self.x_nodes, self.y_nodes), 
            values=Z_values, 
            d=self.d 
        )
        
        # 評価用のラッパー関数（スカラーでも配列でも受け取れるようにする）
        def approx(x, y):
            x_arr = np.atleast_1d(x)
            y_arr = np.atleast_1d(y)
            res = self.fh_interpolator((x_arr, y_arr))
            # minimize関数からのスカラー入力に対応
            if np.isscalar(x) and np.isscalar(y):
                return res[0]
            return res
            
        self.approx_func = approx

    def display_max_err(self,n_eval,graph=False,rel=False):
        x = np.linspace(self.x_bounds[0],self.x_bounds[1],n_eval)
        y = np.linspace(self.y_bounds[0],self.y_bounds[1],n_eval)
        x_grid, y_grid = np.meshgrid(
                x,
                y
            )
        #f_val = self.target_func(x, y)
        #approx_val = self.approx_func(x, y)
        f_val = self.target_func(x_grid, y_grid)
        approx_val = self.approx_func(x_grid, y_grid)
        abs_err = np.abs(approx_val - f_val)
        max_abs_err = np.max(abs_err)
        max_f_val = np.max(f_val)
        rel_err= abs_err / max_f_val
        max_rel_err = max_abs_err / max_f_val
        print(f"Max Abs Err:{max_abs_err}| Max rel err:{max_rel_err}")
        if graph:
            fig_plotly = make_subplots(
                rows = 1, cols = 1,
                specs = [
                    [{'type':'surface'}]
                ],
                subplot_titles=(
                    ['Absolute Error']
                ) if not rel else ['Relative Error'] ,
                vertical_spacing=0.05
            )

            fig_plotly.add_trace(
                go.Surface(
                    x=x_grid,
                    y=y_grid,
                    z=abs_err if not rel else rel_err
                ),
                row=1,col=1
            )
            fig_plotly.show()



                

    def error_func(self, x, y):
        f_val = self.target_func(x, y)
        approx_val = self.approx_func(x, y)
        abs_err = np.abs(approx_val - f_val)
        
        if self.error_metric == 'relative':
            return abs_err / (np.abs(f_val))
        return abs_err

    @staticmethod
    def process_cell_parallel(i, j, x_ext, y_ext, error_func, search_method, grid_res):
        qx_min, qx_max = x_ext[i], x_ext[i+1]
        qy_min, qy_max = y_ext[j], y_ext[j+1]
        
        if search_method == 'grid':
            # --- 超高速グリッドサーチ ---
            gx = np.linspace(qx_min, qx_max, grid_res)
            gy = np.linspace(qy_min, qy_max, grid_res)
            GX, GY = np.meshgrid(gx, gy, indexing='ij')
            
            err_vals = error_func(GX, GY) # ベクトル化で一括計算
            
            max_idx = np.unravel_index(np.argmax(err_vals, axis=None), err_vals.shape)
            max_x, max_y = gx[max_idx[0]], gy[max_idx[1]]
            max_err = err_vals[max_idx]
        else:
            # --- 従来の L-BFGS-B 探索 ---
            x0, y0 = (qx_min + qx_max) / 2.0, (qy_min + qy_max) / 2.0
            res = minimize(lambda p: -error_func(p[0], p[1]), 
                        x0=[x0, y0], bounds=[(qx_min, qx_max), (qy_min, qy_max)],
                        method='L-BFGS-B', tol=1e-5)
            max_x, max_y, max_err = res.x[0], res.x[1], -res.fun
            
        return i, j, max_x, max_y, max_err
    # x方向、y方向の移動を決める
    def calculate_displacements(self):
            err_f = self.error_func
            
            # 境界を含めた座標配列を作成 (長さ Nx+2, Ny+2)
            x_ext = np.concatenate(([self.x_bounds[0]], self.x_nodes, [self.x_bounds[1]]))
            y_ext = np.concatenate(([self.y_bounds[0]], self.y_nodes, [self.y_bounds[1]]))
            
            # 1. 全てのセルについて並列で極値探索 ( (Nx+1)*(Ny+1) 回 )
            # 例: 20x20ノードなら 21x21=441セル
            cell_results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(FHMinimax2D.process_cell_parallel)(
                i, j, x_ext, y_ext, err_f, self.search_method, self.grid_res
            )
            for i in range(self.Nx + 1) for j in range(self.Ny + 1)
            )
            
            # 2. セルの結果を2次元配列に整理
            cell_data = {} # (i, j) -> (max_x, max_y, max_err)
            self.local_extrema = []
            for i, j, mx, my, me in cell_results:
                cell_data[(i, j)] = (mx, my, me)
                self.local_extrema.append(me)

            # 3. 各ノードに対して、周囲4つのセルの結果を配分する
            for i in range(self.Nx):
                for j in range(self.Ny):
                    x_i, y_j = self.x_nodes[i], self.y_nodes[j]
                    
                    # ノード(i,j)を囲む4つのセルインデックス
                    # x_ext[i+1], y_ext[j+1] が現在のノード座標
                    surrounding_cells = [
                        (i+1, j+1), # 右上 (第1象限)
                        (i, j+1),   # 左上 (第2象限)
                        (i, j),     # 左下 (第3象限)
                        (i+1, j)    # 右下 (第4象限)
                    ]
                    
                    quad_data = []
                    sum_t = 0.0
                    
                    for c_idx in surrounding_cells:
                        max_x, max_y, max_err = cell_data[c_idx]
                        dx = max_x - x_i
                        dy = max_y - y_j
                        dist = np.hypot(dx, dy)
                        t = max_err / dist if dist > 1e-10 else 0.0
                        quad_data.append((t, dx, dy))
                        sum_t += t
                    
                    delta_x_star, delta_y_star = 0.0, 0.0
                    if sum_t > 0:
                        for (t, dx, dy) in quad_data:
                            delta_x_star += (t / sum_t) * dx
                            delta_y_star += (t / sum_t) * dy
                            
                    self.ideal_dx[i, j] = delta_x_star
                    self.ideal_dy[i, j] = delta_y_star
                    
            # 平均化
            self.actual_dx = np.mean(self.ideal_dx, axis=1)
            self.actual_dy = np.mean(self.ideal_dy, axis=0)

            # --- 追加：境界ノードの変位をゼロに固定（ピン留め） ---
            self.actual_dx[0] = 0.0
            self.actual_dx[-1] = 0.0
            self.actual_dy[0] = 0.0
            self.actual_dy[-1] = 0.0

            return self.actual_dx, self.actual_dy

    def update_nodes(self, learning_rate=1.0):
        self.x_nodes += self.actual_dx * learning_rate
        self.y_nodes += self.actual_dy * learning_rate

        # 順序の逆転（交差）防止
        self.x_nodes = np.sort(self.x_nodes)
        self.y_nodes = np.sort(self.y_nodes)

    def plot_grid_and_errors(self, iteration):
        X, Y = np.meshgrid(np.linspace(self.x_bounds[0], self.x_bounds[1], 100),
                           np.linspace(self.y_bounds[0], self.y_bounds[1], 100))
        Z = self.error_func(X, Y)

        fig, ax = plt.subplots(figsize=(9, 7))
        c = ax.pcolormesh(X, Y, Z, shading='auto', cmap='Reds', alpha=0.5)
        fig.colorbar(c, ax=ax, label='Error')
        
        for xi in self.x_nodes: ax.axvline(xi, color='gray', linestyle='-', alpha=0.4)
        for yj in self.y_nodes: ax.axhline(yj, color='gray', linestyle='-', alpha=0.4)
            
        XX, YY = np.meshgrid(self.x_nodes, self.y_nodes, indexing='ij')
        ax.scatter(XX, YY, color='black', s=40, zorder=5, label='Nodes')
        ax.quiver(XX, YY, self.ideal_dx, self.ideal_dy, color='blue', alpha=0.3, angles='xy', scale_units='xy', scale=1, label='Ideal Pull')
        
        actual_dx_grid, actual_dy_grid = np.meshgrid(self.actual_dx, self.actual_dy, indexing='ij')
        ax.quiver(XX, YY, actual_dx_grid, actual_dy_grid, color='red', angles='xy', scale_units='xy', scale=1, label='Actual Move')
        
        max_err = np.max(Z)
        ax.set_title(f'Iteration {iteration} | Max Error: {max_err:.2e}')
        ax.set_xlim(self.x_bounds)
        ax.set_ylim(self.y_bounds)
        ax.legend(loc='upper right')
        plt.show()

    def add_refinement_axes(self, target_x, target_y):
        """
        指定された座標に新しい補間軸を追加する（重複回避の安全装置付き）
        """
        # 既存ノードとの距離をチェック（近すぎるとFH法のゼロ割りでNaNになる）
        min_dist_x = np.min(np.abs(self.x_nodes - target_x))
        min_dist_y = np.min(np.abs(self.y_nodes - target_y))
        
        # 距離が 1e-5 以下なら、微小値(1e-4)を足して衝突を回避する
        if min_dist_x < 1e-5:
            target_x += 1e-4
        if min_dist_y < 1e-5:
            target_y += 1e-4

        # 既存のノード配列に新しい座標を挿入してソート
        self.x_nodes = np.sort(np.append(self.x_nodes, target_x))
        self.y_nodes = np.sort(np.append(self.y_nodes, target_y))
        
        self.Nx = len(self.x_nodes)
        self.Ny = len(self.y_nodes)
        
        self.ideal_dx = np.zeros((self.Nx, self.Ny))
        self.ideal_dy = np.zeros((self.Nx, self.Ny))
        
        print(f"[*] 軸を追加しました: 新しいグリッドサイズ {self.Nx} x {self.Ny}")
    
    def optimize(self, num_iterations=12, learning_rate=1.0, verbose=True, plot=False, eval_res=100):
        # 履歴をリセット
        self.max_history = []
        self.min_history = []
        self.rel_history = []
        self.max_loc_history = []  
        
        # 評価グリッドと真値の準備
        x_eval = np.linspace(self.x_bounds[0], self.x_bounds[1], eval_res)
        y_eval = np.linspace(self.y_bounds[0], self.y_bounds[1], eval_res)
        X_eval, Y_eval = np.meshgrid(x_eval, y_eval, indexing='ij')
        
        f_val = self.target_func(X_eval, Y_eval)
        f_max_abs = np.max(np.abs(f_val)) + 1e-15

        # 【新設】クラス変数としてベストな状態を保持
        self.best_max_err = float('inf')
        best_x_nodes = None
        best_y_nodes = None

        # 【修正】record_history フラグを追加
        def evaluate_and_prepare_next(record_history=True):
            approx_val = self.approx_func(X_eval, Y_eval)
            abs_err = np.abs(approx_val - f_val)
            
            max_idx = np.unravel_index(np.argmax(abs_err, axis=None), abs_err.shape)
            global_max_abs = abs_err[max_idx]
            global_max_rel = global_max_abs / f_max_abs
            max_x = X_eval[max_idx]
            max_y = Y_eval[max_idx]
            
            self.calculate_displacements()
            valid_peaks = [p for p in self.local_extrema if p > 1e-16]
            min_peak = np.min(valid_peaks) if len(valid_peaks) > 0 else 0.0
            
            # フラグがTrueのときのみ履歴配列に追加する
            if record_history:
                self.max_history.append(global_max_abs)
                self.min_history.append(min_peak)
                self.rel_history.append(global_max_rel)
                self.max_loc_history.append((max_x, max_y)) 
            
            return global_max_abs, min_peak, global_max_rel, max_x, max_y

        # --- 初期状態 (Iter 0) ---
        self.fit_rational_function()
        cur_max, cur_min, cur_rel, cur_x, cur_y = evaluate_and_prepare_next(record_history=True)
        
        self.best_max_err = cur_max
        best_x_nodes = self.x_nodes.copy()
        best_y_nodes = self.y_nodes.copy()
        
        if verbose:
            print(f"--- Initial (Iter 0) --- LR: {learning_rate:.2f}")
            print(f"  Max Abs: {cur_max:.6e} at (x={cur_x:.3f}, y={cur_y:.3f}) | Min Peak: {cur_min:.6e} | Rel: {cur_rel:.6e}")
        if plot:
            self.plot_grid_and_errors(iteration=0)

        # --- 最適化ループ ---
        for it in range(1, num_iterations + 1):
            self.update_nodes(learning_rate)
            self.fit_rational_function()
            cur_max, cur_min, cur_rel, cur_x, cur_y = evaluate_and_prepare_next(record_history=True)

            # 過去のベストより良ければセーブ
            if cur_max < self.best_max_err:
                self.best_max_err = cur_max
                best_x_nodes = self.x_nodes.copy()
                best_y_nodes = self.y_nodes.copy()

            if verbose:
                print(f"--- Iteration {it} --- LR: {learning_rate:.2f}")
                print(f"  Max Abs: {cur_max:.6e} at (x={cur_x:.3f}, y={cur_y:.3f}) | Min Peak: {cur_min:.6e} | Rel: {cur_rel:.6e}")
            if plot:
                self.plot_grid_and_errors(iteration=it)

        # ==========================================
        # ループ終了後、最強の陣形に巻き戻す
        # ==========================================
        if verbose:
            print(f"  => Restoring to Best State (Min Max Err: {self.best_max_err:.6e})")
        
        self.x_nodes = best_x_nodes.copy()
        self.y_nodes = best_y_nodes.copy()
        self.fit_rational_function()
        
        # 【重要】履歴を汚さずに、クラス内部の状態（次の移動ベクトル等）だけを更新する
        evaluate_and_prepare_next(record_history=False)

class FHMinimax2D_Add_Axes:
    def __init__(self, target_func, x_bounds, y_bounds, Nx, Ny, d=3, init_nodes=None, error_metric='absolute', search_method='grid', grid_res=15):
        """
        Floater-Hormann法に基づく2次元有理ミニマックス最適化クラス
        p, q (次数)の代わりに d (FHブレンド次数)を使用します。
        """
        self.search_method = search_method
        self.grid_res = grid_res
            
        self.target_func = target_func
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.Nx = Nx
        self.Ny = Ny
        self.d = d  # FH法の次数パラメータ（通常 3 程度が安定します）
        self.error_metric = error_metric
        
        print(f"Nodes: {Nx * Ny} (Nx={Nx}, Ny={Ny}), FH Degree: d={d}")
        print("Mode: Barycentric Floater-Hormann Interpolation (No Linear Solvers)")

        if init_nodes is not None:
            self.x_nodes, self.y_nodes = init_nodes
        else:
            # デフォルトは等間隔
            self.x_nodes = np.linspace(x_bounds[0], x_bounds[1], Nx + 2)[1:-1]
            self.y_nodes = np.linspace(y_bounds[0], y_bounds[1], Ny + 2)[1:-1]
        
        self.fh_interpolator = None
        self.approx_func = None 
        
        self.ideal_dx = np.zeros((Nx, Ny))
        self.ideal_dy = np.zeros((Nx, Ny))
        self.actual_dx = np.zeros(Nx)
        self.actual_dy = np.zeros(Ny)

        self.local_extrema = []
        self.max_history = [] 
        self.min_history = [] 
        self.rel_history = []
        self.max_loc_history = []
        self.min_loc_history = []
        self.rel_linf_history = []
        self.naive_rel_history = []
        
        

    def fit_rational_function(self):
        """
        FH法で直接補間器を構築する（爆速・クラッシュなし）
        """
        # 現在のノード位置での真値 (Z) を計算
        XX, YY = np.meshgrid(self.x_nodes, self.y_nodes, indexing='ij')
        Z_values = self.target_func(XX, YY)
        
        # FH補間器のインスタンスを作成（これ自体が補間関数になります）
        self.fh_interpolator = MultivariateFloaterHormannInterpolator(
            points=(self.x_nodes, self.y_nodes), 
            values=Z_values, 
            d=self.d 
        )
        
        # 評価用のラッパー関数（スカラーでも配列でも受け取れるようにする）
        def approx(x, y):
            x_arr = np.atleast_1d(x)
            y_arr = np.atleast_1d(y)
            res = self.fh_interpolator((x_arr, y_arr))
            # minimize関数からのスカラー入力に対応
            if np.isscalar(x) and np.isscalar(y):
                return res[0]
            return res
            
        self.approx_func = approx

    def display_max_err(self,n_eval,graph=False,rel=False):
        x = np.linspace(self.x_bounds[0],self.x_bounds[1],n_eval)
        y = np.linspace(self.y_bounds[0],self.y_bounds[1],n_eval)
        x_grid, y_grid = np.meshgrid(
                x,
                y
            )
        #f_val = self.target_func(x, y)
        #approx_val = self.approx_func(x, y)
        f_val = self.target_func(x_grid, y_grid)
        approx_val = self.approx_func(x_grid, y_grid)
        abs_err = np.abs(approx_val - f_val)
        max_abs_err = np.max(abs_err)
        max_f_val = np.max(f_val)
        rel_err= abs_err / max_f_val
        max_rel_err = max_abs_err / max_f_val
        print(f"Max Abs Err:{max_abs_err}| Max rel err:{max_rel_err}")
        if graph:
            fig_plotly = make_subplots(
                rows = 1, cols = 1,
                specs = [
                    [{'type':'surface'}]
                ],
                subplot_titles=(
                    ['Absolute Error']
                ) if not rel else ['Relative Error'] ,
                vertical_spacing=0.05
            )

            fig_plotly.add_trace(
                go.Surface(
                    x=x_grid,
                    y=y_grid,
                    z=abs_err if not rel else rel_err
                ),
                row=1,col=1
            )
            fig_plotly.show()



                

    def error_func(self, x, y):
        f_val = self.target_func(x, y)
        approx_val = self.approx_func(x, y)
        abs_err = np.abs(approx_val - f_val)
        
        if self.error_metric == 'relative':
            return abs_err / (np.abs(f_val))
        return abs_err

    @staticmethod
    def process_cell_parallel(i, j, x_ext, y_ext, error_func, search_method, grid_res):
        qx_min, qx_max = x_ext[i], x_ext[i+1]
        qy_min, qy_max = y_ext[j], y_ext[j+1]
        
        if search_method == 'grid':
            # --- 超高速グリッドサーチ ---
            gx = np.linspace(qx_min, qx_max, grid_res)
            gy = np.linspace(qy_min, qy_max, grid_res)
            GX, GY = np.meshgrid(gx, gy, indexing='ij')
            
            err_vals = error_func(GX, GY) # ベクトル化で一括計算
            
            max_idx = np.unravel_index(np.argmax(err_vals, axis=None), err_vals.shape)
            max_x, max_y = gx[max_idx[0]], gy[max_idx[1]]
            max_err = err_vals[max_idx]
        else:
            # --- 従来の L-BFGS-B 探索 ---
            x0, y0 = (qx_min + qx_max) / 2.0, (qy_min + qy_max) / 2.0
            res = minimize(lambda p: -error_func(p[0], p[1]), 
                        x0=[x0, y0], bounds=[(qx_min, qx_max), (qy_min, qy_max)],
                        method='L-BFGS-B', tol=1e-5)
            max_x, max_y, max_err = res.x[0], res.x[1], -res.fun
            
        return i, j, max_x, max_y, max_err
    # x方向、y方向の移動を決める
    def calculate_displacements(self):
            err_f = self.error_func
            
            # 境界を含めた座標配列を作成 (長さ Nx+2, Ny+2)
            x_ext = np.concatenate(([self.x_bounds[0]], self.x_nodes, [self.x_bounds[1]]))
            y_ext = np.concatenate(([self.y_bounds[0]], self.y_nodes, [self.y_bounds[1]]))
            
            # 1. 全てのセルについて並列で極値探索 ( (Nx+1)*(Ny+1) 回 )
            # 例: 20x20ノードなら 21x21=441セル
            cell_results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(FHMinimax2D.process_cell_parallel)(
                i, j, x_ext, y_ext, err_f, self.search_method, self.grid_res
            )
            for i in range(self.Nx + 1) for j in range(self.Ny + 1)
            )
            
            # 2. セルの結果を2次元配列に整理
            cell_data = {} # (i, j) -> (max_x, max_y, max_err)
            self.local_extrema = []
            for i, j, mx, my, me in cell_results:
                cell_data[(i, j)] = (mx, my, me)
                self.local_extrema.append(me)

            # 3. 各ノードに対して、周囲4つのセルの結果を配分する
            for i in range(self.Nx):
                for j in range(self.Ny):
                    x_i, y_j = self.x_nodes[i], self.y_nodes[j]
                    
                    # ノード(i,j)を囲む4つのセルインデックス
                    # x_ext[i+1], y_ext[j+1] が現在のノード座標
                    surrounding_cells = [
                        (i+1, j+1), # 右上 (第1象限)
                        (i, j+1),   # 左上 (第2象限)
                        (i, j),     # 左下 (第3象限)
                        (i+1, j)    # 右下 (第4象限)
                    ]
                    
                    quad_data = []
                    sum_t = 0.0
                    
                    for c_idx in surrounding_cells:
                        max_x, max_y, max_err = cell_data[c_idx]
                        dx = max_x - x_i
                        dy = max_y - y_j
                        dist = np.hypot(dx, dy)
                        t = max_err / dist if dist > 1e-10 else 0.0
                        quad_data.append((t, dx, dy))
                        sum_t += t
                    
                    delta_x_star, delta_y_star = 0.0, 0.0
                    if sum_t > 0:
                        for (t, dx, dy) in quad_data:
                            delta_x_star += (t / sum_t) * dx
                            delta_y_star += (t / sum_t) * dy
                            
                    self.ideal_dx[i, j] = delta_x_star
                    self.ideal_dy[i, j] = delta_y_star
                    
            # 平均化
            self.actual_dx = np.mean(self.ideal_dx, axis=1)
            self.actual_dy = np.mean(self.ideal_dy, axis=0)

            # --- 追加：境界ノードの変位をゼロに固定（ピン留め） ---
            self.actual_dx[0] = 0.0
            self.actual_dx[-1] = 0.0
            self.actual_dy[0] = 0.0
            self.actual_dy[-1] = 0.0

            return self.actual_dx, self.actual_dy

    def update_nodes(self, learning_rate=1.0):
        self.x_nodes += self.actual_dx * learning_rate
        self.y_nodes += self.actual_dy * learning_rate

        # 順序の逆転（交差）防止
        self.x_nodes = np.sort(self.x_nodes)
        self.y_nodes = np.sort(self.y_nodes)

    def plot_grid_and_errors(self, iteration):
        X, Y = np.meshgrid(np.linspace(self.x_bounds[0], self.x_bounds[1], 100),
                           np.linspace(self.y_bounds[0], self.y_bounds[1], 100))
        Z = self.error_func(X, Y)

        fig, ax = plt.subplots(figsize=(9, 7))
        c = ax.pcolormesh(X, Y, Z, shading='auto', cmap='Reds', alpha=0.5)
        fig.colorbar(c, ax=ax, label='Error')
        
        for xi in self.x_nodes: ax.axvline(xi, color='gray', linestyle='-', alpha=0.4)
        for yj in self.y_nodes: ax.axhline(yj, color='gray', linestyle='-', alpha=0.4)
            
        XX, YY = np.meshgrid(self.x_nodes, self.y_nodes, indexing='ij')
        ax.scatter(XX, YY, color='black', s=40, zorder=5, label='Nodes')
        ax.quiver(XX, YY, self.ideal_dx, self.ideal_dy, color='blue', alpha=0.3, angles='xy', scale_units='xy', scale=1, label='Ideal Pull')
        
        actual_dx_grid, actual_dy_grid = np.meshgrid(self.actual_dx, self.actual_dy, indexing='ij')
        ax.quiver(XX, YY, actual_dx_grid, actual_dy_grid, color='red', angles='xy', scale_units='xy', scale=1, label='Actual Move')
        
        max_err = np.max(Z)
        ax.set_title(f'Iteration {iteration} | Max Error: {max_err:.2e}')
        ax.set_xlim(self.x_bounds)
        ax.set_ylim(self.y_bounds)
        ax.legend(loc='upper right')
        plt.show()

    def add_refinement_axes(self, target_x, target_y):
        """
        指定された座標に新しい補間軸を追加する（重複回避の安全装置付き）
        """
        # 既存ノードとの距離をチェック（近すぎるとFH法のゼロ割りでNaNになる）
        min_dist_x = np.min(np.abs(self.x_nodes - target_x))
        min_dist_y = np.min(np.abs(self.y_nodes - target_y))
        
        # 距離が 1e-5 以下なら、微小値(1e-4)を足して衝突を回避する
        if min_dist_x < 1e-5:
            target_x += 1e-4
        if min_dist_y < 1e-5:
            target_y += 1e-4

        # 既存のノード配列に新しい座標を挿入してソート
        self.x_nodes = np.sort(np.append(self.x_nodes, target_x))
        self.y_nodes = np.sort(np.append(self.y_nodes, target_y))
        
        self.Nx = len(self.x_nodes)
        self.Ny = len(self.y_nodes)
        
        self.ideal_dx = np.zeros((self.Nx, self.Ny))
        self.ideal_dy = np.zeros((self.Nx, self.Ny))
        
        print(f"[*] 軸を追加しました: 新しいグリッドサイズ {self.Nx} x {self.Ny}")
    
    def optimize(self, num_iterations=12, learning_rate=1.0, verbose=True, plot=False, eval_res=100):
        # 履歴をリセット
        self.max_history = []
        self.min_history = []
        self.rel_history = []
        self.max_loc_history = []  
        
        # 評価グリッドと真値の準備
        x_eval = np.linspace(self.x_bounds[0], self.x_bounds[1], eval_res)
        y_eval = np.linspace(self.y_bounds[0], self.y_bounds[1], eval_res)
        X_eval, Y_eval = np.meshgrid(x_eval, y_eval, indexing='ij')
        
        f_val = self.target_func(X_eval, Y_eval)
        f_max_abs = np.max(np.abs(f_val)) + 1e-15

        # 【新設】クラス変数としてベストな状態を保持
        self.best_max_err = float('inf')
        best_x_nodes = None
        best_y_nodes = None

        # 【修正】record_history フラグを追加
        def evaluate_and_prepare_next(record_history=True):
            approx_val = self.approx_func(X_eval, Y_eval)
            abs_err = np.abs(approx_val - f_val)
            
            max_idx = np.unravel_index(np.argmax(abs_err, axis=None), abs_err.shape)
            global_max_abs = abs_err[max_idx]
            global_max_rel = global_max_abs / f_max_abs
            max_x = X_eval[max_idx]
            max_y = Y_eval[max_idx]
            
            self.calculate_displacements()
            valid_peaks = [p for p in self.local_extrema if p > 1e-16]
            min_peak = np.min(valid_peaks) if len(valid_peaks) > 0 else 0.0
            
            # フラグがTrueのときのみ履歴配列に追加する
            if record_history:
                self.max_history.append(global_max_abs)
                self.min_history.append(min_peak)
                self.rel_history.append(global_max_rel)
                self.max_loc_history.append((max_x, max_y)) 
            
            return global_max_abs, min_peak, global_max_rel, max_x, max_y

        # --- 初期状態 (Iter 0) ---
        self.fit_rational_function()
        cur_max, cur_min, cur_rel, cur_x, cur_y = evaluate_and_prepare_next(record_history=True)
        
        self.best_max_err = cur_max
        best_x_nodes = self.x_nodes.copy()
        best_y_nodes = self.y_nodes.copy()
        
        if verbose:
            print(f"--- Initial (Iter 0) --- LR: {learning_rate:.2f}")
            print(f"  Max Abs: {cur_max:.6e} at (x={cur_x:.3f}, y={cur_y:.3f}) | Min Peak: {cur_min:.6e} | Rel: {cur_rel:.6e}")
        if plot:
            self.plot_grid_and_errors(iteration=0)

        # --- 最適化ループ ---
        for it in range(1, num_iterations + 1):
            self.update_nodes(learning_rate)
            self.fit_rational_function()
            cur_max, cur_min, cur_rel, cur_x, cur_y = evaluate_and_prepare_next(record_history=True)

            # 過去のベストより良ければセーブ
            if cur_max < self.best_max_err:
                self.best_max_err = cur_max
                best_x_nodes = self.x_nodes.copy()
                best_y_nodes = self.y_nodes.copy()

            if verbose:
                print(f"--- Iteration {it} --- LR: {learning_rate:.2f}")
                print(f"  Max Abs: {cur_max:.6e} at (x={cur_x:.3f}, y={cur_y:.3f}) | Min Peak: {cur_min:.6e} | Rel: {cur_rel:.6e}")
            if plot:
                self.plot_grid_and_errors(iteration=it)

        # ==========================================
        # ループ終了後、最強の陣形に巻き戻す
        # ==========================================
        if verbose:
            print(f"  => Restoring to Best State (Min Max Err: {self.best_max_err:.6e})")
        
        self.x_nodes = best_x_nodes.copy()
        self.y_nodes = best_y_nodes.copy()
        self.fit_rational_function()
        
        # 【重要】履歴を汚さずに、クラス内部の状態（次の移動ベクトル等）だけを更新する
        evaluate_and_prepare_next(record_history=False)
    def add_x_axis(self, target_x):
        """ X軸（縦線）を1本だけ追加する """
        # 既存ノードとの衝突回避
        min_dist_x = np.min(np.abs(self.x_nodes - target_x))
        if min_dist_x < 1e-5:
            target_x += 1e-4

        self.x_nodes = np.sort(np.append(self.x_nodes, target_x))
        self.Nx = len(self.x_nodes)
        
        # 配列の再初期化
        self.ideal_dx = np.zeros((self.Nx, self.Ny))
        self.ideal_dy = np.zeros((self.Nx, self.Ny))
        self.actual_dx = np.zeros(self.Nx)
        self.actual_dy = np.zeros(self.Ny)
        print(f"[*] X軸を追加: x={target_x:.4f} (新グリッド: {self.Nx} x {self.Ny})")

    def add_y_axis(self, target_y):
        """ Y軸（横線）を1本だけ追加する """
        # 既存ノードとの衝突回避
        min_dist_y = np.min(np.abs(self.y_nodes - target_y))
        if min_dist_y < 1e-5:
            target_y += 1e-4

        self.y_nodes = np.sort(np.append(self.y_nodes, target_y))
        self.Ny = len(self.y_nodes)
        
        # 配列の再初期化
        self.ideal_dx = np.zeros((self.Nx, self.Ny))
        self.ideal_dy = np.zeros((self.Nx, self.Ny))
        self.actual_dx = np.zeros(self.Nx)
        self.actual_dy = np.zeros(self.Ny)
        print(f"[*] Y軸を追加: y={target_y:.4f} (新グリッド: {self.Nx} x {self.Ny})")

def run_adaptive_refinement(target_func, x_bounds, y_bounds,Nx,Ny,d,init_nodes,max_grid_size=400,grid_res=10,learning_rate = 1.0,num_iterations=10,err_tol=10e-16):
    """
    補間軸追加 ＆ Acklam最適化ループ
    """
    # 1. 初期化 
    print("=== Adaptive Refinement Start ===")
    optimizer = FHMinimax2D_Add_Axes(
        target_func=target_func,
        x_bounds=x_bounds, y_bounds=y_bounds,
        Nx=Nx, Ny=Ny, d=d, 
        init_nodes=init_nodes,
        search_method="grid", grid_res=grid_res
    )

    # 評価用の高解像度1次元配列
    eval_res = 1000
    x_eval = np.linspace(x_bounds[0], x_bounds[1], eval_res)
    y_eval = np.linspace(y_bounds[0], y_bounds[1], eval_res)

    iteration = 1
    # max_grid_size に達するまで回す
    while optimizer.Nx * optimizer.Ny < max_grid_size:
        print(f"\n========== Outer Loop {iteration} (Grid: {optimizer.Nx}x{optimizer.Ny}) ==========")
        
        # 2. Inner Loop: Acklamアルゴリズムによる微調整
        # ※ ここで「これ以上下がらない」まで局所最適化を行う
        optimizer.optimize(num_iterations=num_iterations, learning_rate=learning_rate, verbose=True)
        
        # 【修正箇所】履歴の最後尾ではなく、明示的に best_max_err を参照する
        current_max_err = optimizer.best_max_err
        
        if current_max_err < err_tol: # 目標精度
            print(f"Goal Reached! Error: {current_max_err:.2e}")
            break

        # 3. Outer Loop: 新しい補間軸の評価と追加
        # --- S_x(x) の計算 ---
        # 任意の x と、既存の全ての y_nodes の交点での誤差を計算
        X_grid_for_x, Y_grid_for_x = np.meshgrid(x_eval, optimizer.y_nodes, indexing='ij')
        err_x = optimizer.error_func(X_grid_for_x, Y_grid_for_x) # (eval_res, Ny) の配列
        S_x = np.sum(err_x, axis=1) # y方向（axis=1）に合計
        best_x_idx = np.argmax(S_x)
        max_score_x = S_x[best_x_idx]
        target_x = x_eval[best_x_idx]

        # --- S_y(y) の計算 ---
        # 任意の y と、既存の全ての x_nodes の交点での誤差を計算
        X_grid_for_y, Y_grid_for_y = np.meshgrid(optimizer.x_nodes, y_eval, indexing='ij')
        err_y = optimizer.error_func(X_grid_for_y, Y_grid_for_y) # (Nx, eval_res) の配列
        S_y = np.sum(err_y, axis=0) # x方向（axis=0）に合計
        best_y_idx = np.argmax(S_y)
        max_score_y = S_y[best_y_idx]
        target_y = y_eval[best_y_idx]

        # 4. スコアを比較して追加
        print(f"  -> Score X (at x={target_x:.3f}): {max_score_x:.4e}")
        print(f"  -> Score Y (at y={target_y:.3f}): {max_score_y:.4e}")
        
        if max_score_x > max_score_y:
            optimizer.add_x_axis(target_x)
        else:
            optimizer.add_y_axis(target_y)
            
        iteration += 1

    print("\n=== Adaptive Refinement Finished ===")
    # 【修正箇所】最終的に出来上がったグリッドで、最後にもう一度だけ最適化を走らせる
    print(f"Final Optimization (Grid: {optimizer.Nx}x{optimizer.Ny})")
    optimizer.optimize(num_iterations=num_iterations, learning_rate=learning_rate, verbose=True)


    return optimizer



def run_greedy_refinement_then_optimize(target_func, x_bounds, y_bounds, Nx, Ny, d, init_nodes, max_grid_size=400, grid_res=10, learning_rate=1.0, num_iterations=10):
    """
    補間軸を上限まで一気に追加し、最後に一回だけAcklam最適化を行うループ
    """
    print("=== Greedy Axis Addition Start ===")
    optimizer = FHMinimax2D_Add_Axes(
        target_func=target_func,
        x_bounds=x_bounds, y_bounds=y_bounds,
        Nx=Nx, Ny=Ny, d=d, 
        init_nodes=init_nodes,
        search_method="grid", grid_res=grid_res
    )

    # 評価用の高解像度1次元配列
    eval_res = 1000
    x_eval = np.linspace(x_bounds[0], x_bounds[1], eval_res)
    y_eval = np.linspace(y_bounds[0], y_bounds[1], eval_res)

    iteration = 1
    
    # --- フェーズ1: 軸の追加（最適化はしない） ---
    # max_grid_size に達するまでひたすら軸を追加する
    while optimizer.Nx * optimizer.Ny < max_grid_size:
        print(f"\n========== Adding Axis {iteration} (Grid: {optimizer.Nx}x{optimizer.Ny}) ==========")
        
        # 【重要】現在のノードを使って有理関数を再構築する（これをやらないと誤差が更新されない）
        optimizer.fit_rational_function()

        # --- S_x(x) の計算 ---
        # 任意の x と、既存の全ての y_nodes の交点での誤差を計算
        X_grid_for_x, Y_grid_for_x = np.meshgrid(x_eval, optimizer.y_nodes, indexing='ij')
        err_x = optimizer.error_func(X_grid_for_x, Y_grid_for_x) # (eval_res, Ny) の配列
        S_x = np.sum(err_x, axis=1) # y方向（axis=1）に合計
        best_x_idx = np.argmax(S_x)
        max_score_x = S_x[best_x_idx]
        target_x = x_eval[best_x_idx]

        # --- S_y(y) の計算 ---
        # 任意の y と、既存の全ての x_nodes の交点での誤差を計算
        X_grid_for_y, Y_grid_for_y = np.meshgrid(optimizer.x_nodes, y_eval, indexing='ij')
        err_y = optimizer.error_func(X_grid_for_y, Y_grid_for_y) # (Nx, eval_res) の配列
        S_y = np.sum(err_y, axis=0) # x方向（axis=0）に合計
        best_y_idx = np.argmax(S_y)
        max_score_y = S_y[best_y_idx]
        target_y = y_eval[best_y_idx]

        # スコアを比較して追加
        print(f"  -> Score X (at x={target_x:.3f}): {max_score_x:.4e}")
        print(f"  -> Score Y (at y={target_y:.3f}): {max_score_y:.4e}")
        
        if max_score_x > max_score_y:
            optimizer.add_x_axis(target_x)
        else:
            optimizer.add_y_axis(target_y)
            
        iteration += 1

    print("\n=== All Axes Added Successfully ===")
    
    # --- フェーズ2: 最後の最適化（Acklam Minimax） ---
    # 完成したグリッドをベースに、指定回数だけノードを微動させて全体の最大誤差を押し下げる
    print(f"Final Optimization (Grid: {optimizer.Nx}x{optimizer.Ny})")
    optimizer.optimize(num_iterations=num_iterations, learning_rate=learning_rate, verbose=True)

    return optimizer




class CauchyMinimax2D:
    def __init__(self, target_func, x_bounds, y_bounds, Nx, Ny, p_x=2, p_y=2, q_x=2, q_y=2,init_nodes=None,error_metric='absolute',search_method = "grid",grid_res = 15):

        self.search_method = search_method
        self.grid_res = grid_res

        self.target_func = target_func
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.Nx = Nx
        self.Ny = Ny
        
        self.error_metric = error_metric
        
        # テンソル積の次数
        self.p_x, self.p_y = p_x, p_y
        self.q_x, self.q_y = q_x, q_y
        
        # 未知数の数を確認
        num_unknowns = (p_x + 1) * (p_y + 1) + (q_x + 1) * (q_y + 1) - 1
        num_nodes = Nx * Ny
        print(f"Nodes: {num_nodes}, Unknowns: {num_unknowns}")
        if num_nodes < num_unknowns:
            print("Warning: ノード数が未知数の数より少ないため、劣決定系になります。")
            

        if init_nodes is not None:
            # 外部から指定されたノードを使用(チェビシェフノードなど)
            self.x_nodes, self.y_nodes = init_nodes
        else:
            # デフォルトでは等間隔
            self.x_nodes = np.linspace(x_bounds[0], x_bounds[1], Nx + 2)[1:-1]
            self.y_nodes = self.y_nodes = np.linspace(y_bounds[0], y_bounds[1], Ny + 2)[1:-1]
        
        self.approx_func = None 
        self.num_coeffs = None
        self.den_coeffs = None
        
        self.ideal_dx = np.zeros((Nx, Ny))
        self.ideal_dy = np.zeros((Nx, Ny))
        self.actual_dx = np.zeros(Nx)
        self.actual_dy = np.zeros(Ny)

        # 【追加1】極値を保存するリストを用意
        self.local_extrema = []
        self.max_history = [] # 各ステップでの極大値リストの最大値の推移
        self.min_history = [] # 各ステップでの極大値リストの最小値の推移
        # 【ここを追加】相対誤差の履歴用
        self.rel_linf_history = []
        self.naive_rel_history = []

    def fit_rational_function(self):
        """現在のノードを用いて、テンソル積有理関数の係数を最小二乗法で決定する"""
        xx, yy = np.meshgrid(self.x_nodes, self.y_nodes, indexing='ij')
        x_flat = xx.flatten()
        y_flat = yy.flatten()
        f_flat = self.target_func(x_flat, y_flat)
        
        # 行列Aの構築
        A_cols = []
        
        # 1. 分子の基底 (x^i * y^j)
        for i in range(self.p_x + 1):
            for j in range(self.p_y + 1):
                A_cols.append((x_flat**i) * (y_flat**j))
                
        # 2. 分母の基底 (-f * x^i * y^j) ※i=0, j=0 (定数項) は除外
        for i in range(self.q_x + 1):
            for j in range(self.q_y + 1):
                if i == 0 and j == 0:
                    continue
                A_cols.append(-f_flat * (x_flat**i) * (y_flat**j))
                
        A = np.column_stack(A_cols)
        b = f_flat # 定数項 b_00 = 1 のため、右辺は f(x,y)
        
        if self.error_metric == 'relative':
            # 相対誤差モード：重み付き最小二乗法 (1/|f| をかける)
            weights = 1.0 / (np.abs(f_flat) + 1e-12)
            A = A * weights[:, np.newaxis]
            b = b * weights

        # 最小二乗法で解く
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        num_terms_count = (self.p_x + 1) * (self.p_y + 1)
        self.num_coeffs = coeffs[:num_terms_count]
        self.den_coeffs = coeffs[num_terms_count:]
        
        # 評価用の近似関数を定義
        def approx(x, y):
            x, y = np.asarray(x), np.asarray(y)
            num_val = np.zeros_like(x, dtype=float)
            idx = 0
            for i in range(self.p_x + 1):
                for j in range(self.p_y + 1):
                    num_val += self.num_coeffs[idx] * (x**i) * (y**j)
                    idx += 1
                    
            den_val = np.ones_like(x, dtype=float) # b_00 = 1
            idx = 0
            for i in range(self.q_x + 1):
                for j in range(self.q_y + 1):
                    if i == 0 and j == 0:
                        continue
                    den_val += self.den_coeffs[idx] * (x**i) * (y**j)
                    idx += 1
            return num_val / den_val
            
        self.approx_func = approx

    
    def error_func(self, x, y):
        f_val = self.target_func(x, y)
        approx_val = self.approx_func(x, y)
        abs_err = np.abs(approx_val - f_val)
        
        if self.error_metric == 'relative':
            return abs_err / (np.abs(f_val) + 1e-12)
        return abs_err
    def find_local_extremum(self, x_min, x_max, y_min, y_max):
        x0 = (x_min + x_max) / 2.0
        y0 = (y_min + y_max) / 2.0
        res = minimize(lambda p: -self.error_func(p[0], p[1]), 
                       x0=[x0, y0], bounds=[(x_min, x_max), (y_min, y_max)],
                       method='L-BFGS-B')
        return res.x[0], res.x[1], -res.fun
    
    
    @staticmethod
    def process_cell_parallel(i, j, x_ext, y_ext, error_func, search_method, grid_res):
        qx_min, qx_max = x_ext[i], x_ext[i+1]
        qy_min, qy_max = y_ext[j], y_ext[j+1]
        
        if search_method == 'grid':
            # --- 超高速グリッドサーチ ---
            gx = np.linspace(qx_min, qx_max, grid_res)
            gy = np.linspace(qy_min, qy_max, grid_res)
            GX, GY = np.meshgrid(gx, gy, indexing='ij')
            
            err_vals = error_func(GX, GY) # ベクトル化で一括計算
            
            max_idx = np.unravel_index(np.argmax(err_vals, axis=None), err_vals.shape)
            max_x, max_y = gx[max_idx[0]], gy[max_idx[1]]
            max_err = err_vals[max_idx]
        else:
            # --- 従来の L-BFGS-B 探索 ---
            x0, y0 = (qx_min + qx_max) / 2.0, (qy_min + qy_max) / 2.0
            res = minimize(lambda p: -error_func(p[0], p[1]), 
                        x0=[x0, y0], bounds=[(qx_min, qx_max), (qy_min, qy_max)],
                        method='L-BFGS-B', tol=1e-5)
            max_x, max_y, max_err = res.x[0], res.x[1], -res.fun
            
        return i, j, max_x, max_y, max_err
    # x方向、y方向の移動を決める
    def calculate_displacements(self):
            err_f = self.error_func
            
            # 境界を含めた座標配列を作成 (長さ Nx+2, Ny+2)
            x_ext = np.concatenate(([self.x_bounds[0]], self.x_nodes, [self.x_bounds[1]]))
            y_ext = np.concatenate(([self.y_bounds[0]], self.y_nodes, [self.y_bounds[1]]))
            
            # 1. 全てのセルについて並列で極値探索 ( (Nx+1)*(Ny+1) 回 )
            # 例: 20x20ノードなら 21x21=441セル
            cell_results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(CauchyMinimax2D.process_cell_parallel)(
                i, j, x_ext, y_ext, err_f, self.search_method, self.grid_res
            )
            for i in range(self.Nx + 1) for j in range(self.Ny + 1)
            )
            
            # 2. セルの結果を2次元配列に整理
            cell_data = {} # (i, j) -> (max_x, max_y, max_err)
            self.local_extrema = []
            for i, j, mx, my, me in cell_results:
                cell_data[(i, j)] = (mx, my, me)
                self.local_extrema.append(me)

            # 3. 各ノードに対して、周囲4つのセルの結果を配分する
            for i in range(self.Nx):
                for j in range(self.Ny):
                    x_i, y_j = self.x_nodes[i], self.y_nodes[j]
                    
                    # ノード(i,j)を囲む4つのセルインデックス
                    # x_ext[i+1], y_ext[j+1] が現在のノード座標
                    surrounding_cells = [
                        (i+1, j+1), # 右上 (第1象限)
                        (i, j+1),   # 左上 (第2象限)
                        (i, j),     # 左下 (第3象限)
                        (i+1, j)    # 右下 (第4象限)
                    ]
                    
                    quad_data = []
                    sum_t = 0.0
                    
                    for c_idx in surrounding_cells:
                        max_x, max_y, max_err = cell_data[c_idx]
                        dx = max_x - x_i
                        dy = max_y - y_j
                        dist = np.hypot(dx, dy)
                        t = max_err / dist if dist > 1e-10 else 0.0
                        quad_data.append((t, dx, dy))
                        sum_t += t
                    
                    delta_x_star, delta_y_star = 0.0, 0.0
                    if sum_t > 0:
                        for (t, dx, dy) in quad_data:
                            delta_x_star += (t / sum_t) * dx
                            delta_y_star += (t / sum_t) * dy
                            
                    self.ideal_dx[i, j] = delta_x_star
                    self.ideal_dy[i, j] = delta_y_star
                    
            # 平均化
            self.actual_dx = np.mean(self.ideal_dx, axis=1)
            self.actual_dy = np.mean(self.ideal_dy, axis=0)

            # --- 追加：境界ノードの変位をゼロに固定（ピン留め） ---
            self.actual_dx[0] = 0.0
            self.actual_dx[-1] = 0.0
            self.actual_dy[0] = 0.0
            self.actual_dy[-1] = 0.0

            return self.actual_dx, self.actual_dy


    # def calculate_displacements(self):
    #     # 毎回の計算の最初に極値のリストをリセット
    #     self.local_extrema = []
    #     for i in range(self.Nx):
    #         for j in range(self.Ny):
    #             x_i = self.x_nodes[i]
    #             y_j = self.y_nodes[j]
                
    #             x_left   = self.x_nodes[i-1] if i > 0 else self.x_bounds[0]
    #             x_right  = self.x_nodes[i+1] if i < self.Nx-1 else self.x_bounds[1]
    #             y_bottom = self.y_nodes[j-1] if j > 0 else self.y_bounds[0]
    #             y_top    = self.y_nodes[j+1] if j < self.Ny-1 else self.y_bounds[1]
                
    #             quadrants = [
    #                 (x_i, x_right, y_j, y_top),
    #                 (x_left, x_i, y_j, y_top),
    #                 (x_left, x_i, y_bottom, y_j),
    #                 (x_i, x_right, y_bottom, y_j)
    #             ]
                
    #             quad_data = []
    #             sum_t = 0.0
                
    #             for (qx_min, qx_max, qy_min, qy_max) in quadrants:
    #                 max_x, max_y, max_err = self.find_local_extremum(qx_min, qx_max, qy_min, qy_max)

    #                 # 見つけた極値の絶対値をリストに記録
    #                 self.local_extrema.append(max_err)

    #                 dx = max_x - x_i
    #                 dy = max_y - y_j
    #                 dist = np.hypot(dx, dy)
    #                 t = max_err / dist if dist > 1e-10 else 0.0
    #                 quad_data.append((t, dx, dy))
    #                 sum_t += t
                
    #             delta_x_star, delta_y_star = 0.0, 0.0
    #             if sum_t > 0:
    #                 for (t, dx, dy) in quad_data:
    #                     weight = t / sum_t
    #                     delta_x_star += weight * dx
    #                     delta_y_star += weight * dy
                        
    #             self.ideal_dx[i, j] = delta_x_star
    #             self.ideal_dy[i, j] = delta_y_star
                
    #     self.actual_dx = np.mean(self.ideal_dx, axis=1)
    #     self.actual_dy = np.mean(self.ideal_dy, axis=0)

    def update_nodes(self, learning_rate=0.4):
        self.x_nodes += self.actual_dx * learning_rate
        self.y_nodes += self.actual_dy * learning_rate

    def plot_grid_and_errors(self, iteration):
        X, Y = np.meshgrid(np.linspace(self.x_bounds[0], self.x_bounds[1], 100),
                           np.linspace(self.y_bounds[0], self.y_bounds[1], 100))
        Z = self.error_func(X, Y)

        fig, ax = plt.subplots(figsize=(9, 7))
        c = ax.pcolormesh(X, Y, Z, shading='auto', cmap='Reds', alpha=0.5)
        fig.colorbar(c, ax=ax, label='Absolute Error')
        
        for xi in self.x_nodes: ax.axvline(xi, color='gray', linestyle='-', alpha=0.4)
        for yj in self.y_nodes: ax.axhline(yj, color='gray', linestyle='-', alpha=0.4)
            
        XX, YY = np.meshgrid(self.x_nodes, self.y_nodes, indexing='ij')
        ax.scatter(XX, YY, color='black', s=40, zorder=5, label='Nodes')
        
        ax.quiver(XX, YY, self.ideal_dx, self.ideal_dy, color='blue', alpha=0.3, angles='xy', scale_units='xy', scale=1, label='Ideal Pull')
        
        actual_dx_grid, actual_dy_grid = np.meshgrid(self.actual_dx, self.actual_dy, indexing='ij')
        ax.quiver(XX, YY, actual_dx_grid, actual_dy_grid, color='red', angles='xy', scale_units='xy', scale=1, label='Actual Move')
        
        max_err = np.max(Z)
        ax.set_title(f'Iteration {iteration} | Max Error: {max_err:.2e}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(self.x_bounds)
        ax.set_ylim(self.y_bounds)
        ax.legend(loc='upper right')
        plt.show()
    
    def optimize(self, num_iterations=12, learning_rate=0.4, verbose=True, plot=False, anzensouchi=False):
        # 履歴をリセット
        self.max_history, self.min_history = [], []
        self.rel_linf_history, self.naive_rel_history = [], []
        
        it = 1
        while it <= num_iterations:
            old_x, old_y = self.x_nodes.copy(), self.y_nodes.copy()
            self.fit_rational_function()
            self.calculate_displacements()
            
            cur_max = np.max(self.local_extrema)
            cur_min = np.min(self.local_extrema)

            # --- 【追加：相対誤差の計算】 ---
            # 50x50程度なら計算負荷はほぼゼロです
            X, Y = np.meshgrid(np.linspace(self.x_bounds[0], self.x_bounds[1], 50),
                               np.linspace(self.y_bounds[0], self.y_bounds[1], 50))
            f_val = self.target_func(X, Y)
            err = np.abs(self.approx_func(X, Y) - f_val)
            
            rel_linf = cur_max / (np.max(np.abs(f_val)) + 1e-15)
            rel_naive = np.max(err / (np.abs(f_val) + 1e-15))
            
            # 履歴に保存（これで後から取り出せます）
            self.rel_linf_history.append(rel_linf)
            self.naive_rel_history.append(rel_naive)
            # -------------------------------

            # 安全装置の判定
            if anzensouchi and len(self.max_history) > 0 and cur_max > self.max_history[-1] * 5:
                self.x_nodes, self.y_nodes = old_x, old_y
                learning_rate *= 0.5
                if learning_rate < 1e-7: break
                continue 
            
            self.max_history.append(cur_max)
            self.min_history.append(cur_min)
            
            if verbose:
                print(f"--- iteration {it} --- Max: {cur_max:.6e} | Min: {cur_min:.6e}")
                print(f"    Rel L-inf: {rel_linf:.4e} | Naive: {rel_naive:.4e}")

            if plot:
                self.plot_grid_and_errors(iteration=it)
            
            self.update_nodes(learning_rate)
            it += 1
       
            
    def _generate_chebyshev_nodes(self, n, bounds):
            """1次元のチェビシェフ点を生成し、指定された範囲に変換する"""
            x_min, x_max = bounds
            # [-1, 1] 区間のチェビシェフ点 (第1種) [cite: 956]
            k = np.arange(1, n + 1)
            u = np.cos((2 * (n - k) + 1) / (2 * n) * np.pi)
            
            # [x_min, x_max] への線形変換 [cite: 970]
            return 0.5 * (x_min + x_max) + 0.5 * (x_max - x_min) * u