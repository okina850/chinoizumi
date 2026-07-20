import numpy as np
from scipy import integrate, optimize
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.optimize import brentq
from scipy.optimize import minimize
from numba import njit

import sympy as sp
from IPython.display import display, Math,Latex,Markdown
#from scipy.interpolate import AAA
#from pymor.reductors.aaa import PAAAReductor
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import time



def plot_function(target_func,x_bounds,y_bounds):
    x_grid, y_grid = np.meshgrid(
            np.linspace(x_bounds[0],x_bounds[1],100),
            np.linspace(y_bounds[0],y_bounds[1],100)
    )
    fig_plotly = make_subplots(
                    rows = 1, cols = 1,
                    specs = [
                        [{'type':'surface'}]
                    ],
                    subplot_titles=(
                        ['Target Function']
                    ),
                    vertical_spacing=0.05
                )

    fig_plotly.add_trace(
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=target_func(x_grid,y_grid)
        ),
        row=1,col=1
    )
    fig_plotly.show()


def get_chebyshev_nodes(a, b, n):
    """[a, b] 区間に n 個のチェビシェフ点（端点を含む）を生成する"""
    # 基本のチェビシェフ点 [-1, 1] [cite: 356]
    nodes = np.cos(np.pi * np.arange(n) / (n - 1))
    # [a, b] 区間にスケーリング
    return np.sort(0.5 * (a + b) + 0.5 * (b - a) * nodes)

def create_swapped_pairs(F_Y_given_X,x, y_min=-10.0, y_max=10.0, num_points=500):
    """
    指定された x に対して、y のグリッドから (v, y) のペアリストを事前計算する
    """
    # 1. 任意の y の点列 (D_2) を作成
    y_grid = np.linspace(y_min, y_max, num_points)
    
    # 2. brentqを使わず、順方向の積分だけで v=F(y|x) を計算
    # これが資料の「(F_Y|X=x(y), y)のリスト」に相当します
    v_grid = np.array([F_Y_given_X(y, x) for y in y_grid])
    
    # v_grid は単調増加になるため、そのまま逆関数の補間用データとして使える
    return v_grid, y_grid

from pathlib import Path
import numpy as np


def smart_save(path, arr):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, arr)


def get_and_display_formula(interpolator):
    """
    2変数のFloater-Hormann補間器から数式を構築し、表示する
    """
    if interpolator.ndim != 2:
        raise ValueError("この関数は2変数の補間器専用です。")
        
    # SymPyのシンボルを定義
    x, y = sp.symbols('x y')
    
    # 補間器から必要なデータを取得
    x_nodes_interp = interpolator.points[0]
    y_nodes_interp = interpolator.points[1]
    v_weights = interpolator.weights[0]
    w_weights = interpolator.weights[1]
    f_values = interpolator.values
    
    # 分子(Numerator)と分母(Denominator)を初期化
    num = 0
    den = 0
    
    # バリセントリック形式の定義に従って総和を計算
    for i in range(len(x_nodes_interp)):
        for j in range(len(y_nodes_interp)):
            # 重みとノードの項: v_i * w_j / ((x - x_i) * (y - y_j))
            # （※表示をスッキリさせるため、float値をnsimplifyで有理数化できる場合はしておく）
            v_val = sp.nsimplify(v_weights[i], tolerance=1e-10)
            w_val = sp.nsimplify(w_weights[j], tolerance=1e-10)
            xi_val = sp.nsimplify(x_nodes_interp[i], tolerance=1e-10)
            yj_val = sp.nsimplify(y_nodes_interp[j], tolerance=1e-10)
            f_val = sp.nsimplify(f_values[i, j], tolerance=1e-10)
            
            # 各項のベース部分
            term = (v_val * w_val) / ((x - xi_val) * (y - yj_val))
            
            # 分子と分母に加算
            num += term * f_val
            den += term
            
    # r(x,y) = 分子 / 分母
    r_xy = num / den
    
    # IPython環境で綺麗に表示
    display(Math(r'r(x,y) = ' + sp.latex(r_xy)))
    
    return r_xy





# --- 共通で使う真値計算の並列化関数 ---
def parallel_true_values(v_grid, x_grid, func_scalar):
    v_flat = v_grid.flatten()
    x_flat = x_grid.flatten()
    results = Parallel(n_jobs=-1)(
        delayed(func_scalar)(v, x) for v, x in zip(v_flat, x_flat)
    )
    return np.array(results).reshape(v_grid.shape)

# =========================================================
# アプローチB: 逆関数の直接近似 (ハイブリッド・厳密版)
# =========================================================
def build_and_evaluate_hybrid_inverse_CDF(target_func, true_inv_scalar, v_nodes, x_nodes, v_eval, x_eval,d,use_plotly = True):
    # 1. 補間用データ（カンペ）の作成
    V_grid, X_grid = np.meshgrid(v_nodes, x_nodes, indexing='ij')
    
    print("------- アプローチB: ハイブリッド逆関数モード -------")
    print(f"データ生成開始 (ノード数: V={len(v_nodes)} x X={len(x_nodes)})")
    t_start = time.time()
    Z_values = target_func(V_grid, X_grid)
    print(f"データ生成完了: {time.time() - t_start:.4f} sec")

    # 2. FH補間器の構築
    print("FH補間モデル構築開始...")
    t_start = time.time()
    raw_interpolator = MultivariateFloaterHormannInterpolator(
        points=(v_nodes, x_nodes), 
        values=Z_values, 
        d=d 
    )
    print(f"モデル構築完了: {time.time() - t_start:.4f} sec")

    # 【超重要】論文に載せても恥ずかしくないハイブリッドサンプラー
    v_min_safe, v_max_safe = v_nodes[0], v_nodes[-1]
    
    def hybrid_inverse_generator(V, X):
        """
        中央の安全な領域は爆速なFH補間を使用し、
        両端の裾野 (v < v_min または v > v_max) は厳密な brentq を使用する。
        """
        Z_out = np.zeros_like(V)
        
        # 中央と裾野のマスク（条件）を作成
        mask_center = (V >= v_min_safe) & (V <= v_max_safe)
        mask_tail = ~mask_center
        
        # ① 中央部分を爆速で一括計算
        if np.any(mask_center):
            Z_out[mask_center] = raw_interpolator((V[mask_center], X[mask_center]))
            
        # ② 裾野部分を厳密に計算 (インチキなし)
        if np.any(mask_tail):
            v_tails = V[mask_tail]
            x_tails = X[mask_tail]
            # 裾野の要素数だけ、愚直に真値計算を回す
            tail_results = [true_inv_scalar(vt, xt) for vt, xt in zip(v_tails, x_tails)]
            Z_out[mask_tail] = tail_results
            
        return Z_out

    # 3. 評価用グリッドで高速生成テスト
    V_eval_mesh, X_eval_mesh = np.meshgrid(v_eval, x_eval, indexing='ij')
    
    print(f"評価グリッド (V x X = {len(v_eval)} x {len(x_eval)}) でハイブリッド生成開始...")
    t_start = time.time()
    
    # 爆速な補間と、少し重い真値計算が内部で自動的に切り替わります
    Z_interp = hybrid_inverse_generator(V_eval_mesh, X_eval_mesh)
    
    print(f"生成完了: {time.time() - t_start:.4f} sec  ← 注目!!")

    # 4. 真値の計算
    print("※誤差比較のための真値計算 (少し時間がかかります)...")
    t_start = time.time()
    Z_true = parallel_true_values(V_eval_mesh, X_eval_mesh, true_inv_scalar)
    print(f"真値計算完了: {time.time() - t_start:.2f} sec")

    # 誤差
    abs_error = np.abs(Z_true - Z_interp)
    max_true = np.nanmax(np.abs(Z_true))
    rel_error = abs_error / max_true

    # 5. 結果のプロット
    if not use_plotly:

        fig = plt.figure(figsize=(12, 8))

        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot_surface(V_eval_mesh, X_eval_mesh, Z_true, cmap='viridis')
        ax1.set_title("Exact Inverse Function")
        ax1.set_xlabel("v (probability)")
        ax1.set_ylabel("x (condition)")

        ax2 = fig.add_subplot(222, projection='3d')
        ax2.plot_surface(V_eval_mesh, X_eval_mesh, Z_interp, cmap='plasma')
        ax2.scatter(V_grid, X_grid, Z_values, color='r', s=15, label='Data Nodes')
        ax2.set_title("Hybrid FH-Interpolation")
        ax2.set_xlabel("v (probability)")
        ax2.set_ylabel("x (condition)")



        ax3 = fig.add_subplot(223, projection='3d')
        ax3.plot_surface(V_eval_mesh, X_eval_mesh, abs_error, cmap='magma')
        ax3.set_title("Absolute Error")
        ax3.set_xlabel("v (probability)")
        ax3.set_ylabel("x (condition)")

        ax4 = fig.add_subplot(224, projection='3d')
        ax4.plot_surface(V_eval_mesh, X_eval_mesh, rel_error, cmap='magma')
        ax4.set_title(rf"Relative Error (Max($L_{{\infty}}$) : {np.nanmax(rel_error):.3f})")
        ax4.set_xlabel("v (probability)")
        ax4.set_ylabel("x (condition)")

        plt.tight_layout()
        plt.show()
    else:
        # ==========================================
        # Plotly: 1行2列の2画面構成に修正
        # ==========================================
        fig_plotly = make_subplots(
            rows=2, cols=1,
            specs=[
                [{'type': 'surface'}],
                [{'type': 'surface'}]
            ],
            subplot_titles=(
                "Exact vs Approximation",
                rf"Relative Error (Max: {np.nanmax(rel_error):.3f})"
            ),
            horizontal_spacing=0.05
        )

        # --------------------------------------------------
        # 左の図 (col=1): 真値と近似値の重ね合わせ + 補間ノード
        # --------------------------------------------------
        # ① 真値の曲面 (少し透明にして重ね合わせを見やすくする)
        fig_plotly.add_trace(
            go.Surface(
                x=V_eval_mesh, y=X_eval_mesh, z=Z_true,
                colorscale='Viridis', opacity=0.8, showscale=False, name='Exact'
            ),
            row=1, col=1
        )

        # ② 近似値の曲面
        fig_plotly.add_trace(
            go.Surface(
                x=V_eval_mesh, y=X_eval_mesh, z=Z_interp,
                colorscale='Plasma', opacity=0.8, showscale=False, name='Approximation'
            ),
            row=1, col=1
        )

        # ③ 補間ノード (値の位置に赤い丸)
        fig_plotly.add_trace(
            go.Scatter3d(
                x=V_grid.flatten(), 
                y=X_grid.flatten(), 
                z=Z_values.flatten(),
                mode='markers',
                marker=dict(size=4, color='red', symbol='circle'),
                name='Nodes'
            ),
            row=1, col=1
        )

        # --------------------------------------------------
        # 右の図 (col=2): 相対誤差 + 補間ノードの配置
        # --------------------------------------------------
        # ④ 相対誤差の曲面
        fig_plotly.add_trace(
            go.Surface(
                x=V_eval_mesh, y=X_eval_mesh, z=rel_error,
                colorscale='Magma', name='Relative Error'
            ),
            row=2, col=1
        )

        # ⑤ 相対誤差の図上の補間ノード
        # 補間点での誤差はゼロになるので、z軸をすべて0にして底面に配置する
        # fig_plotly.add_trace(
        #     go.Scatter3d(
        #         x=V_grid.flatten(), 
        #         y=X_grid.flatten(), 
        #         z=np.zeros_like(V_grid.flatten()), # z=0 に配置
        #         mode='markers',
        #         marker=dict(size=3, color='cyan', symbol='x'), # 誤差図では見やすく水色の×印に
        #         name='Nodes (Error=0)'
        #     ),
        #     row=1, col=2
        # )

        # --------------------------------------------------
        # レイアウトと軸ラベル (v, x, 適切なz) の設定
        # --------------------------------------------------
        fig_plotly.update_layout(
            height=1400,# 
            width=700,
            title_text="Hybrid Inverse CDF Approximation",
            showlegend=False,
            
            # 左の図の軸設定
            scene=dict(
                xaxis_title="v (probability)",
                yaxis_title="x (condition)",
                zaxis_title="y (value)"
            ),
            # 右の図の軸設定
            scene2=dict(
                xaxis_title="v (probability)",
                yaxis_title="x (condition)",
                zaxis_title="Relative Error"
            )
        )

        fig_plotly.show()
    
    # 本番のシミュレーションで使い回せるように、ハイブリッド関数を返します
    return raw_interpolator, hybrid_inverse_generator
    
    
def generate_chebyshev_nodes(n, bounds):
        """1次元のチェビシェフ点を生成し、指定された範囲に変換する"""
        x_min, x_max = bounds
        # [-1, 1] 区間のチェビシェフ点 (第1種) [cite: 956]
        k = np.arange(1, n + 1)
        u = np.cos((2 * (n - k) + 1) / (2 * n) * np.pi)
        
        # [x_min, x_max] への線形変換 [cite: 970]
        return 0.5 * (x_min + x_max) + 0.5 * (x_max - x_min) * u


def plot_convergence(optimizer):
    """クラスに保存された履歴データを使ってプロットだけを行う"""
    fig, ax = plt.subplots(figsize=(8, 6))
    iterations = range(1, len(optimizer.max_history) + 1)
    
    ax.plot(iterations, optimizer.max_history, color="red", marker="o", label="Max Error")
    ax.plot(iterations, optimizer.min_history, color="blue", marker="o", label="Min Error")
    
    ax.set_title("Minimax Convergence")
    ax.set_yscale('log')
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.show()

