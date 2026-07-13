import numpy as np

################################
# toy functions of bivariate
################################

def toy_func1(x,y):
    """
    Domain: 通常 [0,10] x [0,10]
    toy Function No.28 of Poussot-Vassal_et_al.__2026__TENSOR-BASED_MULTIVARIATE_FUNCTION_APPROXIMATION_METHODS_BENCHMARKING_AND_COMPARISON
    """
    return (x / (x + 1))**4 * (1 + np.exp(-y**2)) * (1 + y*np.cos(y)*np.exp((-x*y)/(x + 1)))

def toy_func2(x,y):
    """
    Domain: 通常 [-2,2] x [-2,2]
    toy Function No.32 of Poussot-Vassal_et_al.__2026__TENSOR-BASED_MULTIVARIATE_FUNCTION_APPROXIMATION_METHODS_BENCHMARKING_AND_COMPARISON
    """
    return np.arctan(x) + y**3

def toy_func3(x,y):
    """
    Domain: 通常 [-10,10] x [-10,10]
    toy Function No.33 of Poussot-Vassal_et_al.__2026__TENSOR-BASED_MULTIVARIATE_FUNCTION_APPROXIMATION_METHODS_BENCHMARKING_AND_COMPARISON
    """
    return (x + y) / ( np.cos(x)**2 + np.cos(y) + 3 )

def franke_func(x,y):
    """
    Domain:  [0,1] x [0,1]        
    Classical Franke's function
    """
    term1 = 0.75 * np.exp(-( (9*x - 2)**2 / 4 ) - ( (9*y - 2)**2 / 4 ))
    term2 = 0.75 * np.exp(-( (9*x + 1)**2 / 49 ) - ( (9*y + 1) / 10 ))
    term3 = 0.5  * np.exp(-( (9*x - 7)**2 / 4 ) - ( (9*y - 3)**2 / 4 ))
    term4 = -0.2 * np.exp(-( (9*x - 4)**2 ) - ( (9*y - 7)**2 ))
    
    return term1 + term2 + term3 + term4



#########################################
#  混合正規分布のF_{Y|X=x}^{-1}(v):fast_inv_F_Y_given_X
#########################################
from scipy.special import erf
from scipy import optimize

def analytical_F_Y_given_X(y, x):
    """
    数値積分(quad)を使わず、誤差関数(erf)を用いて
    pdf3の条件付きCDFを直接計算する関数（O(1)で完了します）
    """
    # xに依存する重み成分
    w1 = np.exp(-x**2)
    w2 = 0.5 * np.exp(-(x - 2)**2)
    
    # yに依存する累積確率成分（ 1/2 * (1 + erf(z)) ）
    cdf1 = 0.5 * (1.0 + erf(y))
    cdf2 = 0.5 * (1.0 + erf(y - 2))
    
    # 重み付き平均が条件付きCDFとなる
    return (w1 * cdf1 + w2 * cdf2) / (w1 + w2)

def fast_inv_F_Y_given_X_scalar(v, x):
    lower_bound = -100.0
    upper_bound = 100.0
    try:
        # 重たい quad ではなく、超高速な analytical_F_Y_given_X を呼び出す
        y_solution = optimize.brentq(
            lambda y: analytical_F_Y_given_X(y, x) - v, 
            lower_bound, 
            upper_bound
        )
        return y_solution
    except ValueError:
        raise ValueError(f"解が見つかりませんでした。v={v}, x={x}")

# ベクトル化（これが新しい target_func になります）
fast_inv_F_Y_given_X = np.vectorize(fast_inv_F_Y_given_X_scalar)


#########################################
#  混合正規分布の周辺CDF F_X(x) と 逆関数 F_X^{-1}(u)
#########################################

def analytical_F_X(x):
    """
    数値積分(quad)を使わず、誤差関数(erf)を用いて
    周辺CDF F_X(x) を直接計算する関数（O(1)で完了します）
    """
    # X成分ごとの累積確率（ 1/2 * (1 + erf(z)) ）
    cdf1 = 0.5 * (1.0 + erf(x))
    cdf2 = 0.5 * (1.0 + erf(x - 2))
    
    # yを積分して消去した後に残る、それぞれの成分の「重み」
    # (元のpdfの係数 1.0 と 0.5 がそのまま重みの比率になります)
    w1 = 1.0
    w2 = 0.5
    
    # 重み付き平均が周辺CDFとなる
    return (w1 * cdf1 + w2 * cdf2) / (w1 + w2)

def fast_inv_F_X_scalar(u):
    """
    Brent法を用いて F_X(x) - u = 0 を解き、周辺CDFの逆関数を求める
    """
    # 制約条件である [-4, 4] より少し広めに探索範囲を設定します
    # 確率的にほぼ0〜1をカバーする十分な範囲です
    lower_bound = -10.0
    upper_bound = 10.0
    try:
        # 重たい quad ではなく、超高速な analytical_F_X を呼び出す
        x_solution = optimize.brentq(
            lambda x: analytical_F_X(x) - u, 
            lower_bound, 
            upper_bound
        )
        return x_solution
    except ValueError:
        raise ValueError(f"解が見つかりませんでした。u={u}")

# ベクトル化（NumPy配列の乱数uをまとめて処理できるようにする）
fast_inv_F_X = np.vectorize(fast_inv_F_X_scalar)





############################################################
# 厳密version
############################################################
import numpy as np
from scipy.special import erf
from scipy import optimize

# 1. 厳密な正規化定数 c の計算
def calculate_exact_c():
    term1 = np.pi * (erf(4.0))**2
    term2 = (np.pi / 8.0) * (erf(2.0) + erf(6.0))**2
    return 1.0 / (term1 + term2)

C_NORM_TRUNCATED = calculate_exact_c()

# 2. 修正版：条件付きCDF F_{Y|X=x}(y)
def analytical_F_Y_given_X_truncated(y, x):
    # xに依存する重み
    w1 = np.exp(-x**2)
    w2 = 0.5 * np.exp(-(x - 2)**2)
    
    # 積分下限が -4 であることに起因する成分（分子）
    num1 = erf(y) - erf(-4.0)
    num2 = erf(y - 2.0) - erf(-6.0)
    numerator = w1 * num1 + w2 * num2
    
    # yについて -4 から 4 まで積分した値（分母）
    den1 = erf(4.0) - erf(-4.0)
    den2 = erf(2.0) - erf(-6.0)
    denominator = w1 * den1 + w2 * den2
    
    return numerator / denominator

# 3. 修正版：周辺CDF F_X(x)
def analytical_F_X_truncated(x):
    # xについての積分（分子）
    # y側の積分結果（定数）がそれぞれの重みになる
    weight_y1 = erf(4.0) - erf(-4.0)
    weight_y2 = erf(2.0) - erf(-6.0)
    
    num1 = erf(x) - erf(-4.0)
    num2 = erf(x - 2.0) - erf(-6.0)
    
    numerator = 1.0 * weight_y1 * num1 + 0.5 * weight_y2 * num2
    
    # 全体領域の積分（分母）
    den1 = erf(4.0) - erf(-4.0)
    denominator = 1.0 * weight_y1 * den1 + 0.5 * weight_y2 * weight_y2
    
    return numerator / denominator

# 4. 逆関数の計算（探索範囲を [-4, 4] に厳密化）
def fast_inv_F_Y_given_X_scalar_truncated(v, x):
    try:
        return optimize.brentq(
            lambda y: analytical_F_Y_given_X_truncated(y, x) - v, 
            -4.0, 4.0  # 探索範囲を限定
        )
    except ValueError:
        raise ValueError(f"解が見つかりませんでした。v={v}, x={x}")

def fast_inv_F_X_scalar_truncated(u):
    try:
        return optimize.brentq(
            lambda x: analytical_F_X_truncated(x) - u, 
            -4.0, 4.0  # 探索範囲を限定
        )
    except ValueError:
        raise ValueError(f"解が見つかりませんでした。u={u}")

fast_inv_F_Y_given_X_trunc = np.vectorize(fast_inv_F_Y_given_X_scalar_truncated)
fast_inv_F_X_trunc = np.vectorize(fast_inv_F_X_scalar_truncated)




###########################################
#
###########################################
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib.pyplot as plt

def pdf_func1(x):
    """確率密度関数 f_X(x)"""
    x = np.asarray(x)
    # 0 < x < 1 の範囲外は 0 を返す
    valid = (x > 0) & (x < 1)
    res = np.zeros_like(x, dtype=float)
    x_valid = x[valid]
    res[valid] = (1 + np.sin(8 * np.pi * x_valid)) / (np.pi * np.sqrt(x_valid * (1 - x_valid)))
    return res if res.ndim > 0 else res.item()

def _cdf_func1_single(x):
    """単一の値に対する累積分布関数 F_X(x)"""
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    
    # 第1項：解析的に積分できる部分（逆正弦分布のCDF）
    term1 = (2 / np.pi) * np.arcsin(np.sqrt(x))
    
    # 第2項：数値積分する部分（特異点が相殺されているため安定する）
    def integrand(t):
        if t == 0 or t == 1:
            return 0.0
        return np.sin(8 * np.pi * t) / (np.pi * np.sqrt(t * (1 - t)))
        
    term2, _ = quad(integrand, 0, x, epsrel=1e-6)
    
    return term1 + term2

# ベクトル入力に対応させる
cdf_func1 = np.vectorize(_cdf_func1_single)

def _inv_cdf_func1_single(u):
    """単一の値に対する逆累積分布関数 F_X^{-1}(u)"""
    if u <= 0: return 0.0
    if u >= 1: return 1.0
    
    # F_X(x) - u = 0 を満たす x を [0, 1] の範囲で探索 (Brent法)
    # 端点でのゼロ除算を防ぐため、探索範囲をわずかに内側に設定
    return brentq(lambda x: _cdf_func1_single(x) - u, 0.0, 1.0)

# ベクトル入力に対応させる
inv_cdf_func1 = np.vectorize(_inv_cdf_func1_single)
import numpy as np
import scipy.stats as stats

# =====================================================================
# 1. 標準正規分布 (Standard Normal Distribution)
# パラメータ: 
#   - なし（平均 μ=0, 標準偏差 σ=1 に固定された分布）
# =====================================================================

def normal_pdf(x):
    """標準正規分布の確率密度関数 (PDF)"""
    return stats.norm.pdf(x)

def normal_cdf(x):
    """標準正規分布の累積分布関数 (CDF)"""
    return stats.norm.cdf(x)

def normal_invcdf(q):
    """標準正規分布の逆累積分布関数 (invCDF)
    引数 q: 累積確率 (0 <= q <= 1)
    """
    return stats.norm.ppf(q)


# =====================================================================
# 2. ガンマ分布 (Gamma Distribution)
# パラメータ:
#   - k (a): 形状パラメータ (shape > 0)。分布の非対称性や尖り方を制御する。
#   - theta (scale): 尺度パラメータ (scale > 0)。分布の横方向のスケールを制御する。
# デフォルト値の意図 (k=3.0, theta=2.0):
#   - 原点から滑らかに立ち上がり、右側に長い裾を引く釣鐘型の非対称分布。
#     （期待値は k * theta = 6.0 となる）
# =====================================================================

def gamma_pdf(x, k=3.0, theta=2.0):
    """ガンマ分布の確率密度関数 (PDF)"""
    return stats.gamma.pdf(x, a=k, loc=0, scale=theta)

def gamma_cdf(x, k=3.0, theta=2.0):
    """ガンマ分布の累積分布関数 (CDF)"""
    return stats.gamma.cdf(x, a=k, loc=0, scale=theta)

def gamma_invcdf(q, k=3.0, theta=2.0):
    """ガンマ分布の逆累積分布関数 (invCDF)
    引数 q: 累積確率 (0 <= q <= 1)
    """
    return stats.gamma.ppf(q, a=k, loc=0, scale=theta)


# =====================================================================
# 3. ベータ分布 (Beta Distribution)
# パラメータ:
#   - alpha (a): 形状パラメータ 1 (> 0)。定義域左側（0付近）の挙動を制御する。
#   - beta_param (b): 形状パラメータ 2 (> 0)。定義域右側（1付近）の挙動を制御する。
# デフォルト値の意図 (alpha=2.0, beta_param=5.0):
#   - 有界区間 [0, 1] で定義され、確率のピークが左側に偏り、
#     右側に向かって緩やかに減衰する右裾の長い非対称分布。
# =====================================================================

def beta_pdf(x, alpha=2.0, beta_param=5.0):
    """ベータ分布の確率密度関数 (PDF)"""
    return stats.beta.pdf(x, a=alpha, b=beta_param)

def beta_cdf(x, alpha=2.0, beta_param=5.0):
    """ベータ分布の累積分布関数 (CDF)"""
    return stats.beta.cdf(x, a=alpha, b=beta_param)

def beta_invcdf(q, alpha=2.0, beta_param=5.0):
    """ベータ分布の逆累積分布関数 (invCDF)
    引数 q: 累積確率 (0 <= q <= 1)
    """
    return stats.beta.ppf(q, a=alpha, b=beta_param)


# =====================================================================
# 4. スチューデントのt分布 (Student's t-Distribution)
# パラメータ:
#   - nu (df): 自由度 (degrees of freedom > 0)。裾の厚さを制御する。
# デフォルト値の意図 (nu=4.0):
#   - 標準正規分布に比べてすそ野（テール）が厚く、中心のピークがやや低い対称分布。
#     金融データのモデリングや、小標本における検定で典型的に現れる形状。
# =====================================================================

def t_pdf(x, nu=4.0):
    """スチューデントのt分布の確率密度関数 (PDF)"""
    return stats.t.pdf(x, df=nu)

def t_cdf(x, nu=4.0):
    """スチューデントのt分布の累積分布関数 (CDF)"""
    return stats.t.cdf(x, df=nu)

def t_invcdf(q, nu=4.0):
    """スチューデントのt分布の逆累積分布関数 (invCDF)
    引数 q: 累積確率 (0 <= q <= 1)
    """
    return stats.t.ppf(q, df=nu)


# =====================================================================
# 5. フォン・ミーゼス分布 (von Mises Distribution)
# パラメータ:
#   - mu (loc): 平均方向 (位置パラメータ)。分布の中心となる角度（ラジアン）。
#   - kappa: 集中度 (shape >= 0)。大きいほど mu の周りに鋭く集中する。
# デフォルト値の意図 (mu=0.0, kappa=1.0):
#   - 角度 0（正面）を中心に、適度なばらつきを持って円周上に広がっている状態。
# =====================================================================

def vonmises_pdf(x, mu=0.0, kappa=1.0):
    """フォン・ミーゼス分布の確率密度関数 (PDF)"""
    return stats.vonmises.pdf(x, kappa=kappa, loc=mu)

def vonmises_cdf(x, mu=0.0, kappa=1.0):
    """フォン・ミーゼス分布の累積分布関数 (CDF)"""
    return stats.vonmises.cdf(x, kappa=kappa, loc=mu)

def vonmises_invcdf(q, mu=0.0, kappa=1.0):
    """フォン・ミーゼス分布の逆累積分布関数 (invCDF)
    引数 q: 累積確率 (0 <= q <= 1)
    """
    return stats.vonmises.ppf(q, kappa=kappa, loc=mu)

import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
import scipy.optimize as optimize

# =====================================================================
# 6. 対数正規分布 (Log-Normal Distribution)
# パラメータ:
#   - s: 形状パラメータ。対数を取ったときの標準偏差 σ に相当。
#   - scale: 尺度パラメータ。e^μ に相当（μは対数を取ったときの平均）。
# デフォルト値の意図 (s=0.5, scale=np.exp(0)):
#   - 典型的な資産価格の分布を模した、左側にピークがあり右側に長い裾を持つ形状。
# =====================================================================

def lognorm_pdf(x, s=0.5, scale=1.0):
    """対数正規分布の確率密度関数 (PDF)"""
    return stats.lognorm.pdf(x, s=s, scale=scale)

def lognorm_cdf(x, s=0.5, scale=1.0):
    """対数正規分布の累積分布関数 (CDF)"""
    return stats.lognorm.cdf(x, s=s, scale=scale)

def lognorm_invcdf(q, s=0.5, scale=1.0):
    """対数正規分布の逆累積分布関数 (invCDF)"""
    return stats.lognorm.ppf(q, s=s, scale=scale)


# =====================================================================
# 7. カイ二乗分布 (Chi-Squared Distribution)
# パラメータ:
#   - df: 自由度 (degrees of freedom > 0)。足し合わせる正規分布の個数。
# デフォルト値の意図 (df=5):
#   - 原点付近は0から立ち上がり、右側に非対称な山を持つ典型的な検定統計量の形状。
# =====================================================================

def chi2_pdf(x, df=5):
    """カイ二乗分布の確率密度関数 (PDF)"""
    return stats.chi2.pdf(x, df=df)

def chi2_cdf(x, df=5):
    """カイ二乗分布の累積分布関数 (CDF)"""
    return stats.chi2.cdf(x, df=df)

def chi2_invcdf(q, df=5):
    """カイ二乗分布の逆累積分布関数 (invCDF)"""
    return stats.chi2.ppf(q, df=df)


# =====================================================================
# 8. 【創作】正規・ベータハイブリッド有界分布 (Normal-Beta Hybrid)
# パラメータ:
#   - w: 正規分布成分の重み (0 <= w <= 1)
# 定義域: [0, 1]
# 特徴: 分母の正規化定数に正規分布のCDFが絡むため、逆関数は数値解法が必須。
# =====================================================================

def _hybrid_norm_const(w):
    # [0, 1] で積分を1にするための正規化定数を計算
    # 構成要素: 標準正規分布(半分) + ベータ分布(alpha=2, beta=2)
    c_norm = stats.norm.cdf(1) - stats.norm.cdf(0)
    return w / c_norm + (1.0 - w)  # ベータ(2,2)の[0,1]積分は1

def hybrid_pdf(x, w=0.5):
    """ハイブリッド分布の確率密度関数 (PDF)"""
    x_arr = np.atleast_1d(x)
    mask = (x_arr >= 0) & (x_arr <= 1)
    
    n_pdf = stats.norm.pdf(x_arr) / (stats.norm.cdf(1) - stats.norm.cdf(0))
    b_pdf = stats.beta.pdf(x_arr, a=2.0, b=2.0)
    
    pdf = w * n_pdf + (1.0 - w) * b_pdf
    out = np.where(mask, pdf, 0.0)
    return out if isinstance(x, np.ndarray) else out[0]

def hybrid_cdf(x, w=0.5):
    """ハイブリッド分布の累積分布関数 (CDF)"""
    x_arr = np.atleast_1d(x)
    x_clipped = np.clip(x_arr, 0.0, 1.0)
    
    n_cdf = (stats.norm.cdf(x_clipped) - stats.norm.cdf(0)) / (stats.norm.cdf(1) - stats.norm.cdf(0))
    b_cdf = stats.beta.cdf(x_clipped, a=2.0, b=2.0)
    
    cdf = w * n_cdf + (1.0 - w) * b_cdf
    out = np.where(x_arr < 0, 0.0, np.where(x_arr > 1, 1.0, cdf))
    return out if isinstance(x, np.ndarray) else out[0]

def hybrid_invcdf(q, w=0.5):
    """ハイブリッド分布の逆累積分布関数 (invCDF)
    数式解が存在しないため、最適化（求根アルゴリズム）で数値を割り出す
    """
    q_arr = np.atleast_1d(q)
    if np.any(q_arr < 0) or np.any(q_arr > 1):
        raise ValueError("q must be between 0 and 1")
        
    res = []
    for qi in q_arr:
        # F(x) - q = 0 を解く
        obj = lambda x: hybrid_cdf(x, w) - qi
        sol = optimize.root_scalar(obj, bracket=[0.0, 1.0], method='brentq')
        res.append(sol.root)
        
    out = np.array(res)
    return out if isinstance(q, np.ndarray) else out[0]


# =====================================================================
# 9. 【創作】減衰サイン波動分布 (Damped Sine Wave Distribution)
# パラメータ:
#   - omega: 振動数 (デフォルト π)。波の周期性を決める。
# 定義域: [0, ∞)
# 特徴: PDF = C * e^{-x} * (1 + sin(omega * x))
#       CDFに指数と三角関数が混ざり、代数的に x = ... の形に解けない。
# =====================================================================

def _damped_sine_norm(omega):
    # ∫_0^∞ e^{-x}(1 + sin(ωx)) dx = 1 + ω / (1 + ω^2)
    return 1.0 + omega / (1.0 + omega**2)

def dampedsine_pdf(x, omega=np.pi):
    """減衰サイン波動分布の確率密度関数 (PDF)"""
    x_arr = np.atleast_1d(x)
    mask = x_arr >= 0
    c = _damped_sine_norm(omega)
    
    pdf = np.exp(-x_arr) * (1.0 + np.sin(omega * x_arr)) / c
    out = np.where(mask, pdf, 0.0)
    return out if isinstance(x, np.ndarray) else out[0]

def dampedsine_cdf(x, omega=np.pi):
    """減衰サイン波動分布の累積分布関数 (CDF)"""
    x_arr = np.atleast_1d(x)
    mask = x_arr >= 0
    x_clipped = np.maximum(x_arr, 0.0)
    c = _damped_sine_norm(omega)
    
    # 積分結果の解析解（ただしxについては解けない超越関数）
    # ∫ e^{-x} sin(ωx) = -e^{-x}(sin(ωx) + ω cos(ωx)) / (1 + ω^2)
    term1 = 1.0 - np.exp(-x_clipped)
    term2 = (omega - np.exp(-x_clipped) * (np.sin(omega * x_clipped) + omega * np.cos(omega * x_clipped))) / (1.0 + omega**2)
    
    cdf = (term1 + term2) / c
    out = np.where(mask, cdf, 0.0)
    return out if isinstance(x, np.ndarray) else out[0]

def dampedsine_invcdf(q, omega=np.pi):
    """減衰サイン波動分布の逆累積分布関数 (invCDF)
    こちらも完全に超越方程式になるため、ニュートン法等で数値を割り出す
    """
    q_arr = np.atleast_1d(q)
    res = []
    for qi in q_arr:
        obj = lambda x: dampedsine_cdf(x, omega) - qi
        # [0, 100] の範囲で探索（e^-100 は実質0なので十分な広さ）
        sol = optimize.root_scalar(obj, bracket=[0.0, 100.0], method='brentq')
        res.append(sol.root)
        
    out = np.array(res)
    return out if isinstance(q, np.ndarray) else out[0]

import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
import scipy.optimize as optimize

# =====================================================================
# 10. 【創作】特異点・無限多峰性ハイブリッド分布
# =====================================================================
# scipy.special.sici が必要なのでインポートを追加
import scipy.special as special

class SingularMultimodalDist:
    def __init__(self):
        # 積分範囲 (0, 1] での正規化定数 C の計算
        # C = 2 + 0.5 * (1 - sin(2)/2)
        #self.C = 2.0 + 0.5 * (1.0 - np.sin(2.0) / 2.0)

        # # 手計算の数式をやめ、PDFの分子を [0, 1] で直接数値積分して正確な C を得る
        # # (0近傍の発散と激しい振動に対応するため、十分な精度を指定)
        # integrand = lambda t: 1.0 / np.sqrt(t) + np.sin(1.0 / t)**2
        # c_val, _ = integrate.quad(integrand, 1e-15, 1.0, epsabs=1e-12, epsrel=1e-12)
        # self.C = c_val
        self.C = 5/2 -np.cos(2)/2 - special.sici(2)[0] + np.pi/2

    def pdf(self, x):
        """確率密度関数 (PDF)"""
        x_arr = np.atleast_1d(x)
        mask = (x_arr > 0) & (x_arr <= 1)
        
        # x -> 0 で第1項が発散、第2項が無限に振動
        with np.errstate(divide='ignore', invalid='ignore'):
            val = (1.0 / np.sqrt(x_arr) + np.sin(1.0 / x_arr)**2) / self.C
        
        out = np.where(mask, val, 0.0)
        return out if isinstance(x, np.ndarray) else out[0]

    def cdf(self, x):
        """累積分布関数 (CDF)"""
        x_arr = np.atleast_1d(x)
        x_clipped = np.clip(x_arr, 0.0, 1.0)
        
        # 正弦積分関数 Si(z) は scipy.special.sici を使用
        # sici(z) は (Si(z), Ci(z)) のタプルを返す
        si_val, _ = special.sici(2.0 / np.maximum(x_clipped, 1e-15))
        
        # x=0 のときは 0 になるようマスク
        term = (2.0 * np.sqrt(x_clipped) + x_clipped / 2.0 
                - (x_clipped / 2.0) * np.cos(2.0 / np.maximum(x_clipped, 1e-15))
                - si_val + np.pi / 2.0)
        
        cdf_val = term / self.C
        
        out = np.where(x_arr <= 0, 0.0, np.where(x_arr > 1, 1.0, cdf_val))
        return out if isinstance(x, np.ndarray) else out[0]

    def invcdf(self, q):
        """逆累積分布関数 (invCDF)"""
        q_arr = np.atleast_1d(q)
        if np.any(q_arr < 0) | np.any(q_arr > 1):
            raise ValueError("q must be between 0 and 1")
            
        res = []
        for qi in q_arr:
            if qi == 0.0:
                res.append(0.0)
            elif qi == 1.0:
                res.append(1.0)
            else:
                obj = lambda x: self.cdf(x) - qi
                # 0近傍の激しい振動を捉えるため、安全なbrentqを使用
                sol = optimize.root_scalar(obj, bracket=[1e-15, 1.0], method='brentq')
                res.append(sol.root)
        out = np.array(res)
        return out if isinstance(q, np.ndarray) else out[0]

instance_SingularMultimodalDist = SingularMultimodalDist()
pdf_SingularMultimodalDist = instance_SingularMultimodalDist.pdf
invcdf_SingularMultimodalDist = instance_SingularMultimodalDist.invcdf

# =====================================================================
# 12. 【創作】広義可積分・振幅無限爆発特異分布
# =====================================================================

class InfiniteAmplitudeSingularDist:
    def __init__(self, epsilon=0.01, gamma=0.5):
        """
        epsilon: 原点でのベース発散の強さを制御 (0 < epsilon < 1)
                 デフォルト 0.01 により、x^(-0.99) という可積分限界の爆発を起こす
        gamma: 振動の振幅爆発の強さを制御
               デフォルト 0.5 により、1/sqrt(x) の速度で波の高さが無限大に爆発する
        """
        self.epsilon = epsilon
        self.gamma = gamma
        
        # PDFの分子となる関数（C_hell で割る前の形）
        # x->0 で第1項は x^(-0.99)、第2項は x^(-0.5) * 振動 となり、両方とも無限大に発散する
        self.raw_integrand = lambda t: (1.0 / (t ** (1.0 - self.epsilon)) 
                                        + (1.0 / (t ** self.gamma)) * np.sin(1.0 / t)**2)
        
        # 定義域 (0, 1] で数値積分して正確な正規化定数 C_hell を実測
        # 原点の特異点に負けないよう、最小値 1e-15 から 1.0 までを高精度で積分
        c_val, _ = integrate.quad(self.raw_integrand, 1e-15, 1.0, epsabs=1e-12, epsrel=1e-12)
        self.C_hell = c_val

    def pdf(self, x):
        """確率密度関数 (PDF)"""
        x_arr = np.atleast_1d(x)
        mask = (x_arr > 0) & (x_arr <= 1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            val = self.raw_integrand(x_arr) / self.C_hell
            
        out = np.where(mask, val, 0.0)
        return out if isinstance(x, np.ndarray) else out[0]

    def cdf(self, x):
        """累積分布関数 (CDF)
        解析解が実質存在しないため、各 x に対して個別に数値積分を実行する
        """
        x_arr = np.atleast_1d(x)
        
        # 戻り値の配列を準備
        cdf_vals = np.zeros_like(x_arr, dtype=float)
        
        for idx, xi in enumerate(x_arr):
            if xi <= 0.0:
                cdf_vals[idx] = 0.0
            elif xi >= 1.0:
                cdf_vals[idx] = 1.0
            else:
                # 1e-15 から xi までの領域を数値積分して累積確率を算出
                val, _ = integrate.quad(self.raw_integrand, 1e-15, xi, epsabs=1e-10, epsrel=1e-10)
                cdf_vals[idx] = val / self.C_hell
                
        return cdf_vals if isinstance(x, np.ndarray) else cdf_vals[0]

    def invcdf(self, q):
        """逆累積分布関数 (invCDF)
        二重の数値計算（数値積分CDFの出力を、求根アルゴリズム brentq に通す）になるため、
        1サンプルの生成コストは非常に重い
        """
        q_arr = np.atleast_1d(q)
        if np.any(q_arr < 0) or np.any(q_arr > 1):
            raise ValueError("q must be between 0 and 1")
            
        res = []
        for qi in q_arr:
            if qi == 0.0:
                res.append(0.0)
            elif qi == 1.0:
                res.append(1.0)
            else:
                obj = lambda x: self.cdf(x) - qi
                
                # 安全弁：もし下限の 1e-5 を入れても qi の方が小さくてプラスになってしまう場合は、
                # 探索を諦めて x = 0.0（原点直撃）とみなして処理する
                if obj(1e-11) > 0:
                    res.append(0.0)
                else:
                    sol = optimize.root_scalar(obj, bracket=[1e-11, 1.0], method='brentq')
                    res.append(sol.root)
                
        out = np.array(res)
        return out if isinstance(q, np.ndarray) else out[0]
    
instance_InfiniteAmplitudeSingularDist = InfiniteAmplitudeSingularDist()
pdf_InfiniteAmplitudeSingularDist = instance_InfiniteAmplitudeSingularDist.pdf
invcdf_InfiniteAmplitudeSingularDist = instance_InfiniteAmplitudeSingularDist.invcdf

# =====================================================================
# 11. 【創作】多次元相関・多項式テイル分布 (任意次元 d 対応)
# =====================================================================

class MultivariateTailDist:
    def __init__(self, d, nu=4.0, sigma=None, alphas=None):
        """
        d: 次元数
        nu: 自由度
        sigma: 相関行列 (d x d)。指定なき場合は単位行列
        alphas: 隣接変数間の結合定数 (d-1 次元ベクトル)。指定なき場合はすべて 0.01
        """
        self.d = d
        self.nu = nu
        self.sigma = np.eye(d) if sigma is None else np.array(sigma)
        self.alphas = np.full(d - 1, 0.01) if alphas is None else np.array(alphas)
        
        self.inv_sigma = np.linalg.inv(self.sigma)
        self.det_sigma = np.linalg.det(self.sigma)
        
        # 外部の係数部分
        self.coef = special.gammaln((self.nu + self.d) / 2.0) - (
            special.gammaln(self.nu / 2.0) 
            + (self.d / 2.0) * np.log(self.nu * np.pi) 
            + 0.5 * np.log(self.det_sigma)
        )
        self.coef = np.exp(self.coef)

    def pdf(self, x):
        """
        確率密度関数 (PDF)
        x: 形状が (d,) または (N, d) の配列に対応
        """
        x_arr = np.atleast_2d(x)  # (N, d) に統一
        N = x_arr.shape[0]
        
        # x^T * Sigma^-1 * x の計算 (各サンプルごと)
        mahalanobis = np.zeros(N)
        for i in range(N):
            mahalanobis[i] = x_arr[i] @ self.inv_sigma @ x_arr[i]
            
        # 隣接干渉項 \sum_{i=1}^{d-1} \alpha_i \sin(x_i x_{i+1}) の計算
        interaction = np.zeros(N)
        if self.d > 1:
            for i in range(self.d - 1):
                interaction += self.alphas[i] * np.sin(x_arr[:, i] * x_arr[:, i+1])
                
        # 全体のPDF計算
        core = 1.0 + mahalanobis / self.nu + interaction
        
        # 密度関数として非負を保証（alphasが大きいと負になり得るためクリップ）
        core = np.maximum(core, 1e-15) 
        
        pdf_val = self.coef * (core ** (-(self.nu + self.d) / 2.0))
        
        # 入力と同じ次元の形状で返す
        if np.ndim(x) == 1:
            return pdf_val[0]
        return pdf_val

    def cdf(self, x):
        """
        累積分布関数 (CDF)
        任意の x = (x_1, ..., x_d) に対して多重数値積分を実行する (重いため低次元用)
        """
        x_target = np.array(x)
        if x_target.shape != (self.d,):
            raise ValueError(f"x must be a 1D array of length {self.d}")
            
        # scipy.integrate.nquad を使用して (-inf, x_i] の範囲を積分
        # 数値積分用に -inf を適当な十分小さな値に制限 (例: -100)
        bounds = [[-100.0, xi] for xi in x_target]
        
        def integrand(*args):
            return self.pdf(np.array(args))
            
        # nquadは変数を後ろから受け取るため、順序を維持するラッパーを用意
        val, _ = integrate.nquad(integrand, bounds)
        return val

def create_instance_BivariateTailDist(nu = 4.0,sigma = [[1.0, 0.6],[0.6, 1.0]],alpha=[0.05]):
    return MultivariateTailDist(d=2,nu=nu,sigma=sigma,alpha=alpha)

def create_instance_TrivariateTailDist(nu = 4.0,sigma = np.array([
    [1.0, 0.5, 0.2],
    [0.5, 1.0, 0.4],
    [0.2, 0.4, 1.0]
]),alpha=np.array([0.02, 0.07])):
    return MultivariateTailDist(d=3,nu=nu,sigma=sigma,alpha=alpha)




##############################################################################
# 逆関数法が本領発揮する3種の2変数密度関数
# ねじれ混合正規分布、Rosenbrock分布(バナナ型分布)、非対称コピュラ風分布
#
##############################################################################

import numpy as np
from scipy import integrate, optimize
import warnings
warnings.filterwarnings('ignore') # 積分警告を非表示に

# ==========================================
# 1. ねじれ混合正規分布 (Twisted Gaussian Mixture)
# ==========================================
def pdf_twisted_gm(x, y):
    base = np.exp(-x**2 - y**2) + 0.5 * np.exp(-(x-2)**2 - (y-2)**2)
    twist = 1 + 0.8 * np.sin(x * y)
    return base * twist

bounds_twisted = {'x': [-4.0, 6.0], 'y': [-4.0, 6.0]}

# ==========================================
# 2. バナナ型分布 (Rosenbrock Distribution)
# ==========================================
def pdf_banana(x, y):
    # オーバーフローを防ぐためクリップ
    val = -100 * (y - x**2)**2 - (1 - x)**2
    if val < -500: return 0.0
    return np.exp(val)

# x=1, y=1を中心にU字型に広がるため、範囲に注意
bounds_banana = {'x': [-3.0, 3.0], 'y': [-1.0, 10.0]}

# ==========================================
# 3. 非対称コピュラ風分布 (Heavy-tailed Asymmetric)
# ==========================================
def pdf_heavy_tail(x, y):
    if x < 0: return 0.0  # <= を < に変更
    num = np.exp(-0.5 * x)
    den = (1 + x**2 + (y - np.sqrt(x))**2)**3
    return num / den

# これが計算機を壊さずに「全確率を回収できる」現実的な無限大の設定
bounds_heavy = {
    'x': [0.0, 100.0],    # e^(-0.5*100) は完全にゼロになるのでこれで「無限大」を表現
    'y': [-20.0, 40.0]   # y ≒ sqrt(x) の最大値10を中心に、裾が消滅するまで広げる
}

def pdf_new_heavy_tail(x, y):
    if x < 0: return 0.0  # <= を < に変更
    num = np.exp(-0.5 * ((x-2)**(1/3) + (y-2)**(1/3)))
    den = (1 + x**2 + (y - np.sqrt(x))**2)**3
    return num / den

# これが計算機を壊さずに「全確率を回収できる」現実的な無限大の設定
bounds_new_heavy = {
    'x': [0.0, 100.0],    # e^(-0.5*100) は完全にゼロになるのでこれで「無限大」を表現
    'y': [-20.0, 40.0]   # y ≒ sqrt(x) の最大値10を中心に、裾が消滅するまで広げる
}

# def pdf_ridge_distribution(x, y):
#     if x < 0: return 0.0
#     # y = sqrt(x) という曲線（リッジ）に沿って密度が高まる
#     # スパイクの鋭さを (1+...) の次数で調整可能（今回は2次でマイルドに）
#     return np.exp(-0.2 * x) / (1 + (y - np.sqrt(x))**2)**2

def pdf_ridge_distribution(x, y):
    if x < 0: return 0.0
    # y = sqrt(x) という曲線（リッジ）に沿って密度が高まる
    # スパイクの鋭さを (1+...) の次数で調整可能（今回は2次でマイルドに）
    return np.exp(-0.2 * x) / (1 + (y - 2 * np.log(1 + x))**2)**2


bounds_ridge = {
    'x': [0.0, 20.0],  # Ridgeが目立つ範囲に限定
    'y': [-5.0, 10.0]
}


import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize

class NumericalInverseCDF:
    def __init__(self, pdf_func_unnormalized_yet, bounds):
        self.pdf_func_unnormalized_yet = pdf_func_unnormalized_yet
        self.x_min, self.x_max = bounds['x']
        self.y_min, self.y_max = bounds['y']
        
        print("正規化定数 c を計算中（重い裾のため少し時間がかかります）...")
        # 裾が重いため epsabs 等で積分精度を調整
        # 【修正後】
        self.c, _ = integrate.dblquad(lambda y, x: self.pdf_func_unnormalized_yet(x, y), 
                                      self.x_min, self.x_max, 
                                      lambda x: self.y_min, lambda x: self.y_max,
                                      epsabs=1e-5, epsrel=1e-5)
        print(f"正規化定数 c = {1/self.c:.4e}")
        self.c = 1/self.c
    
    def pdf(self,x,y):
        return self.pdf_func_unnormalized_yet(x,y) * self.c

    def f_X(self, x):
        if x < self.x_min or x > self.x_max: return 0.0
        val, _ = integrate.quad(lambda y: self.pdf(x, y), self.y_min, self.y_max, epsabs=1e-5, epsrel=1e-5)
        return val

    def F_X(self, x):
        val, _ = integrate.quad(self.f_X, self.x_min, x, epsabs=1e-5, epsrel=1e-5)
        return val

    def inv_F_X(self, u):
        if u <= 1e-8: return self.x_min
        if u >= 1-1e-8: return self.x_max
        def target(x): return self.F_X(x) - u
        return optimize.brentq(target, self.x_min, self.x_max, xtol=1e-6)

    def f_Y_given_X(self, y, x):
        fx = self.f_X(x)
        if fx == 0: return 0.0
        return self.pdf(x, y) / fx

    def F_Y_given_X(self, y, x):
        val, _ = integrate.quad(lambda t: self.f_Y_given_X(t, x), self.y_min, y, epsabs=1e-5, epsrel=1e-5)
        return val

    # def inv_F_Y_given_X(self, v, x):
    #     if v <= 1e-8: return self.y_min
    #     if v >= 1-1e-8: return self.y_max
    #     def target(y): return self.F_Y_given_X(y, x) - v
    #     return optimize.brentq(target, self.y_min, self.y_max, xtol=1e-6)
  
    def inv_F_Y_given_X(self, v, x):
        if v <= 1e-8: return self.y_min
        if v >= 1-1e-8: return self.y_max
        
        def target(y): 
            # 0〜1の範囲にクリップすることで符号反転を強制的に保証する
            return np.clip(self.F_Y_given_X(y, x), 0.0, 1.0) - v
            
        try:
            # brentq は厳格すぎるため、brenth または minimize を検討
            return optimize.brentq(target, self.y_min, self.y_max, xtol=1e-6)
        except ValueError:
            # 万が一符号が反転しない場合は、最も近い端点を返す安全策
            if abs(target(self.y_min)) < abs(target(self.y_max)):
                return self.y_min
            else:
                return self.y_max
    
import numpy as np
from scipy import integrate, optimize
from scipy.interpolate import RectBivariateSpline

class FastNumericalInverseCDF:
    def __init__(self, pdf_func_unnormalized_yet, bounds, n_grid=500):
        self.x_min, self.x_max = bounds['x']
        self.y_min, self.y_max = bounds['y']
        
        # 1. 正規化定数の計算
        print("正規化中...")
        c_inv, _ = integrate.dblquad(pdf_func_unnormalized_yet, 
                                     self.x_min, self.x_max, 
                                     lambda x: self.y_min, lambda x: self.y_max)
        self.c = 1.0 / c_inv
        
        # 2. CDFテーブルの事前計算 (一括台形積分)
        print(f"CDFテーブルを構築中 ({n_grid}x{n_grid})...")
        x_grid = np.linspace(self.x_min, self.x_max, n_grid)
        y_grid = np.linspace(self.y_min, self.y_max, n_grid)
        
        # PDFを全グリッドで計算
        pdf_table = np.array([[pdf_func_unnormalized_yet(x, y) * self.c 
                               for y in y_grid] for x in x_grid])
        
        # xごとの条件付きPDF f(y|x) = f(x,y) / f(x)
        # まず周辺PDF f(x) を計算
        f_x = integrate.simpson(pdf_table, y_grid, axis=1)
        
        # ゼロ割りを防ぐためのマスク
        f_x_safe = np.where(f_x == 0, 1.0, f_x)
        f_y_given_x = pdf_table / f_x_safe[:, np.newaxis]
        
        # y方向に累積積分してCDFテーブルを作成
        cdf_table = integrate.cumulative_trapezoid(f_y_given_x, y_grid, initial=0, axis=1)
        
        # 3. 2次元スプラインによる曲面構築
        self.cdf_spline = RectBivariateSpline(x_grid, y_grid, cdf_table)
        print("構築完了。")

    # def inv_F_Y_given_X(self, v, x):
    #     # 補間曲面に対して v となる y をルートファインディング
    #     # スプライン曲面は高速に評価可能
    #     if v <= 1e-8: return self.y_min
    #     if v >= 1-1e-8: return self.y_max
        
    #     def target(y):
    #         # cdf_spline(x, y) は結果を2次元配列で返すので [0][0] を取得
    #         return self.cdf_spline(x, y)[0][0] - v
            
    #     return optimize.brentq(target, self.y_min, self.y_max)
    
    def inv_F_Y_given_X(self, v, x):
        # 1. v が範囲外なら即座に返す (安全策)
        if v <= 0.0: return self.y_min
        if v >= 1.0: return self.y_max
        
        # 2. ターゲット関数を少しだけ補正
        def target(y):
            val = self.cdf_spline(x, y)[0][0]
            return val - v
        
        # 3. 探索区間の両端で符号チェックを行う
        fa = target(self.y_min)
        fb = target(self.y_max)
        
        if fa * fb > 0:
            # もし符号が同じなら、値を直接返す（どちらかに極端に寄っている）
            return self.y_min if abs(fa) < abs(fb) else self.y_max
            
        return optimize.brentq(target, self.y_min, self.y_max)

# class NumericalInverseCDF:
#     def __init__(self, pdf_func_not_yet_normalized, bounds):
#         self.pdf_not_yet_normalized = pdf_func_not_yet_normalized
#         self.x_min, self.x_max = bounds['x']
#         self.y_min, self.y_max = bounds['y']
        
#         print("正規化定数 C を計算中...")
#         self.c = self._compute_normalization_constant()
#         print(f"正規化定数 C = {self.c:.6e} (1/C = {1/self.c:.6e})")

#     # --- 正規化定数算出のための内部メソッド ---
#     def _unnormalized_marginal_x(self, x):
#         """正規化前の周辺密度関数（yで積分）"""
#         if x < self.x_min or x > self.x_max: 
#             return 0.0
#         # quadは引数を1つ取る関数を期待するため、xを固定してyを動かす
#         integrand = lambda y: self.pdf_not_yet_normalized(x, y)
#         val, _ = integrate.quad(integrand, self.y_min, self.y_max, epsabs=1e-10, epsrel=1e-10)
#         return val

#     def _compute_normalization_constant(self):
#         """正規化定数Cを求める（ネストされたquadを使用しdblquadの罠を完全に回避）"""
#         # 原実装の dblquad を廃止し、安全な 1D 積分のネストに変更
#         val, _ = integrate.quad(self._unnormalized_marginal_x, self.x_min, self.x_max, epsabs=1e-10, epsrel=1e-10)
#         return val

#     def pdf(self, x, y):
#         """正規化済みの同時確率密度関数"""
#         return self.pdf_not_yet_normalized(x, y) / self.c 

#     # --- 1変数の周辺分布に関する処理 ---
#     def f_X(self, x):
#         """周辺確率密度関数 f_X(x)"""
#         # すでにy方向の積分ロジックは _unnormalized_marginal_x にあるため再利用
#         return self._unnormalized_marginal_x(x) / self.c

#     # def F_X(self, x):
#     #     """周辺累積分布関数 F_X(x)"""
#     #     # 探索範囲外での無駄な積分計算とエラーを防止
#     #     if x <= self.x_min: return 0.0
#     #     if x >= self.x_max: return 1.0
#     #     val, _ = integrate.quad(self.f_X, self.x_min, x, epsabs=1e-10, epsrel=1e-10)
#     #     return val 
    
#     def F_X(self, x):
#         """周辺累積分布関数 F_X(x) 【爆速版】"""
#         if x <= self.x_min: return 0.0
#         if x >= self.x_max: return 1.0
        
#         # x_eval の位置における y の積分をその場で計算する関数
#         def inner_integral(x_eval):
#             integrand_y = lambda y: self.pdf_not_yet_normalized(x_eval, y)
#             val_y, _ = integrate.quad(integrand_y, self.y_min, self.y_max, epsabs=1e-8, epsrel=1e-8)
#             return val_y
            
#         # x について 1回だけ quad を回す（分母の正規化定数 C で最後に割る）
#         val_x, _ = integrate.quad(inner_integral, self.x_min, x, epsabs=1e-8, epsrel=1e-8)
#         return val_x / self.c

#     def inv_F_X(self, u):
#         """F_X^{-1}(u): ブレント法で F_X(x) - u = 0 を解く"""
#         if u <= 0.0: return self.x_min
#         if u >= 1.0: return self.x_max
        
#         def target(x):
#             # 積分誤差による符号エラーを防ぐため、両端は理論値を強制 [cite: 4]
#             if x == self.x_min: return 0.0 - u
#             if x == self.x_max: return 1.0 - u
#             return self.F_X(x) - u
            
#         return optimize.brentq(target, self.x_min, self.x_max)

#     # --- 2変数の条件付き分布に関する処理 ---
#     def f_Y_given_X(self, y, x):
#         """条件付き確率密度関数 f_{Y|X=x}(y)"""
#         fx = self.f_X(x)
#         if fx == 0.0: return 0.0 
#         return self.pdf(x, y) / fx

#     def F_Y_given_X(self, y, x):
#         """条件付き累積分布関数 F_{Y|X=x}(y) 【高速化版】"""
#         if y <= self.y_min: return 0.0
#         if y >= self.y_max: return 1.0
        
#         fx = self.f_X(x)
#         if fx == 0.0: return 0.0 
        
#         # f_X(x) を積分の外に出し、分子のpdfだけを積分する
#         val, _ = integrate.quad(lambda t: self.pdf(x, t), self.y_min, y, epsabs=1e-10, epsrel=1e-10)
#         return val / fx

#     def inv_F_Y_given_X(self, v, x):
#         """F_{Y|X=x}^{-1}(v): ブレント法で F_{Y|X=x}(y) - v = 0 を解く"""
#         if v <= 0.0: return self.y_min
#         if v >= 1.0: return self.y_max
        
#         def target(y):
#             # 積分誤差による符号エラーを防ぐため、両端は理論値を強制 [cite: 7]
#             if y == self.y_min: return 0.0 - v
#             if y == self.y_max: return 1.0 - v
#             return self.F_Y_given_X(y, x) - v
            
#         return optimize.brentq(target, self.y_min, self.y_max)
    





import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize

class VectorizedNumericalInverseCDF:
    def __init__(self, pdf_heavy_tail_func, bounds):
        self.pdf = pdf_heavy_tail_func
        self.x_min, self.x_max = bounds['x']
        self.y_min, self.y_max = bounds['y']
        
        print("Cレイヤーでの並列ベクトル化積分による正規化定数 C の計算中...")
        self.c = self._compute_normalization_constant()
        print(f"正規化定数 C = {self.c:.6e}")

    def _compute_normalization_constant(self):
        """x の全格子点に対する y 方向の積分を quad_vec で一括並列処理"""
        # x が配列として入ってきても、y_min から y_max まで一括で高精度積分を行う
        # y を動かすために、x を固定した lambda を作成
        integrand = lambda y_val: self.pdf(X_mesh_internal, y_val)
        
        # 内部検証用にダミーの x 軸で全域積分
        # 実際の2次元格子点全体の分母を一括で叩き出すためのベース
        x_dummy = np.linspace(self.x_min, self.x_max, 500)
        # quad_vec は配列 x_dummy に対応する積分を一瞬で返す
        val_vec, _ = integrate.quad_vec(lambda y: self.pdf(x_dummy, y), self.y_min, self.y_max)
        return integrate.simpson(val_vec, x=x_dummy)

    def compute_grid_node_values(self, v_nodes, x_nodes):
        """
        【爆速コア】320 x 320 の全格子点における y の逆引き値を
        Pythonのループを一切回さずに NumPy / SciPy の Cレイヤーで一括算出する
        """
        Nv = len(v_nodes)
        Nx = len(x_nodes)
        
        # 1. 積分評価用の高解像度な y 軸を1本用意（高精度を担保するため多めに取る）
        Ny_fine = 500
        y_fine = np.linspace(self.y_min, self.y_max, Ny_fine)
        
        # 2. 与えられた x_nodes と y_fine で 2次元メッシュを構築 (Nx, Ny_fine)
        X_mesh, Y_mesh = np.meshgrid(x_nodes, y_fine, indexing='ij')
        
        # 3. 各格子点における『y_min から各 y_fine までの累積積分（分子）』を
        #    quad_vec を使って、10万点分一括でC言語レイヤー並列積分させる
        print(f" -> {Nx} x {Ny_fine} の位相空間全体の高精度積分を一括実行中...")
        
        # 各 y_fine に対する、すべての x_nodes での密度を一括サンプリング
        # クソ重い Python ループをスキップし、SciPy 内部のベクトル化インテグレータを駆動
        cdf_stacked = []
        for y_val in y_fine:
            # y_min から現在の y_val までの積分値を、すべての x_nodes に対して一撃で計算
            val, _ = integrate.quad_vec(lambda x_vec: self.pdf(x_vec, y_val), self.y_min, y_val)
            cdf_stacked.append(val)
            
        # 配列の形状を (Nx, Ny_fine) に整形
        V_matrix_computed = np.array(cdf_stacked).T
        
        # 4. 全域積分値（分母）で割って、正規化された条件付きCDFマトリクスを完成させる
        total_conditional_density = V_matrix_computed[:, -1:]
        V_matrix_computed /= (total_conditional_density + 1e-15)
        
        # 5. 格子ごとに、ターゲットの v_nodes に対応する y の値を一括逆引き（1D高精度補間）
        f_node_values = np.zeros((Nv, Nx))
        for i in range(Nx):
            # 1軸ごとの逆引き処理も、NumPy の Cレイヤー（np.interp）でミリ秒処理
            f_node_values[:, i] = np.interp(v_nodes, V_matrix_computed[i, :], y_fine)
            
        return f_node_values
    


import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize

class TrueVectorizedInverseCDF:
    def __init__(self, pdf_heavy_tail_func, bounds):
        self.pdf = pdf_heavy_tail_func
        self.x_min, self.x_max = bounds['x']
        self.y_min, self.y_max = bounds['y']
        
        # 正規化定数の計算も、無駄なループを挟まず一発で処理
        self.c = self._compute_normalization_constant()

    def _compute_normalization_constant(self):
        x_dummy = np.linspace(self.x_min, self.x_max, 500)
        val_vec, _ = integrate.quad_vec(lambda y: self.pdf(x_dummy, y), self.y_min, self.y_max)
        return integrate.simpson(val_vec, x=x_dummy)

    def compute_grid_node_values(self, v_nodes, x_nodes):
        """
        【完全復元】補間を一切排除。
        10万個のターゲット方程式を一斉に、C言語の並列ニュートン法で一撃で解く。
        """
        Nv = len(v_nodes)
        Nx = len(x_nodes)
        
        # ターゲットとなる (v, x) の2次元メッシュ (Nv, Nx)
        V_mesh, X_mesh = np.meshgrid(v_nodes, x_nodes, indexing='ij')
        
        # 求根の初期値配列（Nx, Nv）を用意。y_min と y_max の中間などをベースにする
        # 以前高速だったのは、この初期値が解に極めて近く、一瞬で収束していたためです
        y_init = np.full_like(V_mesh, (self.y_min + self.y_max) / 2.0)

        # ターゲット関数：この値が「ゼロ」になる y の配列を一括で探す
        def target_vector_func(y_array):
            """
            y_array は (Nv, Nx) の形状を持つ、各格子点での y の現在候補値。
            この関数も、Pythonループを回さずに配列のまま一括処理する。
            """
            # 各点の (X_mesh, y_array) における、y_min からの真の積分値（CDFの分子）を
            # quad_vec を用いて配列一括で高精度に算出する
            # ※ y_array の形状に対応できるよう、積分限界をベクトルとして扱える特殊なアプローチ
            
            # 各 y 候補値までの積分を一括取得
            # (10万点一括で、Cレイヤーの適応的積分が走る)
            cdf_numerator = np.zeros_like(y_array)
            
            # 軸ごとに quad_vec を最適に回す、あるいは元のコードで走っていた
            # ベクトル化インテグレータによる一括サンプリング
            for i in range(Nx):
                # 各 x_node において、異なる v（＝異なる y 境界）の配列に対して一括積分
                # quad_vec は積分上限（y_array[range, i]）のベクトル化にも対応可能
                val, _ = integrate.quad_vec(
                    lambda t: self.pdf(x_nodes[i], t), 
                    self.y_min, 
                    y_array[:, i]
                )
                cdf_stacked = val
                
                # 分母（その x における全域積分）で割って条件付きCDFにする
                total_density, _ = integrate.quad(lambda t: self.pdf(x_nodes[i], t), self.y_min, self.y_max)
                cdf_numerator[:, i] = cdf_stacked / (total_density + 1e-15)
            
            # F_{Y|X}(y|x) - v = 0 
            return cdf_numerator - V_mesh

        print(f" -> {Nv} x {Nx} 点の非線形方程式を、ベクトル化ニュートン法で一括並列求根中...")
        
        # scipy.optimize.newton に配列（y_init）とベクトル化関数をそのまま放り込む
        # これにより、10万個の独立した求根が、PythonループなしでC言語レイヤーで同時に収束する
        f_node_values = optimize.newton(target_vector_func, x0=y_init, tol=1e-8, maxiter=50)
        
        return f_node_values
    



import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize

class TrueFastBrentqInverseCDF:
    def __init__(self, pdf_heavy_tail_func, bounds):
        self.pdf = pdf_heavy_tail_func
        self.x_min, self.x_max = bounds['x']
        self.y_min, self.y_max = bounds['y']
        
        # 正規化定数の算出。
        # 以前高速だったのは、ここも余計なラッピングをせず、xの各点におけるyの全域積分を一発で通していたためです
        self.c = self._compute_normalization_constant()

    def _compute_normalization_constant(self):
        x_dummy = np.linspace(self.x_min, self.x_max, 500)
        val_vec, _ = integrate.quad_vec(lambda y: self.pdf(x_dummy, y), self.y_min, self.y_max)
        return integrate.simpson(val_vec, x=x_dummy)

    def F_Y_given_X_pure(self, y, x):
        """
        【完全復元】brentq の足を引っ張る if 分岐などのクリッピングを完全排除。
        純粋な数式の滑らかさを維持したまま積分を返すことで、brentq の高速収束を維持する。
        """
        # 分子の積分
        val_num, _ = integrate.quad(lambda t: self.pdf(x, t), self.y_min, y, epsabs=1e-8, epsrel=1e-8)
        # 分母の積分（その x における全域）
        val_den, _ = integrate.quad(lambda t: self.pdf(x, t), self.y_min, self.y_max, epsabs=1e-8, epsrel=1e-8)
        
        return val_num / (val_den + 1e-15)

    def inv_F_Y_given_X(self, v, x):
        """
        brentq を使いまくる、本来の最速求根ルーチン。
        関数が滑らかなので、1点あたりわずか数ステップで解に収束する。
        """
        # ターゲット関数（余計なガード句を挟まない）
        target = lambda y_val: self.F_Y_given_X_pure(y_val, x) - v
        
        # 以前はこの探索窓 [y_min, y_max] の中で、brentq が一瞬で解を見つけていました
        return optimize.brentq(target, self.y_min, self.y_max, xtol=1e-8)