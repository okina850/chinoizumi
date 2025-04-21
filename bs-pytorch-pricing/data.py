# data.py
import torch
import numpy as np

# ブラック・ショールズ理論価格を模倣（簡略化）
def black_scholes_call_price(S, K=100, T=1, r=0.05, sigma=0.2):
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def generate_data():
    S = np.linspace(10, 200, 100)
    price = black_scholes_call_price(S)
    X = torch.tensor(S, dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(price, dtype=torch.float32).reshape(-1, 1)
    return X, y
