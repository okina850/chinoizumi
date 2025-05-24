import torch
import pandas as pd
import matplotlib.pyplot as plt
from models.neural_net import OptionPricingNN
from utils.bs_formula import bs_call_price

# データ読み込み
df = pd.read_csv('data/sample_data.csv')

# ブラック・ショールズ理論価格
sigma = 0.2  # 固定とする
df['C_bs'] = df.apply(lambda row: bs_call_price(row['S'], row['K'], row['T'], row['r'], sigma), axis=1)

# PyTorchモデルで予測
model = OptionPricingNN()
model.load_state_dict(torch.load('option_model.pth'))
model.eval()

X = torch.tensor(df[['S', 'K', 'T', 'r']].values, dtype=torch.float32)
with torch.no_grad():
    pred_nn = model(X).numpy().flatten()

df['C_nn'] = pred_nn

# 比較プロット
plt.figure(figsize=(8, 5))
plt.scatter(df['C_market'], df['C_bs'], label='BS Model', alpha=0.6)
plt.scatter(df['C_market'], df['C_nn'], label='NN Model', alpha=0.6)
plt.plot([df['C_market'].min(), df['C_market'].max()],
         [df['C_market'].min(), df['C_market'].max()],
         'k--', label='Perfect Fit')
plt.xlabel('Market Price')
plt.ylabel('Model Price')
plt.legend()
plt.title('Model vs Market Option Prices')
plt.tight_layout()
plt.show()
