import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from models.neural_net import OptionPricingNN
import torch.nn as nn
import torch.optim as optim

# データ読み込み
df = pd.read_csv('data/sample_data.csv')
X = torch.tensor(df[['S', 'K', 'T', 'r']].values, dtype=torch.float32)
y = torch.tensor(df['C_market'].values, dtype=torch.float32).view(-1, 1)

# データローダ
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# モデルと訓練設定
model = OptionPricingNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習
for epoch in range(100):
    for batch_X, batch_y in loader:
        pred = model(batch_X)
        loss = criterion(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

# 保存
torch.save(model.state_dict(), 'option_model.pth')
