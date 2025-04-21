# main.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from model import Net
from data import generate_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# データ生成（簡単なcall option）
X, y = generate_data()

model = Net().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

X, y = X.to(device), y.to(device)

for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 可視化
pred = model(X).detach().cpu().numpy()
plt.plot(X.cpu().numpy(), y.cpu().numpy(), label="True")
plt.plot(X.cpu().numpy(), pred, label="Pred")
plt.legend()
plt.savefig("results/pred_vs_true.png")
