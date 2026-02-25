import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("data/risk_data.csv")
X = df["sentiment"].values.reshape(-1, 1).astype(float)
y = df["risk"].values.reshape(-1, 1).astype(float)

X = torch.tensor(X).float()
y = torch.tensor(y).float()

# -----------------------
# Model
# -----------------------
class RiskANN(nn.Module):
    def __init__(self):
        super(RiskANN, self).__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = RiskANN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -----------------------
# Training Loop
# -----------------------
epochs = 600
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss={loss.item():.6f}")

# -----------------------
# Save Model
# -----------------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/risk_model.pth")
print("✔ risk_model.pth saved!")
