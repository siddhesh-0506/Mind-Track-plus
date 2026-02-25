import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

print(">>> LSTM TRAINING STARTED <<<")

# ----------------------------
# 1. Load or simulate data
# ----------------------------

# Simulate 60 days of sentiment (negative=0, neutral=1, positive=2)
np.random.seed(42)
days = 60
sentiment_scores = np.random.choice([0, 1, 2], size=days)

df = pd.DataFrame({"sentiment": sentiment_scores})
df.to_csv("data/mood_history.csv", index=False)

print(">>> Simulated mood history saved to data/mood_history.csv <<<")

# ----------------------------
# 2. Prepare sequences
# ----------------------------

sequence_length = 5

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(sentiment_scores, sequence_length)

X = torch.tensor(X).long()
y = torch.tensor(y).long()

# ----------------------------
# 3. Define LSTM model
# ----------------------------

class MoodLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=3):
        super(MoodLSTM, self).__init__()
        self.embed = nn.Embedding(3, 8)  # sentiment categories
        self.lstm = nn.LSTM(8, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = MoodLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ----------------------------
# 4. Train
# ----------------------------

print(">>> Training LSTM <<<")

epochs = 25
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

print(">>> Training Completed <<<")

torch.save(model.state_dict(), "models/mood_lstm.pth")
print(">>> LSTM Model Saved at models/mood_lstm.pth <<<")
