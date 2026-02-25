# app.py — MindTrack+ final dashboard
import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import pipeline, AutoConfig
from torch import nn
import numpy as np
import os

st.set_page_config(page_title="MindTrack+ — Mood Dashboard", layout="centered")
st.title("🧠 MindTrack+ — Mood Tracking & Risk")

# ---------------------------
# Paths (adjust if needed)
# ---------------------------
SENT_MODEL = "models/sentiment_model"
RISK_MODEL = "models/risk_model.pth"
LSTM_MODEL = "models/mood_lstm.pth"
HISTORY_CSV = "data/mood_history.csv"

# ---------------------------
# Utilities
# ---------------------------
def ensure_history():
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(HISTORY_CSV):
        pd.DataFrame({"sentiment":[]}).to_csv(HISTORY_CSV, index=False)

def load_history():
    ensure_history()
    df = pd.read_csv(HISTORY_CSV)
    # if only values, ensure column name
    if "sentiment" not in df.columns:
        df = pd.DataFrame({"sentiment": df.iloc[:,0].values})
    return df

def save_history(df):
    df.to_csv(HISTORY_CSV, index=False)

# ---------------------------
# Load Sentiment Pipeline
# ---------------------------
st.sidebar.header("Model Status")
with st.spinner("Loading sentiment model..."):
    try:
        # use top_k=None to get all scores (avoids deprecation)
        sentiment_nlp = pipeline("text-classification", model=SENT_MODEL, tokenizer=SENT_MODEL, top_k=None, device=-1)
        cfg = None
        try:
            cfg = AutoConfig.from_pretrained(SENT_MODEL)
            id2label = getattr(cfg, "id2label", None)
        except Exception:
            id2label = None
        st.sidebar.success("Sentiment model loaded")
    except Exception as e:
        st.sidebar.error("Sentiment load error: " + str(e))
        st.stop()

# ---------------------------
# Load Risk ANN
# ---------------------------
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

risk_model = RiskANN()
risk_loaded = False
if os.path.exists(RISK_MODEL):
    try:
        risk_model.load_state_dict(torch.load(RISK_MODEL))
        risk_model.eval()
        risk_loaded = True
        st.sidebar.success("Risk model loaded")
    except Exception as e:
        st.sidebar.warning("Risk model load failed: " + str(e))
else:
    st.sidebar.info("Risk model missing — predictions disabled")

# ---------------------------
# Load LSTM (simple wrapper)
# ---------------------------
class MoodLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=3):
        super(MoodLSTM, self).__init__()
        self.embed = nn.Embedding(3, 8)
        self.lstm = nn.LSTM(8, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

lstm_loaded = False
lstm_model = MoodLSTM()
if os.path.exists(LSTM_MODEL):
    try:
        lstm_model.load_state_dict(torch.load(LSTM_MODEL))
        lstm_model.eval()
        lstm_loaded = True
        st.sidebar.success("LSTM model loaded")
    except Exception as e:
        st.sidebar.warning("LSTM load failed: " + str(e))
else:
    st.sidebar.info("LSTM model missing — forecasting disabled")

# ---------------------------
# Helper to interpret labels
# ---------------------------
def interpret_label(raw_label, config_id2label=None):
    # Try config mapping if exists
    if config_id2label:
        try:
            # numeric keys possible
            numeric_map = {int(k): v for k, v in config_id2label.items()}
            # if raw_label like "LABEL_2", map index
            if raw_label.startswith("LABEL_"):
                idx = int(raw_label.split("_")[1])
                name = numeric_map.get(idx, None)
                if name:
                    # standardize
                    name = name.capitalize()
                    return name, idx
        except Exception:
            pass
    # fallback: LABEL_*
    if raw_label.startswith("LABEL_"):
        idx = int(raw_label.split("_")[1])
        names = ["Negative","Neutral","Positive"]
        name = names[idx] if idx < len(names) else raw_label
        return name, idx
    # textual label
    if raw_label.upper() in ("NEGATIVE","POSITIVE","NEUTRAL"):
        name = raw_label.capitalize()
        idx = {"NEGATIVE":0,"NEUTRAL":1,"POSITIVE":2}[raw_label.upper()]
        return name, idx
    return raw_label, 1

# ---------------------------
# UI: Input area
# ---------------------------
st.header("📝 Write how you feel (journal)")
input_text = st.text_area("Describe your feelings, thoughts, or how your day went.", height=120)

col1, col2 = st.columns(2)
with col1:
    if st.button("Analyze & Save"):
        if not input_text.strip():
            st.warning("Please enter some text.")
        else:
            # run pipeline (top_k=None returns list of all labels)
            raw = sentiment_nlp(input_text)[0]  # list of dicts
            st.write("**Raw model scores**")
            st.json(raw)

            top = max(raw, key=lambda x: x["score"])
            raw_label = top["label"]
            label_name, label_idx = interpret_label(raw_label, getattr(cfg, "id2label", None))
            st.write(f"**Prediction:** {label_name} (numeric: {label_idx}) — confidence {top['score']:.3f}")

            # append to history
            df = load_history()
            df = pd.concat([df, pd.DataFrame({"sentiment": [label_idx]})], ignore_index=True)
            save_history(df)

            st.success("Saved to history.")

            # risk prediction
            if risk_loaded:
                try:
                    inp = torch.tensor([[float(label_idx)]]).float()
                    r = risk_model(inp).item()
                    st.write(f"Risk score (0-1): {r:.3f}")
                    if r > 0.6:
                        st.error("⚠️ High risk flagged — consider checking in with someone.")
                    else:
                        st.success("😊 Risk low.")
                except Exception as e:
                    st.info("Risk prediction failed: " + str(e))
            else:
                st.info("Risk model not available.")

with col2:
    if st.button("Clear history"):
        save_history(pd.DataFrame({"sentiment": []}))
        st.experimental_rerun()

# ---------------------------
# Show history & timeline
# ---------------------------
st.subheader("📈 Mood timeline")
df = load_history()
if len(df) == 0:
    st.info("No mood history yet. Add an entry to get started.")
else:
    st.write(f"Entries: {len(df)}")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["sentiment"].values, marker="o")
    ax.set_ylim(-0.1, 2.1)
    ax.set_yticks([0,1,2])
    ax.set_yticklabels(["Neg","Neu","Pos"])
    ax.set_xlabel("Day (index)")
    ax.set_ylabel("Sentiment")
    st.pyplot(fig)

    # show table of last 10 entries
    st.write("Last 10 entries (most recent last):")
    st.table(pd.DataFrame({
        "day": list(range(len(df)-10 if len(df)>10 else 0, len(df))),
        "sentiment": df["sentiment"].tail(10).values
    }))

# ---------------------------
# Forecasting using LSTM
# ---------------------------
st.subheader("🔮 Forecast (next-day mood prediction)")
if not lstm_loaded:
    st.info("LSTM model not available for forecasting.")
else:
    seq_len = 5
    if len(df) < seq_len:
        st.info(f"Need at least {seq_len} days of history to forecast (you have {len(df)}).")
    else:
        seq = df["sentiment"].values[-seq_len:]
        with torch.no_grad():
            x = torch.tensor(seq).long().unsqueeze(0)  # shape 1,seq
            out = lstm_model(x)  # logits for 3 classes
            probs = torch.softmax(out, dim=1).numpy().flatten()
            pred_idx = int(np.argmax(probs))
            pred_name = ["Negative","Neutral","Positive"][pred_idx]
            st.write(f"Next-day prediction: **{pred_name}** (scores: {probs.round(3).tolist()})")

# ---------------------------
# Export / Download history
# ---------------------------
st.subheader("💾 Export")
st.write("Download your mood history CSV:")
with open(HISTORY_CSV, "rb") as f:
    st.download_button("Download CSV", data=f, file_name="mood_history.csv", mime="text/csv")

st.caption("App running locally — models must exist in models/*.pth / models/sentiment_model")
