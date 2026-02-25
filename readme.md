# 🧠 MindTrack+ — Mood Tracking & Risk Prediction Dashboard

MindTrack+ is an AI-based mood tracking system that analyzes journal text, predicts sentiment, estimates risk level, and forecasts future mood trends using deep learning models.

The project combines:

* 🤖 NLP Sentiment Analysis (DistilBERT)
* 📊 Mood History Tracking
* ⚠️ Risk Prediction (ANN)
* 🔮 Mood Forecasting (LSTM)
* 🌐 Interactive Dashboard (Streamlit)

---

## 🚀 Features

* Analyze daily journal entries using NLP
* Detect **Negative / Neutral / Positive** sentiment
* Save mood history automatically
* Visualize mood timeline
* Predict risk score from sentiment
* Forecast next-day mood using LSTM
* Export mood history as CSV

---

## 🧱 Project Structure

```
MindTrack/
│
├── app.py                     # Streamlit dashboard
│
├── data/
│   ├── sentiment_data.csv
│   ├── risk_data.csv
│   └── mood_history.csv
│
├── models/                    # Generated after training (NOT uploaded)
│   ├── sentiment_model/
│   ├── risk_model.pth
│   └── mood_lstm.pth
│
├── generate_dataset.py
├── train_sentiment.py
├── train_risk.py
├── train_lstm.py
├── extract_features.py
├── test_pipeline.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone <your-repo-link>
cd MindTrack
```

### 2️⃣ Create virtual environment (recommended)

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux**

```bash
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Training (IMPORTANT)

Models are not included in the repository because they are large.

Run these scripts to generate models locally:

```bash
python generate_dataset.py
python train_sentiment.py
python train_risk.py
python train_lstm.py
```

This will create:

```
models/
 ├── sentiment_model/
 ├── risk_model.pth
 └── mood_lstm.pth
```

---

## ▶️ Run the Dashboard

```bash
streamlit run app.py
```

The app will open automatically in your browser.

---

## 📈 How It Works

### 1️⃣ Sentiment Model

* Uses **DistilBERT** from HuggingFace Transformers
* Classifies text into:

  * Negative
  * Neutral
  * Positive

### 2️⃣ Risk Prediction

* Simple Artificial Neural Network (ANN)
* Input: sentiment score
* Output: risk value (0–1)

### 3️⃣ Mood Forecasting

* LSTM model trained on mood history
* Predicts next-day mood trend

---

## 🧾 Requirements

Main libraries used:

* torch
* transformers
* pandas
* numpy
* scikit-learn
* streamlit
* matplotlib

---

## ❗ Notes

* First run may download pretrained models from HuggingFace (~250MB).
* Models must exist in `/models` for full functionality.
* If models are missing, the app will still run but predictions will be disabled.

---

## 🧑‍💻 Author

**Siddhesh Adhalrao**

Computer Science — Cloud Computing Track
AI / ML / Full-stack projects

---

## ⭐ Future Improvements

* Real-world dataset integration
* User authentication
* Cloud deployment (AWS / GCP)
* Better risk modeling
* Multi-user mood analytics

---
