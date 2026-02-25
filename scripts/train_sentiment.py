import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

print(">>> SCRIPT LOADED <<<")
print(">>> LOADING DATA <<<")

data = pd.read_csv("data/sentiment_data.csv")

label_map = {"negative": 0, "neutral": 1, "positive": 2}
data["label"] = data["label"].map(label_map)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    data["text"].tolist(),
    data["label"].tolist(),
    test_size=0.2,
    random_state=42
)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(">>> TOKENIZING DATA <<<")

def encode(texts, labels):
    enc = tokenizer(texts, truncation=True, padding=True, max_length=128)
    enc["labels"] = labels
    return enc

train_enc = encode(train_texts, train_labels)
val_enc   = encode(val_texts,   val_labels)

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, enc):
        self.enc = enc
    def __len__(self):
        return len(self.enc["labels"])
    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.enc.items()}

train_ds = SentimentDataset(train_enc)
val_ds   = SentimentDataset(val_enc)

print(">>> LOADING MODEL <<<")

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

args = TrainingArguments(
    output_dir="results",
    per_device_train_batch_size=4,
    num_train_epochs=3
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

print(">>> TRAIN START <<<")
trainer.train()
print(">>> AFTER TRAIN <<<")

trainer.save_model("models/sentiment_model")
print(">>> MODEL SAVED <<<")

tokenizer.save_pretrained("models/sentiment_model")
print(">>> TOKENIZER SAVED <<<")

print(">>> ALL DONE <<<")
