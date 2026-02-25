from transformers import pipeline
import json

model_path = "models/sentiment_model"
print("Loading pipeline from:", model_path)
nlp = pipeline("text-classification", model=model_path, tokenizer=model_path, return_all_scores=True)
tests = [
    "I am thrilled today, best day ever!",
    "I feel so sad and hopeless.",
    "It was an okay day, nothing special."
]
out = {}
for t in tests:
    scores = nlp(t)[0]
    out[t] = scores
    print("\nTEXT:", t)
    print("RAW SCORES:", scores)
    top = max(scores, key=lambda x: x["score"])
    print("TOP:", top)
print("\nDone.")
