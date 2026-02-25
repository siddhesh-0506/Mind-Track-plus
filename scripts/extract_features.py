import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

print(">>> FEATURE EXTRACTION STARTED <<<")

model_path = "models/sentiment_model"

print(">>> LOADING MODEL <<<")
nlp = pipeline(
    "text-classification",
    model=model_path,
    tokenizer=model_path,
    return_all_scores=True
)

print(">>> MODEL LOADED <<<")

sample_texts = [
    "I am feeling very stressed today.",
    "Everything is going great!",
    "I feel lonely and tired."
]

print(">>> ANALYZING SAMPLE TEXTS <<<")

output = {}
for text in sample_texts:
    scores = nlp(text)[0]     # returns list of label → score
    output[text] = scores

with open("results/sample_features.json", "w") as f:
    json.dump(output, f, indent=4)

print(">>> FEATURES SAVED TO results/sample_features.json <<<")
