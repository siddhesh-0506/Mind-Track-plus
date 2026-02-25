import pandas as pd
import random

print(">>> GENERATING DATASET <<<")

positive_templates = [
    "I feel fantastic today!",
    "I'm really happy with how things are going.",
    "Everything is working out perfectly.",
    "I'm excited and full of energy.",
    "This is the best day I've had in a long time.",
    "I feel confident and motivated.",
    "Life is good and I'm grateful.",
    "I'm laughing and smiling a lot today.",
    "Things are improving for me.",
    "I feel supported and appreciated.",
]

neutral_templates = [
    "Today was okay, nothing special.",
    "I'm feeling normal, not too good, not too bad.",
    "It's just a regular day.",
    "Nothing much happened today.",
    "I don't feel strongly about anything right now.",
    "It was an average day for me.",
    "I feel kind of neutral.",
    "Things are stable and unchanged.",
    "I'm just going through the day.",
    "Nothing major on my mind right now.",
]

negative_templates = [
    "I feel terrible today.",
    "I'm sad and losing motivation.",
    "Everything feels heavy and stressful.",
    "I feel lonely and hopeless.",
    "This is one of the worst days recently.",
    "I'm frustrated and tired of everything.",
    "I feel anxious and overwhelmed.",
    "Nothing is going right and it's draining.",
    "I feel ignored and unimportant.",
    "Life feels really difficult right now.",
]

# Multiply templates to create 600 items each
def expand_templates(templates, count=600):
    data = []
    for i in range(count):
        sentence = random.choice(templates)
        # Add small variations
        variations = [
            "",
            " right now.",
            " today.",
            " these days.",
            " at the moment.",
            " lately.",
            " honestly.",
            " to be honest.",
            " I guess.",
            " I think.",
        ]
        sentence = sentence + random.choice(variations)
        data.append(sentence)
    return data

positive_data = expand_templates(positive_templates)
neutral_data = expand_templates(neutral_templates)
negative_data = expand_templates(negative_templates)

# Build DataFrame
df = pd.DataFrame({
    "text": positive_data + neutral_data + negative_data,
    "label": ["positive"] * 600 + ["neutral"] * 600 + ["negative"] * 600
})

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

# Save
df.to_csv("data/sentiment_data.csv", index=False)

print(">>> DATASET GENERATED & SAVED at data/sentiment_data.csv <<<")
print(df.head())
