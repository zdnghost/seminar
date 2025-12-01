from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import torch
import kagglehub
import pandas as pd

# =======================
# 1️ Load dataset
# =======================
path = kagglehub.dataset_download("linhlpv/vietnamese-sentiment-analyst")
csv_path = f"{path}/data.csv"
df = pd.read_csv(csv_path, encoding="latin1")
dataset = Dataset.from_pandas(df, preserve_index=False)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
print(dataset)

# =======================
# 2️ Label mapping
# =======================
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {0: "negative", 1: "neutral", 2: "positive"}
vi2en = {
    "NEG": "negative",
    "NEU": "neutral",
    "POS": "positive"
}
# =======================
# 3️ Load PhoBERT tokenizer
# =======================
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

def tokenize(batch):
    texts = [str(x) if x is not None else "" for x in batch["content"]]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize, batched=True)

# =======================
# 4️ Convert labels to IDs
# =======================
def convert_label(batch):
    labels = []
    for s in batch["label"]:
        s_en = vi2en.get(s, s) 
        if s_en not in label2id:
            raise ValueError(f"Unknown label: {s_en}")
        labels.append(label2id[s_en])
    batch["label"] = labels
    return batch

dataset = dataset.map(convert_label, batched=True)

# =======================
# 5️ Load PhoBERT model
# =======================
model = AutoModelForSequenceClassification.from_pretrained(
    "vinai/phobert-base-v2",
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

# =======================
# 6️ Training configuration
# =======================
args = TrainingArguments(
    output_dir="./output_phobert_sentiment",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# =======================
# 7️ Metrics
# =======================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# =======================
# 8️ Trainer
# =======================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# =======================
# 9️ Train
# =======================
trainer.train()

# =======================
# 10️ Save
# =======================
model.save_pretrained("phobert-sentiment")
tokenizer.save_pretrained("phobert-sentiment")

print("Fine-tuning xong! Model lưu tại 'phobert-sentiment'")
