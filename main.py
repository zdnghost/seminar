import tkinter as tk
from tkinter import scrolledtext
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import sqlite3
from datetime import datetime

# =======================
# Load model
# =======================
MODEL_PATH = "phobert-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

class_weights = torch.tensor([2.0, 0.3, 0.3]).to(device)
id2label = {0: "negative", 1: "neutral", 2: "positive"}


# =======================
# Predict function
# =======================
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding="max_length", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)

        weighted_probs = probs * class_weights
        weighted_probs = weighted_probs / weighted_probs.sum(dim=-1, keepdim=True)

        pred_id = torch.argmax(weighted_probs, dim=-1).item()

    return id2label[pred_id], weighted_probs.squeeze().tolist()


# =======================
# SQLite Database
# =======================
DB_PATH = "sentiment.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text_content TEXT,
            sentiment TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(text, sentiment):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO sentiment_history (text_content, sentiment, created_at)
        VALUES (?, ?, ?)
    """, (text, sentiment, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def load_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT text_content, sentiment, created_at
        FROM sentiment_history
        ORDER BY id DESC
    """)
    data = cursor.fetchall()
    conn.close()
    return data


# Init Database
init_db()


# =======================
# Tkinter UI
# =======================
root = tk.Tk()
root.title("Sentiment Analysis - PhoBERT (SQLite)")
root.geometry("700x650")

# Input Label
tk.Label(root, text="Nhập văn bản:", font=("Arial", 12)).pack()

# Text input box
input_box = scrolledtext.ScrolledText(root, height=5, font=("Arial", 12))
input_box.pack(fill=tk.X, padx=10)

# Result Label
result_label = tk.Label(root, text="", font=("Arial", 12), fg="blue")
result_label.pack(pady=10)


# Predict handler
def on_predict():
    text = input_box.get("1.0", tk.END).strip()
    if not text:
        result_label.config(text="Vui lòng nhập nội dung!")
        return

    sentiment, probs = predict_sentiment(text)

    result_label.config(
        text=f"Sentiment: {sentiment}"
    )

    save_to_db(text, sentiment)
    update_history()


# Predict Button
tk.Button(root, text="Phân tích cảm xúc",
          command=on_predict, font=("Arial", 12),
          bg="#4CAF50", fg="white").pack(pady=10)

# History label
tk.Label(root, text="Lịch sử phân tích:", font=("Arial", 12)).pack()

# History display box
history_box = scrolledtext.ScrolledText(root, height=18, font=("Arial", 11))
history_box.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)


def update_history():
    history = load_history()
    history_box.delete("1.0", tk.END)

    for text, sentiment, ts in history:
        history_box.insert(tk.END,
            f"[{ts}]\n"
            f"Text: {text}\n"
            f"Sentiment: {sentiment}\n"
            "---------------------------------\n"
        )

# Load history on start
update_history()

root.mainloop()
