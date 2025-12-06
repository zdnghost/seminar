# ui.py
import tkinter as tk
from tkinter import scrolledtext
from model import predict_sentiment
from database import save_to_db, load_history

def create_ui(root):

    root.title("Sentiment Analysis - PhoBERT (SQLite)")
    root.geometry("700x650")

    tk.Label(root, text="Nhập văn bản:", font=("Arial", 12)).pack()

    input_box = scrolledtext.ScrolledText(root, height=5, font=("Arial", 12))
    input_box.pack(fill=tk.X, padx=10)

    result_label = tk.Label(root, text="", font=("Arial", 12), fg="blue")
    result_label.pack(pady=10)
    def on_predict():
        text = input_box.get("1.0", tk.END).strip()

        if not text:
            result_label.config(text="Vui lòng nhập nội dung!")
            return

        sentiment, _ = predict_sentiment(text)
        result_label.config(text=f"Sentiment: {sentiment}")

        save_to_db(text, sentiment)
        update_history()

    tk.Button(
        root,
        text="Phân tích cảm xúc",
        command=on_predict,
        font=("Arial", 12),
        bg="#4CAF50",
        fg="white"
    ).pack(pady=10)
    
    # --- History box ---
    tk.Label(root, text="Lịch sử phân tích:", font=("Arial", 12)).pack()
    history_box = scrolledtext.ScrolledText(root, height=18, font=("Arial", 11))
    history_box.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)

    def update_history():
        data = load_history()
        history_box.delete("1.0", tk.END)

        for text, sentiment, ts in data:
            history_box.insert(tk.END,
                f"[{ts}]\nText: {text}\nSentiment: {sentiment}\n"
                "---------------------------------\n"
            )



    update_history()
