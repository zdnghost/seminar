from underthesea import word_tokenize
from transformers import pipeline
import sqlite3

def preprocess(text):
    text = " ".join(word_tokenize(text))
    text = text.lower()
    text = text[:50]
    return text

sentiment_pipeline = pipeline("sentiment-analysis", model="vinai/phobert-base-v2")

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label']
    score = result['score']
    if score < 0.5:
        label = "NEUTRAL"
    return label, score

def validate(text):
    if len(text) < 5:
        return False
    return True

def save_result(text, sentiment, db_path="sentiment.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sentiments
                 (text TEXT, sentiment TEXT)''')
    c.execute("INSERT INTO sentiments VALUES (?, ?)", (text, sentiment))
    conn.commit()
    conn.close()

def pipeline_full(text):
    text_pre = preprocess(text)
    if not validate(text_pre):
        print("Câu không hợp lệ, thử lại")
        return
    sentiment, score = analyze_sentiment(text_pre)
    save_result(text_pre, sentiment)
    print(f"Câu: {text_pre} | Sentiment: {sentiment} | Score: {score}")

pipeline_full("Rất vui hôm nay")
