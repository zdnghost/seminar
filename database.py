# database.py
import sqlite3
from datetime import datetime

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
    rows = cursor.fetchall()
    conn.close()
    return rows
