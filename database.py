import sqlite3
from datetime import datetime

# Đường dẫn file database SQLite
DB_PATH = "sentiment.db"


def init_db():
    """
    Khởi tạo database:
    - Nếu chưa tồn tại bảng `sentiment_history` thì tạo mới.
    - Bảng chứa nội dung, kết quả phân tích và timestamp.
    """
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
    """
    Lưu một bản ghi phân tích cảm xúc vào database.
    
    Params:
        text (str): nội dung người dùng nhập
        sentiment (str): kết quả phân tích (positive / neutral / negative)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO sentiment_history (text_content, sentiment, created_at)
        VALUES (?, ?, ?)
    """, (
        text,
        sentiment,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # thời gian hiện tại
    ))

    conn.commit()
    conn.close()


def load_history():
    """
    Tải tất cả lịch sử phân tích, sắp xếp theo id giảm dần (mới nhất → cũ nhất).
    
    Returns:
        list[tuple]: danh sách các bản ghi dạng (text_content, sentiment, created_at)
    """
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