# model.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "phobert-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
# Tùy chọn GPU nếu có
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()# Chuyển sang chế độ inference

# Trọng số lớp giúp tăng độ chính xác cho các lớp mất cân bằng
class_weights = torch.tensor([2.0, 0.5, 0.3]).to(device)
# Mapping từ ID → nhãn cảm xúc
id2label = {0: "negative", 1: "neutral", 2: "positive"}


def predict_sentiment(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        # Softmax chuẩn hóa về xác suất
        probs = F.softmax(logits, dim=-1)
        # Áp trọng số để xử lý imbalance dataset
        weighted_probs = probs * class_weights
        weighted_probs = weighted_probs / weighted_probs.sum(dim=-1, keepdim=True)
        # Lấy nhãn có xác suất cao nhất
        pred_id = torch.argmax(weighted_probs, dim=-1).item()

    return id2label[pred_id], weighted_probs.squeeze().tolist()
