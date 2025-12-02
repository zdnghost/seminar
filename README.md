# Sentiment Analysis App (PhoBERT + Tkinter + SQLite)
## 1. Requirements
- Python 3.9+
- `pip` (or `uv`)
- Virtual environment (recommended)
- CUDA GPU (optional but recommended for training)
## 2. Environment Setup
Create and activate a virtual environment:
```
    python -m venv venv
```
Windows :
```
    venv\Scripts\activate
```
Linux/macOS :
```
    source venv/bin/activate
```
Install dependencies :
  
```
    pip install -r requirements.txt
```
## Run the Sentiment Analysis App
```
    python main.py
```
## Fine-tuning the PhoBERT Model

This project uses the Kaggle dataset : [Vietnamese Sentiment Analyst ](https://www.kaggle.com/datasets/linhlpv/vietnamese-sentiment-analyst/code)

To run Fine-tuning :
```
    python fine-tuning.py
```
