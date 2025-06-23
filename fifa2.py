import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import platform

# ===== [1] 한글 폰트 설정 =====
if platform.system() == 'Windows':
    font_name = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    font_name = 'AppleGothic'
else:
    font_name = 'NanumGothic'

plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

# ===== [2] 감성 분석 모델 불러오기 =====
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ===== [3] 데이터 불러오기 및 정제 =====
df = pd.read_csv("fifa_cleaned.csv")
df = df.dropna(subset=["content", "at"])
df = df.sample(500, random_state=42)

# ===== [4] 날짜 처리 =====
df['date'] = pd.to_datetime(df['at'], errors='coerce')
df = df.dropna(subset=['date'])
df['year_month'] = df['date'].dt.to_period('M').astype(str)

# ===== [5] 배치 감성 분석 함수 =====
def predict_sentiment_batch(texts, batch_size=32):
    sentiments = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        preds = probs.argmax(dim=1).tolist()
        sentiments.extend(preds)
    return sentiments

# ===== [6] 감성 분석 실행 =====
df['sentiment'] = predict_sentiment_batch(df['content'].tolist())
df['is_negative'] = df['sentiment'] == 0

# ===== [7] 부정 리뷰 개수 집계 & 내림차순 정렬 =====
negative_df = df[df['is_negative']]
negative_counts = negative_df['year_month'].value_counts().sort_values(ascending=False)

# ===== [8] 시각화 (내림차순 순서로 바 차트) =====
plt.figure(figsize=(12, 6))
negative_counts.plot(kind='bar', color='tomato')
plt.title("부정 리뷰 수 (연/월별, 높은 순)")
plt.xlabel("연-월")
plt.ylabel("부정 리뷰 수")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
