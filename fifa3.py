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
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # 빠른 영어 감성 모델
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ===== [3] 데이터 불러오기 및 정제 =====
df = pd.read_csv("fifa_cleaned.csv")
df = df.dropna(subset=["content", "at"])
df = df.sample(500, random_state=42)  # 샘플링

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
        preds = probs.argmax(dim=1).tolist()  # 1 = 긍정, 0 = 부정
        sentiments.extend(preds)
    return sentiments

# ===== [6] 감성 분석 실행 =====
df['sentiment'] = predict_sentiment_batch(df['content'].tolist())
df['sentiment_label'] = df['sentiment'].apply(lambda x: '긍정' if x == 1 else '부정')

# ===== [7] 피벗 테이블로 긍정/부정 리뷰 수 집계 =====
sentiment_counts = df.pivot_table(index='year_month', columns='sentiment_label', aggfunc='size', fill_value=0)

# ===== [8] 월별 정렬 (오름차순) =====
sentiment_counts = sentiment_counts.sort_index()

# ===== [9] 시각화 =====
plt.figure(figsize=(12, 6))
sentiment_counts.plot(kind='bar', stacked=False, color=['tomato', 'skyblue'])
plt.title("긍정/부정 리뷰 수 (연/월별)")
plt.xlabel("연-월")
plt.ylabel("리뷰 수")
plt.xticks(rotation=45)
plt.legend(title="감정")
plt.tight_layout()
plt.show()
