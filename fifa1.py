import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import platform

# ===== [1] 한글 폰트 설정 =====
if platform.system() == 'Windows':
    font_name = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # macOS
    font_name = 'AppleGothic'
else:  # Linux, Colab 등
    font_name = 'NanumGothic'

plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

# ===== [2] 감성 분석 모델 불러오기 =====
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # 빠르고 가벼운 영어 감성 분석 모델
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ===== [3] 데이터 불러오기 =====
df = pd.read_csv("fifa_cleaned.csv")
df = df.dropna(subset=["content", "at"])  # 결측치 제거
df = df.sample(500, random_state=42)  # 샘플링 (속도 개선)

# ===== [4] 날짜 처리 =====
df['date'] = pd.to_datetime(df['at'], errors='coerce')
df = df.dropna(subset=['date'])  # 날짜 변환 실패 제거
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

# ===== [7] 긍정 리뷰만 선택 =====
df['is_positive'] = df['sentiment'] == 1
positive_df = df[df['is_positive']]

# ===== [8] 연/월별 긍정 리뷰 수 계산 =====
positive_counts = positive_df['year_month'].value_counts().sort_values(ascending=False)

# ===== [9] 시각화 =====
plt.figure(figsize=(12, 6))
positive_counts.plot(kind='bar', color='skyblue')
plt.title("긍정 리뷰 수 (연/월별, 내림차순)")
plt.xlabel("연-월")
plt.ylabel("긍정 리뷰 수")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
