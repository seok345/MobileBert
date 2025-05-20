import pandas as pd

# 데이터 불러오기
df = pd.read_csv("fifa_cleaned.csv", encoding="utf-8-sig")  # 또는 "cp949"

# 컬럼 이름 소문자 처리 및 공백 제거
df.columns = df.columns.str.strip().str.lower()

# 필요한 컬럼 존재 확인
if 'content' not in df.columns or 'as' not in df.columns:
    raise ValueError("데이터에 'content' 또는 'as' 컬럼이 없습니다.")

# 결측치 제거 및 라벨 형식 정리
df = df.dropna(subset=['content', 'as'])
df['as'] = pd.to_numeric(df['as'], errors='coerce')
df = df.dropna(subset=['as'])
df = df[df['as'].isin([0, 1])]

# 1만 개 랜덤 추출 (random_state는 고정값 사용 시 재현 가능)
df_sampled = df.sample(n=10000, random_state=42).reset_index(drop=True)

# 확인
print(df_sampled.head())

# 저장 (선택)
df_sampled.to_csv("fifa_10000_sampled.csv", index=False, encoding="utf-8-sig")
print("1만 개 샘플 저장 완료: fifa_10000_sampled.csv")
