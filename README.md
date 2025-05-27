# 🎮 MobileBERT를 활용한 FIFA 게임 리뷰 긍부정 분석 프로젝트

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/ko/thumb/d/dd/FIFA_Online_4_logo.svg/640px-FIFA_Online_4_logo.svg.png" width="300"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/pycharm-%23000000.svg?&style=for-the-badge&logo=pycharm&logoColor=white" />
  <img src="https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/pytorch-%23EE4C2C.svg?&style=for-the-badge&logo=pytorch&logoColor=white" />
</p>

---

## 1. 🧭 개요

FIFA 시리즈는 축구 팬들 사이에서 가장 인기 있는 스포츠 게임 중 하나로, 수많은 유저들이 다양한 리뷰를 남기고 있습니다.  
이러한 리뷰는 게임에 대한 객관적인 평가를 도출할 수 있는 소중한 자료입니다.  

본 프로젝트는 **MobileBERT**를 활용하여 유저 리뷰를 **긍정/부정**으로 자동 분류하고, 전체적인 감성 흐름을 분석하는 것을 목표로 합니다.  
모델 경량화와 빠른 추론 속도를 고려하여 MobileBERT를 선택했습니다.

---

## 2. 📊 데이터

### ✅ 수집 및 전처리

- **소스**: 자체 크롤링한 FIFA 리뷰 데이터셋 (`fifa3.csv`)
- **전처리 방식**:  
  - 3점(중립) 리뷰 제거  
  - 리뷰 길이 20자 미만 제거  
  - 점수 기반 라벨링:  
    - 1~2점 → 부정(0)  
    - 4~5점 → 긍정(1)

```python
df = df[(df["score"] != 3) & (df["content"].str.len() >= 20)]
df["as"] = df["score"].apply(lambda x: 0 if x <= 2 else 1)
