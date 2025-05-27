#  MobileBERT를 활용한 FIFA 게임 리뷰 긍부정 분석 프로젝트


<p align="center">
  <img src="https://img.shields.io/badge/pycharm-%23000000.svg?&style=for-the-badge&logo=pycharm&logoColor=white" />
  <img src="https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/pytorch-%23EE4C2C.svg?&style=for-the-badge&logo=pytorch&logoColor=white" />
</p>

---

## 1. 🧭 개요

FIFA 시리즈는 축구 팬들 사이에서 인기 있는 스포츠 게임 중 하나로 수많은 유저들이 다양하고 많은 리뷰가 있어서 이 데이터셋을 활용했습니다 
다양하고 많은 리뷰로 게임에 대한 객관적인 평가를 나오게 할수 있는 자료입니다  

이 프로젝트는 **MobileBERT**를 활용하여 유저 리뷰를 긍정,부정으로 자동 분류하는것을 목표로 했습니다

---

## 2. 📊 데이터

### ✅ 수집 및 전처리

- **소스**: 데이터셋을 들고와 2021~2022년 FIFA 리뷰 데이터셋 (`fifa_cleaned.csv`)
- **전처리 방식**:  
  - 3점(중립) 리뷰 제거  
  - 리뷰 길이 20자 미만 제거  
  - 점수 기반 라벨링:  
    - 1~2점 → 부정(0)  
    - 4~5점 → 긍정(1)
