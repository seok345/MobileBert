#  MobileBERT를 활용한 FIFA 게임 리뷰 긍부정 분석 프로젝트


<p align="center">
  <img src="https://img.shields.io/badge/pycharm-%23000000.svg?&style=for-the-badge&logo=pycharm&logoColor=white" />
  <img src="https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/pytorch-%23EE4C2C.svg?&style=for-the-badge&logo=pytorch&logoColor=white" />
</p>

---

## 1.  개요

피파(명칭: FIFA) 시리즈는 전 세계 축구 팬들과 게이머들에게 가장 사랑받는 스포츠 게임입니다. 온라인 커뮤니티와 리뷰 플랫폼에서도 많은 유저들이 피파 게임 내 선호 선수, 팀 전략, 업데이트 내용 등에 관한 의견과 평가를 남기것으로 
사용자 리뷰는 업데이트 후 팬들의 감정을 이해하는 데 중요한 데이터가 될 수 것 같아
프로젝트는 피파 관련 사용자 리뷰 데이터를 수집하여, 리뷰 텍스트의 감성(긍정과 부정)을 자동으로 구분하는 인공지능 모델을 개발하는 것을 목표로 했고
이를 통해 사용자들이 어떤 점에 만족하고, 어떤 부분에 불만이 많은지 파악하는 동시에, 새로운 업데이트나 등에 활용할 수 있는 인사이트를 도출하고자 했고
이 프로젝트는 **MobileBERT**를 활용하여 유저 리뷰를 긍정,부정으로 자동 분류하는것을 목표로 생각하고 해봤다

## 2.  데이터

### 데이터 분석 개요

 데이터 구성 및 특징
데이터 원천: kaggle에 있는 FIFA 데이터를 사용했으며
데이터 포맷: 사용자의 평가, 리뷰 데이터, 날짜,평점 등으로 이루어진 구조로
데이터 개수: 약 150,000건 이상 추출 가능하며, 여기서 정제와 전처리를 거침고
리뷰 텍스트: 짧은 코멘트부터 상세한 설명까지 다양하며 긍정적 또는 부정적 감정을 내포했다

 데이터 특징 및 분석 포인트
감성 분석: 리뷰 텍스트를 긍정과 부정으로 분류하는 것뿐만 아니라, 업데이트에 대한 감성 트렌드도 파악하기 위해 월마다 분류했으며
리뷰 길이: 20글자 이상으로 직설적인 후기와 오락가락하는 긴 감상평이 공존하며
평점 분포: 1점(최악)부터 5점(최고)까지 구성, 대부분은 4~5점이 많이 있었다

###  수집 및 전처리
 엑셀 원본 파일 요약:
 | index | userName        | content                                  | score | at                  |
|-------|------------------|------------------------------------------|-------|---------------------|
| 0     | Aastha Ranjitkar | Wow great game                           | 5     | 2022-03-23 13:40:00 |
| 1     | Arun Sharma      | Nice game football                       | 4     | 2022-03-23 13:40:00 |
| 2     | Arya Nandha      | Good                                     | 5     | 2022-03-23 13:37:00 |
| 3     | Ramesh Murali    | In hard mode my player...                | 1     | 2022-03-23 13:36:00 |
다
  

전처리 파일 요약: 
| unnamed: 0 | username           | userimage    | content                 | score | thumbsupcount  | reviewcreatedversion   | at                  | replycontent  | as |
|------------|--------------------|------------- |-------------------------|-------|----------------|------------------------|---------------------|---------------|----|
| 3          | Ramesh Murali      | https://...  | In hard mode my play... | 1     | 0              | 15.5.04                | 2022-03-23 13:36:00 | NaN           | 0  |
| 9          | Syed Ammar         | https://...  | Very nice and has a ... | 4     | 0              | NaN                    | 2022-03-23 13:29:00 | NaN           | 1  |
| 12         | Komikedy           | https://...  | Things that need to ... | 5     | 0              | 15.5.04                | 2022-03-23 13:21:00 | NaN           | 1  |
| 13         | onuh rose          | https://...  | Remains the best soc... | 5     | 0              | NaN                    | 2022-03-23 13:20:00 | NaN           | 1  |
| 18         | ameerul tafhim     | https://...  | this game so good bu... | 4     | 0              | NaN                    | 2022-03-23 13:14:00 | NaN           | 1  |
| 20         | Aqish Mieqheal     | https://...  | Please gift me Neyma... | 5     | 0              | 다

감성분석 모델 성능
모델: 사전학습된 MobileBERT 또는 간단한 BERT 계열
학습 결과:적절한 데이터 전처리와 함께, 감성 분류 정확도가 87%가 나왔으며
초반 학습은 높은 손실값으로 시작했으나, 반복 학습을 통해 높은 정확도가 나왔다

 실제 적용과 평가
유저들의 리뷰 데이터에 모델을 적용하였을 때, 긍정/부정 예측 정확도는 약 87% 이상으로 나타났으며
유저들이 남긴 리뷰 텍스트로 긍정과 부정이 많은 순으로 분류를 했다

## 결과 화면:
### 학습 후 결과 이미지: ![이미지](https://raw.githubusercontent.com/seok345/MobileBert/main/png/5.png)
## 전처리를 하고 데이터를 학습시키며 정확도를 높이기 위해 모델 저장을 하며 계속 학습을 시키고 
### 학습후 정확도 결과 이미지: ![이미지](https://raw.githubusercontent.com/seok345/MobileBert/main/png/4.png)
##  계속 학습을 시킨 저장된 모델을  정확도를 보기 위해 실행을 시켜보니 87%까지 도달했으며
### 긍정이 많은순: ![이미지](https://raw.githubusercontent.com/seok345/MobileBert/main/png/1.png)
## 이렇게 학습을 시키고 정확도를 확인하고 그래프로 긍정이 많은 순으로 보면 22년 1월이 긍정리뷰가 제일 많이 있는것으로 알수있었고
### 부정이 많은순: ![이미지](https://raw.githubusercontent.com/seok345/MobileBert/main/png/2.png)
## 부정이 많은 순으로 그래프를 보면 긍정과 같이 22년 1월이 부정리뷰가 제일 많이 있는것으로 알수있었으며
### 긍부정 많은 순으로 비교: ![이미지](https://raw.githubusercontent.com/seok345/MobileBert/main/png/3.png)
## 긍정, 부정리뷰를 비교해보면 22년 1월리뷰가 압도적으로 리뷰부터 많으며 긍정리뷰가 많아도 부정 리뷰가 많은 것을 볼수있었다

### 느낀 점과 배운 점
이번 프로젝트를 통해 데이터 전처리와 정제의 중요성을 다시 한번 깨달았으며 
특히, 피파처럼 인기가 많은 게임은 다양한 표현 방식을 사용하는 리뷰에서는 일관성과 품질 유지가 감성 예측의 핵심임을 알게 되었고
처음에는 데이터 수도 많아서 힘들었으며 한글깨짐이 많고 오류가 많았으나,글자수 20자 미만 제거, 중립(3점) 제거 등을 통해 점차 성능을 높일 수 있었으며
그리고 이렇게 프로젝트를 하면서 이렇게 많은 리뷰가 있다는것을 알면서 게임은 긍정이 많아도 부정도 많은 것을 알수있었다다

