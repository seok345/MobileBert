#  MobileBERT를 활용한 FIFA 게임 리뷰 긍부정 분석 프로젝트


<p align="center">
  <img src="https://img.shields.io/badge/pycharm-%23000000.svg?&style=for-the-badge&logo=pycharm&logoColor=white" />
  <img src="https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/pytorch-%23EE4C2C.svg?&style=for-the-badge&logo=pytorch&logoColor=white" />
</p>

---

## 1. 🧭 개요

1-1. 문제 정의
피파(명칭: FIFA) 시리즈는 전 세계 축구 팬들과 게이머들에게 가장 사랑받는 스포츠 게임입니다. 온라인 커뮤니티와 리뷰 플랫폼에서도 많은 유저들이 피파 게임 내 선호 선수, 팀 전략, 업데이트 내용 등에 관한 의견과 평가를 남기곤 합니다.
이러한 사용자 리뷰는 새로운 선수 영입, 게임 플레이 전략, 혹은 업데이트 후 팬들의 감정을 이해하는 데 중요한 데이터가 될 수 있습니다.
이번 프로젝트는 피파 관련 사용자 리뷰 데이터를 수집하여, 리뷰 텍스트의 감성(긍정과 부정)을 자동으로 구분하는 인공지능 모델을 개발하는 것을 목표로 합니다.
이를 통해 사용자들이 어떤 점에 만족하고, 어떤 부분에 불만이 많은지 파악하는 동시에, 새로운 업데이트나 선수 선호도 분석 등에 활용할 수 있는 인사이트를 도출하고자 합니다.
이 프로젝트는 **MobileBERT**를 활용하여 유저 리뷰를 긍정,부정으로 자동 분류하는것을 목표로 했습니다

## 2. 📊 데이터

### 데이터 분석 개요

2-1. 데이터 구성 및 특징
데이터 원천: 피파 커뮤니티, SNS, 리뷰 플랫폼 등에서 수집된 사용자 후기 텍스트
데이터 포맷: 선수 이름, 사용자의 평가, 리뷰 텍스트, 날짜, 추천 수, 평점 등으로 이루어진 구조
데이터 개수: 약 50,000건 이상 추출 가능하며, 여기서 정제와 전처리를 거침
리뷰 텍스트: 짧은 코멘트부터 상세한 설명까지 다양, 주로 긍정적 또는 부정적 감정을 내포
2-2. 데이터 특징 및 분석 포인트
감성 분석: 리뷰 텍스트를 긍정과 부정으로 분류하는 것뿐만 아니라, 특정 선수 또는 업데이트에 대한 감성 트렌드도 파악
리뷰 길이: 20글자 이상으로 직설적인 후기와 오락가락하는 긴 감상평이 공존
평점 분포: 1점(최악)부터 5점(최고)까지 구성, 대부분은 4~5점이 우세
추천 수 관계: 추천 수가 높은 리뷰는 대체로 긍정적이며, 평균 평점과도 강한 상관관계 존재

### ✅ 수집 및 전처리

 데이터 전처리 전략
스팸 또는 불필요한 텍스트 제거: 특수 문자, URL, 이모티콘 등 정제
감성 라벨링: 평점 기준 (평점 3 이상: 긍정, 4~5: 중립/무시, 3 이하: 부정) 또는 텍스트 내 감성 키워드 기반 레이블링
리뷰 길이 기준 필터링: 너무 짧거나 너무 긴 리뷰는 제외하여 일관성 확보
선수/이벤트 필터링: 특정 선수 이름 또는 핵심 키워드를 기준으로 하위 집단 분석 진행
이러한 전처리 과정은 모델의 감성 예측에 있어서 중요한 역할을 하며, 데이터의 품질 향상에 기여합니다.

- **소스**: 데이터셋을 들고와 2021~2022년 FIFA 리뷰 데이터셋 (`fifa_cleaned.csv`)
- **전처리 방식**:  
  - 3점(중립) 리뷰 제거  
  - 리뷰 길이 20자 미만 제거  
  - 점수 기반 라벨링:  
    - 1~2점 → 부정(0)  
    - 4~5점 → 긍정(1)
   
 엑셀 원본 파일:
 | index | userName        | content                                  | score | at                  |
|-------|------------------|------------------------------------------|-------|---------------------|
| 0     | Aastha Ranjitkar | Wow great game                           | 5     | 2022-03-23 13:40:00 |
| 1     | Arun Sharma      | Nice game football                       | 4     | 2022-03-23 13:40:00 |
| 2     | Arya Nandha      | Good                                     | 5     | 2022-03-23 13:37:00 |
| 3     | Ramesh Murali    | In hard mode my player...                | 1     | 2022-03-23 13:36:00 |
| 4     | Nandita Biswas   | Bad game had dawon                       | 1     | 2022-03-23 13:36:00 |
| 5     | Ishrak Araf      | ?뵦                                      | 5     | 2022-03-23 13:35:00 |
| 6     | winn kyaw soe    | I like this game.                        | 5     | 2022-03-23 13:33:00 |
| 7     | pheromotone b.   | good                                     | 5     | 2022-03-23 13:32:00 |
| 8     | Ali FanBoy       | Best game ever                           | 5     | 2022-03-23 13:30:00 |
| 9     | Syed Ammar       | Very nice and has a lot of good character| 4     | 2022-03-23 13:29:00 |

전처리 파일: 
| unnamed: 0 | username           | userimage    | content                 | score | thumbsupcount  | reviewcreatedversion   | at                  | replycontent  | as |
|------------|--------------------|------------- |-------------------------|-------|----------------|------------------------|---------------------|---------------|----|
| 3          | Ramesh Murali      | https://...  | In hard mode my play... | 1     | 0              | 15.5.04                | 2022-03-23 13:36:00 | NaN           | 0  |
| 9          | Syed Ammar         | https://...  | Very nice and has a ... | 4     | 0              | NaN                    | 2022-03-23 13:29:00 | NaN           | 1  |
| 12         | Komikedy           | https://...  | Things that need to ... | 5     | 0              | 15.5.04                | 2022-03-23 13:21:00 | NaN           | 1  |
| 13         | onuh rose          | https://...  | Remains the best soc... | 5     | 0              | NaN                    | 2022-03-23 13:20:00 | NaN           | 1  |
| 18         | ameerul tafhim     | https://...  | this game so good bu... | 4     | 0              | NaN                    | 2022-03-23 13:14:00 | NaN           | 1  |
| 20         | Aqish Mieqheal     | https://...  | Please gift me Neyma... | 5     | 0              | NaN                    | 2022-03-23 13:06:00 | NaN           | 1  |
| 21         | Achraf Yahiaoui    | https://...  | The game play need t... | 2     | 0              | 15.5.04                | 2022-03-23 12:59:00 | NaN           | 0  |
| 23         | Safiqul alam Boksh | https://...  | The game is dogsh*t.... | 1     | 0              | 15.5.04                | 2022-03-23 12:56:00 | NaN           | 0  |
| 24         | WOLF GAMER         | https://...  | Forget the last time... | 1     | 4              | 15.5.04                | 2022-03-23 12:56:00 | NaN           | 0  |
| 25         | saif hashmi        | https://...  | Why am I getting tea... | 1     | 0              | NaN                    | 2022-03-23 12:55:00 | NaN           | 0  |

     
  3-1. 사용 환경 및 패키지
개발 환경: Python 3.9 이상, Jupyter Notebook 또는 PyCharm 환경
주요 패키지: pandas, transformers, PyTorch 또는 TensorFlow, scikit-learn, matplotlib 등을 활용
3-2. 감성분석 모델 성능
모델: 사전학습된 MobileBERT 또는 간단한 BERT 계열
학습 결과:적절한 데이터 전처리와 함께, 감성 분류 정확도가 최대 85~90%에 달함
초반 학습은 높은 손실값으로 시작했으나, 반복 학습을 통해 점진적 수렴과 높은 정확도 도달

3-3. 실제 적용과 평가
특정 선수 리뷰 데이터에 모델을 적용하였을 때, 긍정/부정 예측 정확도는 약 85% 이상으로 나타남
유저들이 남긴 리뷰 텍스트의 감성 트렌드와 일치하는 분석 결과 도출 가능



### 느낀 점과 배운 점
이번 프로젝트를 통해 데이터 전처리와 정제의 중요성을 다시 한번 깨달았습니다.
특히, 피파처럼 많은 선수와 다양한 표현 방식을 사용하는 스포츠 게임 리뷰에서는 일관성과 품질 유지가 감성 예측의 핵심임을 알게 됐습니다.
초기엔 텍스트 내 의미 파악이 어려워 오류가 많았으나, 키워드 기반 감성 라벨링, 불필요한 노이즈 제거 등을 통해 점차 성능을 높일 수 있었습니다.

또한, 모델이 다양한 사용자 의견의 뉘앙스를 얼마나 잘 파악하는지 테스트하면서, 감성 분석의 한계와 함께 더 나은 피처 엔지니어링 방법에 대한 아이디어도 발전시킬 수 있었습니다.

[출처](https://www.kaggle.com/datasets/mohitksharma/fifa-vs-pes-review-war?select=FIFA.csv)
