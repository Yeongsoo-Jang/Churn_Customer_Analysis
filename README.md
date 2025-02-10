# Churn_Customer_Analysis
 e-commerce fashion flatform CRM

### Table of Contents in this notebook 

* [Analysis. 이탈 고객을 잡으면 기업에 얼마나 이익을 가져올까]
    * [데이터 전처리]
    * [이탈 고객 정의]
    * [이탈 고객이 매출에 미치는 영향]
    * [이탈 관련 핵심 feature 선정]
    * [이탈 고객 예측]
    * [고객 군집]
    * [장바구니 분석]

# 데이터 분석 포트폴리오

## 프로젝트: 인도네시아 이커머스 플랫폼 고객 이탈 예측 및 관리

### 1. 문제 정의
- **문제점**: 활성 사용자 이탈 대응 방안 부재
- **배경**:
  - 최근 6개월 동안 월별 잔존율이 최대 47%, 최소 33%로 감소.
  - 라마단 이후 매출 하락과 고객 이탈 증가가 확인됨.
- **목표**:
  - 고객별 이탈 확률을 예측하고, 이를 기반으로 고객 유지 전략을 수립하여 매출 상승에 기여.

---

### 2. 가설 설정
- **주요 가설**: 이탈 고객과 활성 고객의 첫 거래 및 최근 방문 행동 데이터는 다를 것이다.
- **분석 데이터**:
  - 고객 정보, 클릭 스트림 데이터, 거래 내역, 제품 정보.

---

### 3. 실험 설계
#### 데이터 처리 및 분석
- **데이터 전처리**:
  - 결측값 처리 및 이상치 제거.
  - 다중공선성 높은 변수 제거(PCA 병합).
- **가설 검정**:
  - 통계적 검정을 통해 이탈/활성 고객의 매출 평균 차이 확인.
- **행동 분석**:
  - 클릭 수, 스크롤 수, 검색 횟수 등 행동 변수와 이탈 여부 간 상관관계 분석.

#### 머신러닝 모델링
- **모델 선정**: CatBoost Classifier
- **모델 평가**:
  - 주요 지표: 정확도(0.88), ROC AUC(0.97), F1 Score(0.90).
- **변수 중요도 분석**:
  - SHAP 값을 활용하여 주요 변수 도출 (예: Add_to_Cart, Click).

---

### 4. 결과 도출
#### 주요 인사이트
1. **이탈 고객 특징**:
   - 가입 기간이 길수록 이탈 확률 증가.
   - 클릭, 스크롤 등 행동 데이터가 적을수록 이탈 가능성 높음.
2. **매출 영향 요인**:
   - 클릭 및 스크롤 증가 시 매출 상승 경향 확인.

#### 대시보드 설계
- 고객별 주요 지표(CLTV, AOV)와 이탈 확률을 실시간으로 관리할 수 있는 대시보드 생성.
- 월별 추천 상품을 제안하여 마케팅 전략 강화.

#### 비즈니스 임팩트
- 활성 고객 증가 시 매출 약 9,460만 루피아(한화 약 840만 원) 상승 예상.
- 초개인화 마케팅으로 고객 만족도 및 재구매율 향상.

---

### 사용 기술 및 도구
- **프로그래밍 언어**: Python (Pandas, Numpy, Scikit-learn 등)
- **데이터베이스**: MySQL
- **시각화 도구**: Tableau

---

### 결론
본 프로젝트를 통해 데이터 기반 의사결정의 중요성을 확인하였으며, 머신러닝 모델과 대시보드를 활용해 비즈니스 문제를 효과적으로 해결할 수 있음을 입증했습니다.


[대시보드 Tableau Public Link]
https://public.tableau.com/views/ChurncustomerMarketBasketAnalysis/sheet4?:language=ko-KR&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link
