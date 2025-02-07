import os
import pandas as pd
import numpy as np
import ast
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

# rfm_df 함수 생성
# 지난 1년간 기록을 바탕으로 점수 산정

# rfm dataframe 생성
def rfm(df):

    present_day = df['transaction_date'].max() + dt.timedelta(days = 2)  # Timestamp('2022-08-02 00:00:00')

    rfm = df.groupby('customer_id').agg({'transaction_date': lambda x: (present_day - x.max()).days,
                                        'session_id': lambda x: x.nunique(),
                                        'amount': lambda x: x.sum(),
                                        'first_join_date': lambda x: (present_day - x.max()).days})

    rfm.columns = ['recency', 'frequency', 'monetary', 'join_period']
    rfm = rfm.reset_index()

    return rfm

# rfm score 
def get_rfm_scores(df) -> pd.core.frame.DataFrame:

    # recency 구분
    recency_quantiles = [0.0, 0.25, 0.45, 0.60, 0.75, 1.0]
    recency_bins = df['recency'].quantile(recency_quantiles).tolist()
    df["recency_score"] = pd.cut(df["recency"], bins = recency_bins, labels=[5, 4, 3, 2, 1], include_lowest = True)
    
    # frequency 구분
    df["frequency_score"] = pd.qcut(
        df["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    )
    
    # monetary 구분
    df["monetary_score"] = pd.qcut(df["monetary"], 5, labels=[1, 2, 3, 4, 5])
    
    # 가입 기간 구분
    df["period_score"] = pd.cut(df["join_period"], 
                                bins=[0, 90, 180, 270, 360, float('inf')],
                                labels=[5, 4, 3, 2, 1],
                                right = True)
    
    df['RF_SCORE'] = df["recency_score"].astype(str) + df["frequency_score"].astype(str)
    df["RFM_SCORE"] = df["recency_score"].astype(str) + df["frequency_score"].astype(str) + \
        df["monetary_score"].astype(str)

    return df

# rfm segmentation
def get_segment(df):
    seg_map = {
    r'[1-2][1-2]': 'hibernating', # Hibernating: Recency와 Frequency가 모두 낮은 고객 (오랫동안 거래가 없고 활동이 적음)
    r'[1-2][3-4]': 'at_Risk', # At Risk: Recency가 낮고 Frequency가 중간 수준인 고객 (과거에는 활발했으나 최근 거래가 줄어든 고객)
    r'[1-2]5': 'cant_loose', # Cannot Lose Them: Recency가 낮고 Frequency가 높아 중요한 고객 (이탈 방지가 필요한 고객)
    r'3[1-2]': 'about_to_sleep', # About To Sleep: Recency는 중간 수준이지만 Frequency가 낮은 고객 (거래가 줄어들 가능성이 있는 고객)
    r'33': 'need_attention', # Need Attention: Recency와 Frequency가 모두 중간 수준인 고객 (추가적인 관심이 필요한 고객)
    r'[3-4][4-5]': 'loyal_customers', # Loyal Customers: Recency와 Frequency가 모두 높은 고객 (충성도가 높은 고객)
    r'41|51': 'promising', # Promising: Recency가 높지만 Frequency는 낮은 고객 (잠재적으로 성장 가능성이 있는 신규 또는 초기 고객)
    r'[4-5][2-3]': 'potential_loyalists', # Potential Loyalists: Recency와 Frequency가 중간에서 높은 수준인 고객 (충성도가 높아질 가능성이 있는 고객)
    r'5[4-5]': 'champions' # Champions: Recency와 Frequency가 모두 최고 수준인 VIP 고객 (가장 가치 있는 고객)
}
      
    df['segment'] = df['RF_SCORE'].replace(seg_map, regex = True)
    df.loc[df['period_score'] == 5, 'segment'] = 'new_customers' # New Customers: 가입 기간 점수가 5점인 고객
    
    return df

rfm_df = rfm(merged_df)
rfm_df = get_rfm_scores(rfm_df)
rfm_df = get_segment(rfm_df)

# customer_id 기준 파생 변수 생성

def get_customer_info(merged_df, customer_df, click_stream_df):

    # 평균 구매 주기 계산
    group = merged_df.groupby('customer_id')

    def purchase_cycle(group):
        if len(group) > 1:
            return group['transaction_date'].diff().mean().days
        else:
            return None
        
    df_avg_cycle = group.apply(lambda group: purchase_cycle(group)).reset_index(name = 'avg_pur_cycle').fillna(0)

    # 고객별 AOV 계산
    df_aov = group['amount'].mean().reset_index(name = 'aov').round(2)

    # 고객별 전환율(total -> booking)
    data = merged_df[['session_id', 'customer_id']]
    df = pd.merge(click_stream_df, data, on='session_id')
    total_session = df.groupby('customer_id')['session_id'].count()
    booking_session = df[df['event_name'] == 'BOOKING'].groupby('customer_id')['session_id'].count()
    conversion_rate = (booking_session / total_session * 100).fillna(0).round(2).reset_index(name = 'conversion_rate')

    # rfm_df
    rfm_df2 = rfm_df[['customer_id', 'recency', 'frequency', 'join_period']]

    # session별 행동 분석
    data = merged_df[['session_id', 'customer_id', 'amount']]
    session_pivot = pd.pivot_table(data = click_stream_df, index='session_id', columns='event_name', values='datetime', aggfunc='count').reset_index().fillna(0)
    session_pivot = session_pivot.merge(data, how='inner', on='session_id')
    session_df = session_pivot.groupby('customer_id').agg\
        (
        CLICK= ('CLICK', 'sum'),
        SCROLL= ('SCROLL', 'sum'),
        SEARCH= ('SEARCH', 'sum'),
        ADD_TO_CART = ('ADD_TO_CART', 'sum'),
        ADD_PROMO = ('ADD_PROMO', 'sum'),
        ITEM_DETAIL = ('ITEM_DETAIL', 'sum'),
        BOOKING = ('BOOKING', 'sum'),
        amount= ('amount', 'sum')
        ).reset_index()
        
    # 고객 특성(성별, 나이, 지역)
    current_date = datetime.now()
    customer = customer_df[['customer_id', 'gender_encoding', 'birthdate', 'home_location_lat', 'home_location_long']].copy()
    customer.loc[:, 'age'] = pd.to_datetime(customer['birthdate']).apply(
        lambda x: current_date.year - x.year - ((current_date.month, current_date.day) < (x.month, x.day))
    )
    customer.drop(columns=['birthdate'], inplace=True)

    # RFM 고객 세그먼트
    seg = rfm_df[['customer_id', 'segment']].copy()

    # 이탈 유저 여부
    df = df_avg_cycle[df_avg_cycle['avg_pur_cycle'] != 0].copy()
    Q1 = df['avg_pur_cycle'].quantile(0.25)  # IQR 계산
    Q3 = df['avg_pur_cycle'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR # 상한선 계산 (이상치 기준)
    print(f"IQR 상한선: {upper_bound}일")
    last_pur = merged_df.groupby('customer_id')[['transaction_date']].max().reset_index()
    last_pur.columns = ['customer_id', 'last_purchase']
    present_day = merged_df['transaction_date'].max() + dt.timedelta(days = 2)  # Timestamp('2022-08-02 00:00:00')
    churn_threshold = dt.timedelta(days = upper_bound) # 현재 데이터 상으로는 141일
    last_pur['is_churned'] = (present_day - last_pur['last_purchase']) > churn_threshold
    last_pur.drop(columns='last_purchase', inplace=True)

    # 결과 데이터프레임 생성
    customer_info = pd.merge(df_avg_cycle, df_aov, on='customer_id')
    customer_info = customer_info.merge(rfm_df2, on='customer_id')
    customer_info = customer_info.merge(session_df, on='customer_id')
    customer_info = customer_info.merge(customer, on='customer_id')
    customer_info = customer_info.merge(seg, on='customer_id')
    customer_info = customer_info.merge(last_pur, on='customer_id')
    customer_info = customer_info.merge(conversion_rate, on='customer_id')
    
    return customer_info

customer_info = get_customer_info(merged_df, customer_df, click_stream_df)

# 이탈 고객과 활성 고객 통계 비교

# shaipro-walk 정규성 확인
def get_shapiro(segment1, segment2):
    
    from scipy.stats import shapiro

    segment_data1 = customer_info[customer_info['is_churned'] == segment1]['amount']
    segment_data2 = customer_info[customer_info['is_churned'] == segment2]['amount']
    
    segment_data1 = np.log(segment_data1 + 1) # 왜도 심한 경우 로그 변환
    segment_data2 = np.log(segment_data2 + 1) # 왜도 심한 경우 로그 변환

    shapiro1 = shapiro(segment_data1)
    shapiro2 = shapiro(segment_data2)

    print(f"{segment1}: statistic={shapiro1.statistic}, p-value={shapiro1.pvalue}")
    print(f"{segment2}: statistic={shapiro2.statistic}, p-value={shapiro2.pvalue}")

    if shapiro1.pvalue > 0.05 and shapiro2.pvalue > 0.05:
        print("두 그룹 모두 정규성을 만족합니다.")
    else:
        print("정규성을 만족하지 않는 그룹이 있습니다.")
        
    return segment_data1, segment_data2

# 정규성 시각화 확인
def get_normal(segment1, segment2):
    from scipy.stats import probplot

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    probplot(segment1, dist = 'norm', rvalue=True, plot=ax[0])
    ax[0].set_title('rating_Probplot for Normal Distribution')

    probplot(segment2, dist = 'norm', rvalue=True, plot=ax[1])
    ax[1].set_title('rating_Probplot for Normal Distribution')

    plt.tight_layout()
    plt.show()
    
# 분포 검정
def get_mann(segment1, segment2):
    from scipy.stats import mannwhitneyu

    statistic, pvalue = mannwhitneyu(segment1, segment2, alternative = 'less', method = 'auto')

    print(f'mann-whitney statistic: {statistic}')
    print(f'p-value: {pvalue}')

    if pvalue > 0.05:
        print('귀무가설 채택: 두 샘플의 분포는 동일합니다')
    else:
        print('대립가설 채택: 두 샘플의 분포는 다릅니다')

# T-test 함수
def get_ttest(segment1, segment2):
    from scipy.stats import ttest_ind

    # T-test 수행 (equal_var=False는 Welch's t-test를 수행)
    statistic, pvalue = ttest_ind(segment1, segment2, equal_var=False, alternative='less')

    print(f'T-test statistic: {statistic}')
    print(f'p-value: {pvalue}')

    if pvalue > 0.05:
        print('귀무가설 채택: 두 샘플의 평균은 동일합니다')
    else:
        print('대립가설 채택: 두 샘플의 평균은 다릅니다')        
        
# 시각화로 확인
def get_kde(segment1, segment2):
    mean1 = np.mean(segment1)
    mean2 = np.mean(segment2)
    plt.figure(figsize=(12, 6))
    sns.histplot(segment1, color="blue", label='segment1', kde=True)
    sns.histplot(segment2, color="orange", label='segment2', kde=True)
    plt.axvline(mean1, color='blue', linestyle='--', label=f'Segment 1 Mean: {mean1:.2f}')
    plt.axvline(mean2, color='orange', linestyle='--', label=f'Segment 2 Mean: {mean2:.2f}')
    plt.legend()
    plt.title("Conversion Ratings by Segment")
    plt.show()
True_data, False_data = get_shapiro(True, False)
get_normal(True_data, False_data)
get_mann(True_data, False_data)
get_ttest(True_data, False_data)
get_kde(True_data, False_data)


# 고객 한명 증가할 때 매출 상승 회귀 분석

def lr(merged_df):
    df = merged_df.copy()

    df['year_month'] = df['transaction_date'].dt.to_period('M')
    monthly_cutoff_dates = pd.date_range(start=df['transaction_date'].min(), 
                                        end=df['transaction_date'].max(), 
                                        freq='MS') - dt.timedelta(days=141)
    cutoff_df = pd.DataFrame({'year_month': monthly_cutoff_dates.to_period('M'), 
                            'cutoff_date': monthly_cutoff_dates})
    
    # 각 고객의 마지막 거래 날짜 계산
    last_transactions = df.groupby('customer_id')['transaction_date'].max().reset_index()
    last_transactions.rename(columns={'transaction_date': 'last_transaction'}, inplace=True)

    # 기준 날짜와 비교하여 이탈 여부 판단
    monthly_churn = []
    for _, row in cutoff_df.iterrows():
        cutoff_date = row['cutoff_date']
        year_month = row['year_month']
        
        # 해당 월에 이탈한 고객 수 계산
        churned_customers = last_transactions[last_transactions['last_transaction'] < cutoff_date]
        churned_customers['year_month'] = year_month
        monthly_churn.append(churned_customers)

    monthly_churn_df = pd.concat(monthly_churn)
    
    # 월별 활성 고객 수 계산
    active_customers = df.groupby('year_month')['customer_id'].nunique().reset_index()
    active_customers.rename(columns={'customer_id': 'active_customers'}, inplace=True)

    # 월별 이탈 고객 수 계산
    monthly_churn_count = monthly_churn_df.groupby('year_month')['customer_id'].nunique().reset_index()
    monthly_churn_count.rename(columns={'customer_id': 'churned_customers'}, inplace=True)

    # 데이터 병합 및 이탈률 계산
    monthly_data = pd.merge(active_customers, monthly_churn_count, on='year_month', how='left').fillna(0)
    monthly_data['churn_rate'] = monthly_data['churned_customers'] / monthly_data['active_customers']

    # 유지율 계산
    monthly_data['retention_rate'] = 1 - monthly_data['churn_rate']
    
    # 월별 매출 계산
    revenue = merged_df.copy()
    revenue['transaction_date'] = pd.to_datetime(revenue['transaction_date']).dt.to_period('M')
    revenue = revenue.groupby('transaction_date')[['amount']].sum().round(2).reset_index()
    revenue.rename(columns={'transaction_date': 'year_month'}, inplace=True)
    lr_df = pd.merge(monthly_data, revenue, on = 'year_month')
    
    # 회귀분석 진행
    lr_df = lr_df[lr_df['year_month'] <= '2022-02']
    lr_df = lr_df[lr_df['year_month'] >= '2021-01']
    lr_df['year_month'] = lr_df['year_month'].astype(str)
    import statsmodels.api as sm

    from sklearn.linear_model import LinearRegression

    # 독립 변수와 종속 변수 설정
    X = lr_df[['active_customers']]
    y = lr_df['amount']

    # 상수항 추가 (회귀 분석에는 상수항이 필요)
    X = sm.add_constant(X)

    # 회귀 모델 적합
    model = sm.OLS(y, X).fit()

    # 결과 출력
    print(model.summary())


    # 회귀 모델 생성 및 학습
    model_lr = LinearRegression()
    model_lr.fit(X, y)

    # 회귀 계수와 절편 확인
    coefficients = model_lr.coef_
    intercept = model_lr.intercept_

    coeff_active_customers = model.params['active_customers']
    change_in_revenue = coeff_active_customers * (1100 - 1000)


    # 예측값 계산
    lr_df['predicted_revenue'] = model_lr.predict(X)

    # 그래프 생성
    plt.figure(figsize=(10, 6))
    plt.plot(lr_df['active_customers'], lr_df['amount'], label='Actual Revenue', marker='o', color='blue')
    plt.plot(lr_df['active_customers'], lr_df['predicted_revenue'], label='Predicted Revenue', marker='x', linestyle='--', color='orange')
    plt.title('Actual vs Predicted Revenue')
    plt.xlabel('active_customers')
    plt.ylabel('Revenue')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    print(f"활성 고객 100명 증가시 매출 변화: {change_in_revenue}")
    print("회귀 계수:", coefficients)
    print("절편:", intercept)
    
lr(merged_df)
