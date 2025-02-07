import os
import pandas as pd
import numpy as np
import ast
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime

# 최근 3회 거래 기준, customer_id 기준 파생 변수 생성
def get_merged_data(merged_df, click_stream_df):
    from dateutil.relativedelta import relativedelta
    
    data = merged_df.copy()
    # 마지막 3일의 거래 데이터 추출
    last3_dates = data.groupby('customer_id')['transaction_date'].unique().apply(lambda x: x[-3:]).explode()
    last3_dates_df = last3_dates.reset_index().rename(columns={0: 'transaction_date'})
    last_3_data = pd.merge(data, last3_dates_df, on=['customer_id', 'transaction_date'])

    # 행동 양상 데이터 결합
    data = pd.pivot_table(click_stream_df, index='session_id', columns=['event_name'], values='datetime', aggfunc='count').reset_index().fillna(0)
    traffic_data = pd.pivot_table(click_stream_df, index='session_id', columns='traffic_source').reset_index().fillna(0)
    traffic_data.columns = ['session_id', 'mobile', 'web']
    data = data.merge(traffic_data, on='session_id')

    last_3_data.drop_duplicates(inplace=True)
    merged_data = last_3_data.merge(data, how='inner', on='session_id')
    
    # 고객별 AOV 계산
    df_aov = merged_df.groupby('customer_id')['amount'].mean().reset_index(name = 'aov').round(2)

    # 고객별 전환율(total -> booking)
    data = merged_df[['session_id', 'customer_id']]
    df = pd.merge(click_stream_df, data, on='session_id')
    total_session = df.groupby('customer_id')['session_id'].count()
    booking_session = df[df['event_name'] == 'BOOKING'].groupby('customer_id')['session_id'].count()
    conversion_rate = (booking_session / total_session * 100).fillna(0).round(2).reset_index(name = 'conversion_rate')

    # 고객 특성(나이)
    current_date = datetime.now()
    merged_data.loc[:, 'age'] = pd.to_datetime(merged_data['birthdate']).apply(
        lambda x: current_date.year - x.year - ((current_date.month, current_date.day) < (x.month, x.day))
    )
    merged_data.drop(columns=['birthdate'], inplace=True)

    # 가입 기간(개월)
    def get_months(dt_origin, format = '%d-%m-%Y'):
        present_day = merged_df['transaction_date'].max() + dt.timedelta(days = 2)
        delta = relativedelta(present_day, dt_origin)
        return delta.months + (delta.years * 12)

    merged_data['join_period'] = merged_data['first_join_date'].apply(get_months)
    
    # 배송 기간
    merged_data['ship_duration'] = (merged_data['shipment_date'] - merged_data['transaction_date']).dt.days
    
    # 삭제 컬럼
    col = ['product_id', 'season_encoding', 'target_sex_encoding', 'transaction_date', 'transaction_time', 'shipment_date', 'shipment_time', 'original_index', 'first_name', 'first_join_date', 'last_name', 'device_id', 'booking_id', 'home_location', 'articleType', 'subCategory', 'baseColour', 'season', 'year', 'productDisplayName']
    merged_data.drop(columns = col, inplace = True)

    # 결과 데이터프레임 생성
    merged_data = pd.merge(merged_data, df_aov, on='customer_id')
    merged_data = merged_data.merge(last_pur, on='customer_id')
    merged_data = merged_data.merge(conversion_rate, on='customer_id')

    # 범주형 데이터 원핫인코딩
    device_type = pd.get_dummies(merged_data['device_type'], prefix="Device", dtype=int)
    payment_method = pd.get_dummies(merged_data['payment_method'], prefix="Pay", dtype=int)
    payment_status = pd.get_dummies(merged_data['payment_status'], prefix="status", dtype=int)
    promo_code = pd.get_dummies(merged_data['promo_code'], prefix="Promo", dtype=int)
    target_sex = pd.get_dummies(merged_data['target_sex'], prefix="Target", dtype=int)
    usage = pd.get_dummies(merged_data['usage'], prefix="usage", dtype=int)
    masterCategory = pd.get_dummies(merged_data['masterCategory'], prefix="Category", dtype=int)

    # 원래 컬럼 삭제
    merged_data_anal = merged_data.drop(columns=[
        'device_type', 
        'payment_method', 
        'payment_status', 
        'promo_code', 
        'target_sex', 
        'usage', 
        'masterCategory'
    ])

    merged_data_anal = pd.concat([merged_data_anal, 
                                device_type, 
                                payment_method, 
                                payment_status,
                                promo_code,
                                target_sex,
                                usage,
                                masterCategory
                                ], axis=1)
    
    return merged_data, merged_data_anal

# 최종 데이터프레임 생성
def get_final_df(merged_data_anal):
    # 복사본 생성 및 불필요한 열 제거
    df = merged_data_anal.copy()
    df.drop(columns=['session_id'], inplace=True)  # session_id 제거
    df['is_churned'] = df['is_churned'].astype(int)  # is_churned 변환

    # 특정 컬럼에 대해 sum을 적용
    aggfunc_sum = {
        'ADD_PROMO': 'sum',
        'ADD_TO_CART': 'sum',
        'BOOKING': 'sum',
        'CLICK': 'sum',
        'HOMEPAGE': 'sum',
        'ITEM_DETAIL': 'sum',
        'PROMO_PAGE': 'sum',
        'SCROLL': 'sum',
        'SEARCH': 'sum',
        'amount': 'sum',
        'quantity': 'sum',
        'shipment_fee': 'sum'
    }

    # 나머지 숫자형 컬럼에 대해 mean을 적용
    all_columns = df.drop(columns = 'customer_id').select_dtypes(include=['number']).columns.tolist()  # 숫자형 열만 선택
    aggfunc = {col: aggfunc_sum.get(col, 'mean') for col in all_columns}  # sum 또는 mean 설정

    # Pivot Table 생성 (한 번에 처리)
    pivot_result = pd.pivot_table(data=df, index=['customer_id'], aggfunc=aggfunc).reset_index()

    return pivot_result

merged_data, merged_data_anal = get_merged_data(merged_df, click_stream_df)
final = get_final_df(merged_data_anal)

# 상관관계 높은 변수 확인 및 제거
def get_corr_reduced(final):
    df = final.copy()
    # 상관관계 행렬 생성
    correlation_matrix = df.corr()

    # 상관계수 임계값 설정 (예: 0.8)
    threshold = 0.86

    # 상관관계가 높은 변수 쌍 확인
    high_correlation_pairs = []
    for i in range(correlation_matrix.shape[0]):
        for j in range(i + 1, correlation_matrix.shape[1]):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                high_correlation_pairs.append((correlation_matrix.index[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

    # 상관관계가 높은 변수 출력
    print("상관관계가 높은 변수 쌍:")
    for pair in high_correlation_pairs:
        print(f"{pair[0]} - {pair[1]}: {pair[2]:.2f}")
        
    # 상관관계가 높은 변수 중 하나를 제거
    def remove_highly_correlated_features(df, correlation_matrix, threshold=0.8):
        columns_to_remove = set()
        for i in range(correlation_matrix.shape[0]):
            for j in range(i + 1, correlation_matrix.shape[1]):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    # 두 변수 중 하나를 선택하여 제거 (예: 두 번째 변수)
                    columns_to_remove.add(correlation_matrix.columns[j])
        return df.drop(columns=columns_to_remove)

    # 상관관계가 높은 변수 제거 후 데이터프레임 반환
    df_reduced = remove_highly_correlated_features(df, correlation_matrix, threshold=threshold)

    print(f"제거된 컬럼 수: {len(df.columns) - len(df_reduced.columns)}")
    print(f"최종 컬럼 수: {len(df_reduced.columns)}")
    
    return df_reduced

# pca로 공선성 높은 변수 합친 최종 데이터
def get_pca_df(df_reduced):

    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    
    X = df_reduced.drop(columns = ['customer_id', 'is_churned', 'amount'])
    y_churn = df_reduced['is_churned']
    y_amount = df_reduced['amount']
    
    # Step 1: 데이터 표준화
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)

    # Step 2: 상관관계 행렬 계산
    correlation_matrix = np.corrcoef(scaled_data, rowvar=False)
    correlation_matrix = np.nan_to_num(correlation_matrix)

    # Step 3: 계층적 클러스터링을 사용하여 변수 그룹화
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0, metric='euclidean', linkage='ward')
    clustering.fit(correlation_matrix)

    # 클러스터별 변수 그룹 생성
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(X.columns[idx])

    print("클러스터 결과:", clusters)

    # Step 4: 각 클러스터에 대해 PCA 적용
    pca_results = pd.DataFrame(index = X.index)

    for cluster_id, variables in clusters.items():
        if len(variables) > 1:  # 클러스터에 변수가 여러 개인 경우 PCA 적용
            pca = PCA(n_components=1)
            pca_result = pca.fit_transform(X[variables])
            # NumPy 배열로 변환 후 평탄화하여 DataFrame에 저장
            pca_results[f'PCA_Cluster_{cluster_id}'] = pd.Series(pca_result.flatten(), index=X.index)
        else:
            # 클러스터에 변수가 하나인 경우 그대로 유지
            pca_results[variables[0]] = X[variables]
            
    
    # Step 5: PCA 결과와 원본 데이터 결합 (PCA에 사용된 변수 제외)
    remaining_columns = [col for col in X.columns if col not in [var for cluster in clusters.values() for var in cluster]]
    X_pca = pd.concat([X[remaining_columns], pca_results], axis=1)

    return X_pca, y_churn, y_amount

df_reduced = get_corr_reduced(final)
X_pca, y_churn, y_amount = get_pca_df(df_reduced)

# 변수 측정 1. mutual_information
def get_mutual_information(X_trans, y):

    # 1. Mutual Information
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import SelectKBest

    # 데이터 표준화
    scaler = StandardScaler()
    X_trans = scaler.fit_transform(X_trans)
    
    # Mutual Information 계산
    mi_scores = mutual_info_classif(X_trans, y)
    
    # k개 feature 선택
    k_best_selector = SelectKBest(score_func=mutual_info_classif, k=10)
    X_selected = k_best_selector.fit_transform(X_trans, y)
    selected_features = k_best_selector.get_support(indices=True)

    # Mutual Information 점수 시각화
    plt.bar(range(len(mi_scores)), mi_scores)
    plt.xlabel('Feature Index')
    plt.ylabel('Mutual Information Score')
    plt.title('Feature Importance based on Mutual Information')
    plt.show()

    # 데이터프레임 확인
    print(X_trans.iloc[:, selected_features])
    
    return selected_features

# 2. RFECV
def get_rfecv(X_trans, y):
    from sklearn.feature_selection import RFECV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    # 데이터 표준화
    scaler = StandardScaler()
    X_trans = scaler.fit_transform(X_trans)

    # 랜덤포레스트와 RFECV 실행
    estimator = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(5)
    rfecv = RFECV(estimator=estimator, step=5, cv=cv, scoring='f1', min_features_to_select=5, n_jobs=-1)
    rfecv.fit(X_trans, y)

    # 결과 출력
    print("Optimal number of features:", rfecv.n_features_)
    
    # 데이터프레임 확인
    print(X_trans.iloc[:, rfecv.support_])
    
    # 선택된 변수와 중요도 추출
    selected_features = X_trans.columns[rfecv.support_]  # RFECV가 선택한 변수
    feature_importances = rfecv.estimator_.feature_importances_  # 선택된 변수의 중요도

    # 데이터프레임 생성 (변수 이름과 중요도)
    importance_df = pd.DataFrame({
        "Feature": selected_features,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    # 시각화
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"], color="#F0EDCC")
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance from RFECV")
    plt.gca().invert_yaxis()  # 상위 중요도가 위로 오도록 정렬
    plt.tight_layout()
    plt.show()
    
    return rfecv.support_
    
# X_trans = get_trans_X_y(X_pca)
selected_features = get_mutual_information(X_pca, y_churn)
rfecv_support_ = get_rfecv(X_pca, y_churn)

print(X_pca.iloc[:, selected_features].columns)
print(X_pca.iloc[:, rfecv_support_].columns)
