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

# rfecv 변수 선택
# AutoGluon 모델링(sklearn 1.4.0버전이어야 함. 현재는 1.6.1)

col = ['PCA_Cluster_9', 'PCA_Cluster_2', 'Target_Men', 'conversion_rate',
       'join_period', 'shipment_fee']

def get_autogluon(X_trans, y):
    
    # Step 1: 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_trans)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_scaled[col], y, test_size=0.2, random_state=42, stratify=y)

    # SMOTE 객체 생성 및 적용
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    from autogluon.tabular import TabularPredictor
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    predictor = TabularPredictor(label='is_churned', problem_type = 'binary', eval_metric='precision',).fit(train_data)

    # 테스트 데이터로 평가
    performance = predictor.evaluate(test_data)
    print(performance)

    # 예측 결과 확인 (테스트 데이터에 대한 확률 출력)
    predictions = predictor.predict_proba(test_data)
    print(predictions.head())
    
    y_pred = predictor.predict(X_test)  # 실제 값과 예측 값
    y_prob = predictor.predict_proba(X_test).iloc[:, 1]
    
    return X_train, X_test, y_train, y_test, predictor, y_pred, y_prob
   
X_train, X_test, y_train, y_test, predictor, y_pred, y_prob = get_autogluon(X_pca, y_churn)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca[col])

# 테스트 데이터에 대한 예측 수행
predictions = predictor.predict(X_scaled)  # 이탈 여부 (0 또는 1)
probabilities = predictor.predict_proba(X_scaled)  # 이탈 확률 (0~1)

# 결과 확인
print("Predictions (이탈 여부):")
print(predictions.head())
print("\nProbabilities (이탈 확률):")
print(probabilities.head())

# 혼동행렬 및 roc_auc 커브 시각화

def get_metrics(X_test, y_test, predictor):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_curve, auc
    
    y_pred = predictor.predict(X_test)  # 실제 값과 예측 값
    cm = confusion_matrix(y_test, y_pred) # 혼동행렬 계산

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Not Churned", "Churned"],
                yticklabels=["Not Churned", "Churned"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    # 예측 확률 계산 (양성 클래스에 대한 확률)
    y_prob = predictor.predict_proba(X_test).iloc[:, 1]

    # ROC 커브 계산
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # ROC 커브 시각화
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 대각선 기준선
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    
    return y_pred, y_prob

y_pred, y_prob = get_metrics(X_test, y_test, predictor)