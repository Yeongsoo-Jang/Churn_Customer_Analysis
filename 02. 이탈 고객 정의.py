import os
import pandas as pd
import numpy as np
import ast
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

# 유동적으로 날짜 설정

def duration(merged_df, click_stream_df, years = 1):
    columns1 = ['birthdate', 'first_join_date', 'transaction_date', 'shipment_date']
    columns2 = ['transaction_time', 'shipment_time']
    for col in columns1:
        merged_df[f'{col}'] = pd.to_datetime(merged_df[f'{col}'])
        
    for col in columns2:
        merged_df[f'{col}'] = pd.to_datetime(merged_df[f'{col}']).dt.time

    click_stream_df['event_date'] = pd.to_datetime(click_stream_df['event_date'])

    # 최근 1년 거래 데이터 사용(날짜는 유동적으로 변화 가능)
    # 코로나 등 빠르게 변하는 이커머스 시장 상황 반영
    present_day = merged_df['transaction_date'].max() + dt.timedelta(days = 2)
    yearago = present_day - pd.DateOffset(years = years)
    
    mask = merged_df['transaction_date'] >= yearago
    merged_df = merged_df[mask]
    mask = click_stream_df['event_date'] >= yearago
    click_stream_df = click_stream_df[mask]
    
    return merged_df, click_stream_df

merged_df, click_stream_df = duration(merged_df, click_stream_df, years = 1)

# 평균 구매 주기 계산
def avg_purchase(merged_df):

    group = merged_df.groupby('customer_id')

    def purchase_cycle(group):
        if len(group) > 1:
            return group['transaction_date'].diff().mean().days
        else:
            return None
        
    df_avg_cycle = group.apply(lambda group: purchase_cycle(group)).reset_index(name = 'avg_pur_cycle').fillna(0)


    # 데이터 준비
    df = df_avg_cycle[df_avg_cycle['avg_pur_cycle'] != 0].copy()

    # 평균 계산
    mean_value = df['avg_pur_cycle'].mean()

    # Seaborn 스타일 설정
    sns.set_theme(style="whitegrid")  # 배경 스타일 설정
    plt.figure(figsize=(10, 6))  # 그래프 크기 설정

    # 히스토그램 생성 (KDE 제외)
    sns.histplot(
        df['avg_pur_cycle'],  # 데이터
        color="#F0EDCC",  # 히스토그램 색상
        bins=30,  # 구간 개수 설정
        edgecolor="black",  # 막대 테두리 색상
        stat="density",  # 밀도로 정규화
        kde=False  # KDE 제외
    )

    # KDE 플롯 추가 (별도 생성)
    sns.kdeplot(
        df['avg_pur_cycle'],  # 데이터
        color="#02343F",  # KDE 선 색상
        linewidth=2  # 선 두께 설정
    )

    # 평균선 추가
    plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')

    # 그래프 제목 및 축 레이블 추가
    plt.title("Average Purchase Cycle Distribution", fontsize=16, fontweight="bold")  # 제목 설정
    plt.xlabel("Average Purchase Cycle (days)", fontsize=12)  # x축 레이블 설정
    plt.ylabel("Density", fontsize=12)  # y축 레이블 설정

    # 축 눈금 스타일 조정
    plt.xticks(fontsize=10)  # x축 눈금 폰트 크기 조정
    plt.yticks(fontsize=10)  # y축 눈금 폰트 크기 조정

    # 그래프 테두리 제거
    sns.despine()  # 상단과 오른쪽 테두리 제거

    # 범례 추가
    plt.legend(fontsize=12)

    # 그래프 표시
    plt.show()

avg_purchase(merged_df)