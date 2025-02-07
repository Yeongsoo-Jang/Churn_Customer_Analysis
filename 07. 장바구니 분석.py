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

col = ['booking_id', 'customer_id', 'transaction_date', 'transaction_time', 'subCategory']
apriori_df = merged_df[col]


# period_day 컬럼 생성
def get_period(time):
    if time.hour < 6:
        return 'Night'  # 밤 (00:00 ~ 05:59)
    elif time.hour < 12:
        return 'Morning'  # 아침 (06:00 ~ 11:59)
    elif time.hour < 18:
        return 'Afternoon'  # 오후 (12:00 ~ 17:59)
    else:
        return 'Evening'  # 저녁 (18:00 ~ 23:59)


apriori_df['period_day'] = apriori_df['transaction_time'].apply(get_period)

# weekday_weekend 컬럼 생성
def get_weekday_weekend(date):
    if date.weekday() < 5:  # 월요일(0) ~ 금요일(4)
        return 'Weekday'
    else:  # 토요일(5), 일요일(6)
        return 'Weekend'

apriori_df['weekday_weekend'] = apriori_df['transaction_date'].apply(get_weekday_weekend)

df = apriori_df.copy()
# 날짜 결합
df['transaction_date'] = df['transaction_date'].astype(str)
df['transaction_time'] = df['transaction_time'].astype(str)
df['transaction_datetime'] = pd.to_datetime(df['transaction_date'] + ' ' + df['transaction_time'])

# 월 추출
df['month'] = df['transaction_datetime'].dt.month
df['month'] = df['month'].replace((1,2,3,4,5,6,7,8,9,10,11,12), 
                                          ('January','February','March','April','May','June','July','August',
                                          'September','October','November','December'))

# 시간 추출
df['hour'] = df['transaction_datetime'].dt.hour
# 텍스트 변경
hour_in_num = (0, 1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)
hour_in_obj = ('00-01', '01-02', '02-03', '03-04', '04-05', '05-06', '06-07', '07-08','08-09','09-10','10-11','11-12','12-13','13-14','14-15',
               '15-16','16-17','17-18','18-19','19-20','20-21','21-22','22-23','23-24')
df['hour'] = df['hour'].replace(hour_in_num, hour_in_obj)

# 요일 추출
df['weekday'] = df['transaction_datetime'].dt.weekday
df['weekday'] = df['weekday'].replace((0,1,2,3,4,5,6), 
                                          ('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'))

# 시간 변환
df.drop(['transaction_datetime'], axis = 1, inplace = True)
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df['transaction_time'] = pd.to_datetime(df['transaction_time']).dt.time
df.dropna(inplace=True)

from mlxtend.frequent_patterns import association_rules, apriori
df = df[df['month'] == 'August']

transactions_str = df.groupby(['customer_id', 'subCategory'])['subCategory'].count().reset_index(name ='Count')

# 피벗 / 0과 1 인코딩
my_basket = transactions_str.pivot_table(index='customer_id', columns='subCategory', values='Count', aggfunc='sum').fillna(0)
# making a function which returns 0 or 1
# 0 means item was not in that transaction, 1 means item present in that transaction

def encode(x):
    if x<=0:
        return 0
    if x>=1:
        return 1

# applying the function to the dataset

my_basket_sets = my_basket.applymap(encode)

# 기존 데이터프레임을 Boolean 타입으로 변환
my_basket_sets = my_basket_sets.astype(bool)
# NaN 값을 False로 대체
my_basket_sets = my_basket_sets.fillna(False)

frequent_items = apriori(my_basket_sets, min_support = 0.01, use_colnames = True)
frequent_items['length'] = frequent_items['itemsets'].apply(lambda x: len(x))

# 최종 데이터 생성
rules = association_rules(frequent_items, metric='lift', min_threshold=1, num_itemsets=len(my_basket_sets))
rules.sort_values('confidence', ascending = False, inplace = True)

# 튜플 데이터 변환
def frozenset_str(rules):
    import re
    
    # 괄호 안의 텍스트 추출
    rules['antecedents'] = rules['antecedents'].astype(str)
    rules['antecedents'] = rules['antecedents'].str.strip()
    rules['antecedents'] = rules['antecedents'].str.extract(r'\((.*?)\)')
    
    rules['consequents'] = rules['consequents'].astype(str)
    rules['consequents'] = rules['consequents'].str.strip()
    rules['consequents'] = rules['consequents'].str.extract(r'\((.*?)\)')
    
    return rules

rules = frozenset_str(rules)

rules.to_csv('basket.csv', sep = ',', encoding='utf-8', index=False)

