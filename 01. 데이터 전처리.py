# 작업 디렉토리 설정
import os
import pandas as pd
import numpy as np
import ast

os.chdir('project/final/e-commerce/EDA/dataset/raw_data')

customer_raw = pd.read_csv('customer.csv')
product_raw = pd.read_csv('product.csv')
transactions_raw = pd.read_csv('transactions.csv')
click_stream_raw = pd.read_csv('click_stream.csv')

# raw_data 전처리

def click_stream_preprocessing(click_stream_raw):
    # copy()
    click_stream_df = click_stream_raw.copy()
    
    # 날짜
    click_stream_df['event_date'] = click_stream_df['event_time'].str.split('T').str[0] # 년-월-일
    click_stream_df['event_hms'] = click_stream_df['event_time'].apply(lambda x : x.split('T')[1].split('.')[0]) # 시간-분-초
    click_stream_df['datetime'] = pd.to_datetime(click_stream_df['event_date'] + ' ' + click_stream_df['event_hms'])
    click_stream_df['event_hms'] = pd.to_datetime(click_stream_df['event_hms'], format='%H:%M:%S').dt.time # dtype 변환
    click_stream_df['event_date'] = pd.to_datetime(click_stream_df['event_date'], format='%Y-%m-%d')

    # event_metadata가 NaN인 경우 event_name으로 채우기
    click_stream_df['event_metadata'].fillna(click_stream_df['event_name'], inplace=True)

    # traffic_source 인코딩
    # {'MOBILE': 1, 'WEB': 0}
    mapping = {'MOBILE': 1, 'WEB': 0}
    click_stream_df['traffic_source_encoding'] = click_stream_df['traffic_source'].map(mapping)
    
    # 2020년 1월 1일 이후 데이터만 사용
    mask = click_stream_df['event_date'] >= '2020-01-01'
    click_stream_df = click_stream_df[mask]
      
    return click_stream_df

def customer_preprocessing(customer_raw):
    
    # copy()
    customer_df = customer_raw.copy()

    # 날짜형으로 변환
    customer_df['birthdate'] = pd.to_datetime(customer_df['birthdate'], infer_datetime_format='%Y-%m-%d')
    customer_df['first_join_date'] = pd.to_datetime(customer_df['first_join_date'], infer_datetime_format='%Y-%m-%d')

    # customer_raw['gender'] mapping 작업
    # {'F': 0, 'M': 1}
    mapping = {'F': 0, 'M': 1}
    customer_df['gender_encoding'] = customer_df['gender'].map(mapping)
        
    return customer_df

def product_preprocessing(product_raw):
    
    # copy()
    product_df = product_raw.copy()
    
    # baseColour가 Null인 행만 처리
    product_df.loc[product_df['baseColour'].isnull(), 'baseColour'] = \
        product_df['productDisplayName'].str.extract(r'(Blue|Black|White|Red|Green)', expand=False).fillna('Not Applicable')

    # Footwear 결측치는 대부분 모든 시즌에 적합하므로 'All Seasons'로 대체
    product_df.loc[product_df['season'].isnull() & (product_df['masterCategory'] == 'Footwear'), 'season'] = 'All Seasons'

    # masterCategory가 'Apparel'이고 season이 NaN인 행을 'Summer'로 대체
    product_df.loc[(product_df['masterCategory'] == 'Apparel') & (product_df['season'].isnull()), 'season'] = 'Summer'

    # year가 NaN인 행을 'unknown'으로 대체
    product_df['year'].fillna('unknown', inplace = True)

    # usage가 NaN인 행을 'None'으로 대체 -- 상품의 유형과 상품명으로도 사용 목적 파악 가능
    product_df['usage'].fillna('None', inplace = True)

    # productDisplayName가 NaN인 행을 'Unknown'으로 대체 -- 상품 유형으로 제품 파악하거나 분석에서 제외 가능
    product_df['productDisplayName'].fillna('Unknown', inplace = True)
    
    # product_raw['gender'] mapping 작업
    # {'Boys': 0, 'Girls': 1, 'Men': 2, 'Unisex': 3, 'Women': 4}
    product_df.rename(columns = {'gender': 'target_sex'}, inplace = True)
    map_target = np.unique(product_df['target_sex'])
    mapping = {value: idx for idx, value in enumerate(map_target)}
    product_df['target_sex_encoding'] = product_df['target_sex'].map(mapping)

    # product['id'] 컬럼 이름 'product_id'로 변경
    product_df.rename(columns = {'id': 'product_id'}, inplace = True)

    # product_raw['season'] mapping 작업
    # {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3, 'All Seasons': 4}
    mapping = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3, 'All Seasons': 4}
    product_df['season_encoding'] = product_df['season'].map(mapping)
           
    return product_df    

def transactions_preprocessing(transactions_raw):
    
    # copy()
    transactions_df = transactions_raw.copy()

    # 날짜
    transactions_df['transaction_date'] = transactions_df['created_at'].str.split('T').str[0] # 년-월-일
    transactions_df['transaction_time'] = transactions_df['created_at'].apply(lambda x : x.split('T')[1].split('.')[0]) # 시간-분-초
    transactions_df['shipment_date'] = transactions_df['shipment_date_limit'].str.split('T').str[0] # 년-월-일
    transactions_df['shipment_time'] = transactions_df['shipment_date_limit'].apply(lambda x : x.split('T')[1].split('.')[0]) # 시간-분-초
    transactions_df['transaction_time'] = pd.to_datetime(transactions_df['transaction_time'], format='%H:%M:%S').dt.time # dtype 변환
    transactions_df['shipment_time'] = pd.to_datetime(transactions_df['shipment_time'], format='%H:%M:%S').dt.time
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'], format='%Y-%m-%d')
    transactions_df['shipment_date'] = pd.to_datetime(transactions_df['shipment_date'], format='%Y-%m-%d')
    
    # payment_status mapping 작업
    # {'Failed': 0, 'Success': 1}
    mapping = {'Failed': 0, 'Success': 1}
    transactions_df['payment_status_encoding'] = transactions_df['payment_status'].map(mapping)

    # null 값 대체 - 'promocode' -> 'Nothing
    transactions_df.fillna('Nothing', inplace=True)

    # promo_code & payment_method 범주형 값 one-hot-encoding
    transactions_df = pd.concat([transactions_df, pd.get_dummies(transactions_df['promo_code'])], axis =1) 
    transactions_df = pd.concat([transactions_df, pd.get_dummies(transactions_df['payment_method'])], axis =1)
    
    # 2020년 1월 1일 이후 데이터만 사용
    mask = transactions_df['transaction_date'] >= '2020-01-01'
    transactions_df = transactions_df[mask]
    
    return transactions_df 

customer_df = customer_preprocessing(customer_raw)
product_df = product_preprocessing(product_raw)
transactions_df = transactions_preprocessing(transactions_raw)
click_stream_df = click_stream_preprocessing(click_stream_raw)

# json 처리
# product_metadata의 문자형을 리스트로 변환
def transactions_json_pre(transactions_raw):

    def convert_list(value):
        try:
            return ast.literal_eval(value) if isinstance(value, str) else value
        except (ValueError, SyntaxError):
            return None
        
    transactions_raw['product_metadata'] = transactions_raw['product_metadata'].apply(convert_list)

    # 압축 해제
    temp_exploded = transactions_raw['product_metadata'].explode()
    temp_exploded.index.name = 'original_index' # 원본 인덱스 이름 설정

    exploded_df = pd.json_normalize(temp_exploded) # json 데이터프레임화

    # exploded_df에 원본 인덱스 추가
    exploded_df['original_index'] = temp_exploded.index
    
    return exploded_df

def clickstream_json_pre(click_stream_raw):

    def convert_list(value):
        try:
            return ast.literal_eval(value) if isinstance(value, str) else value
        except (ValueError, SyntaxError):
            return None
        
    click_stream_raw['product_metadata'] = click_stream_raw['product_metadata'].apply(convert_list)

    # 압축 해제
    temp_exploded = click_stream_raw['product_metadata'].explode()
    temp_exploded.index.name = 'original_index' # 원본 인덱스 이름 설정

    exploded_df = pd.json_normalize(temp_exploded) # json 데이터프레임화

    # exploded_df에 원본 인덱스 추가
    exploded_df['original_index'] = temp_exploded.index
    
    return exploded_df

transactions_json = transactions_json_pre(transactions_raw)
click_stream_json = clickstream_json_pre(click_stream_raw)


# 데이터 병합
# customer + transactions + product 병합

def merged_raw(customer_df, transactions_df, product_df, transactions_json):
    merged_df = pd.merge(customer_df, transactions_df, on = 'customer_id', how='right')
    columns = ['customer_id', 'first_name', 'last_name', 'gender_encoding', 'birthdate', 'device_type', 'device_id', 'home_location_lat', 'home_location_long', 'home_location', 'first_join_date', 'booking_id', 'session_id', 'payment_method', 'payment_status', 'promo_code', 'promo_amount', 'total_amount', 'shipment_fee', 'transaction_date', 'transaction_time', 'shipment_date', 'shipment_time', 'shipment_location_lat', 'shipment_location_long']
    merged_df = pd.merge(transactions_json, merged_df[columns], left_on='original_index', right_index=True)
    merged_df = merged_df.merge(product_df, on='product_id', how='left') # 상품 정보는 없지만 거래 기록은 있는 product_id가 619건

    # 중복값 처리
    def remove_duplicates(df):
        df['promo_amount'] = df.groupby('session_id')['promo_amount'].transform(lambda x: x.where(~x.duplicated(keep='first'), 0))
        df['shipment_fee'] = df.groupby('session_id')['shipment_fee'].transform(lambda x: x.where(~x.duplicated(keep='first'), 0))
        return df

    merged_df = remove_duplicates(merged_df)

    # 결제 금액 총합 다시 계산
    merged_df['total_amount'] = merged_df['item_price'] * merged_df['quantity'] - merged_df['promo_amount'] + merged_df['shipment_fee']
    merged_df.rename(columns={'total_amount': 'amount'}, inplace=True) # 이름 변경
    
    return merged_df

merged_df = merged_raw(customer_df, transactions_df, product_df, transactions_json)

# 데이터 저장

# 작업 디렉토리 설정
os.chdir('project/final/e-commerce/EDA/dataset/dev_data')

customer_df.to_csv('customer_df.csv', sep=',', encoding='utf-8', index=False)
product_df.to_csv('product_df.csv', sep=',', encoding='utf-8', index=False)
transactions_df.to_csv('transactions_df.csv', sep=',', encoding='utf-8', index=False)
click_stream_df.to_csv('click_stream_df.csv', sep=',', encoding='utf-8', index=False)
merged_df.to_csv('merged_df.csv', sep=',', encoding='utf-8', index=False)
transactions_json.to_csv('transactions_json.csv', sep=',', encoding='utf-8', index=False)
click_stream_json.to_csv('click_stream_json.csv', sep=',', encoding='utf-8', index=False)