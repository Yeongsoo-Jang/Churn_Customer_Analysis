from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
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


# 덴드로그램
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca.iloc[:, col])

plt.figure(figsize=(15,10))
plt.title('Dendrogram')
plt.xlabel('ID')
plt.ylabel('Euclidean distances')
dgram = dendrogram(linkage(X_scaled, method = 'ward'))
plt.show()

# kmeans 군집
kmeans = KMeans(n_clusters=6, random_state=42).fit(X_scaled)
df_centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X_scaled.columns)

# 레이더 차트 확인
import plotly.graph_objects as go

def plot_radar_from_centroid(df_centroids):
  fig = go.Figure()
  categories = df_centroids.columns
  for row in df_centroids.iterrows():
    fig.add_trace(go.Scatterpolar(
        r=row[1].tolist(),
        theta=categories,
        fill='toself',
        name='cluster {}'.format(row[0])
    ))

  fig.update_layout(
      autosize=False,
      width=1000,
      height=800,
  )
  fig.show()
  
plot_radar_from_centroid(df_centroids)

#Kmeans Centroid를 실제값으로 풀어서 클러스터별 특성 살펴보기
df_cluster_res = pd.DataFrame(scaler.inverse_transform(X_scaled), columns = X_scaled.columns)
df_cluster_res['cluster'] = kmeans.labels_
df_cluster_res.head()

probabilities.columns = ['churn_no_proba', 'churn_yes_proba']
data = pd.concat([probabilities, final[['customer_id', 'is_churned']]], axis = 1)
data = pd.concat([data, df_cluster_res], axis = 1)
data

data.to_csv('cluster.csv', sep = ',', encoding = 'utf-8', index = True)
merged_data.to_csv('merged_data.csv', sep = ',', encoding = 'utf-8', index = False)