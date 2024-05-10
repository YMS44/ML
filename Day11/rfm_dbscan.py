from scipy.cluster.hierarchy import linkage , dendrogram
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


rfm = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Cases\Recency Frequency Monetary\rfm_data_customer.csv",index_col=0)
rfm  = rfm.drop('most_recent_visit',axis=1)
# Create scaler: scaler
scaler = StandardScaler().set_output(transform='pandas')
rfmscaled=scaler.fit_transform(rfm)

clust_DB = DBSCAN(eps=1, min_samples=2)
clust_DB.fit(rfmscaled)
print(clust_DB.labels_)

clust_rfm = rfm.copy()
clust_rfm["Clust"] = clust_DB.labels_
clust_rfm.sort_values(by='Clust')

clust_rfm.groupby('Clust').mean()
clust_rfm.sort_values('Clust')


# rfmscaled['Clust'] = clust_DB.labels_
# rfm_scl_inliers = rfmscaled[rfmscaled['Clust']!=-1]
# print(silhouette_score(rfm_scl_inliers.iloc[:,:-1],rfm_scl_inliers.iloc[:,-1]))

eps_range = [0.2,0.4,0.6,1]
mp_range = [2,3,4,5]
cnt = 0
a =[]
for i in eps_range:
    for j in mp_range:
        clust_DB = DBSCAN(eps=i, min_samples=j)
        clust_DB.fit(rfmscaled.iloc[:,:5])
        if len(set(clust_DB.labels_)) > 2:
            cnt = cnt + 1
            rfmscaled['Clust'] = clust_DB.labels_
            rfm_scl_inliers = rfmscaled[rfmscaled['Clust']!=-1]
            sil_sc = silhouette_score(rfm_scl_inliers.iloc[:,:-1],
                             rfm_scl_inliers.iloc[:,-1])
            a.append([cnt,i,j,sil_sc])
            print(i,j,sil_sc)
    
a = np.array(a)
pa = pd.DataFrame(a,columns=['Sr','eps','min_pt','sil'])
print("Best Paramters:")
pa[pa['sil'] == pa['sil'].max()]

### Labels with best parameters
clust_DB = DBSCAN(eps=0.4, min_samples=2)
clust_DB.fit(rfmscaled.iloc[:,:5])
print(clust_DB.labels_)

clust_rfm = rfm.copy()
clust_rfm["Clust"] = clust_DB.labels_
clust_rfm.sort_values(by='Clust')


clust_rfm.groupby('Clust').mean()
clust_rfm.sort_values('Clust')

print('Total Count=',clust_rfm["Clust"].value_counts())
print('Total Count outliers=',clust_rfm["Clust"].value_counts()[-1])
print('Total outliers=',clust_rfm[clust_rfm['Clust']]==-1)

