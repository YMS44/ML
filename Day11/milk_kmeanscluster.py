from scipy.cluster.hierarchy import linkage , dendrogram
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
milk = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day10\milk.csv",index_col=0)
print(milk)

scaler = StandardScaler().set_output(transform='pandas')
df_scaled = scaler.fit_transform(milk)


Ks = [2,3,4,5]
scores = []
for i in Ks:
    clust = KMeans(n_clusters=i,random_state=24,init='random')
    clust.fit(df_scaled)
    scores.append(silhouette_score(df_scaled, clust.labels_))

i_max = np.argmax(scores)
print("Best no. of clusters:", Ks[i_max])
print("Best Score:", scores[i_max])

clust = KMeans(n_clusters=Ks[i_max],random_state=24)
clust.fit(df_scaled)

clust_data = milk.copy()
clust_data['Clust']=clust.labels_
print(clust_data.groupby('Clust').mean())

############################################################




