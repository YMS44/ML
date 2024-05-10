from scipy.cluster.hierarchy import linkage , dendrogram
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
rfm = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Cases\Recency Frequency Monetary\rfm_data_customer.csv",index_col=0)
rfm  = rfm.drop('most_recent_visit',axis=1)


scaler  = StandardScaler().set_output(transform='pandas')
scaler.fit(rfm)
rmfscaled = scaler.transform(rfm)

clust = KMeans(n_clusters=2,random_state=24)
clust.fit(rmfscaled)

print(clust.inertia_)

Ks = [2,3,4,5,6]
scores = []
for i in Ks:
    clust = KMeans(n_clusters=i,random_state=24,init='random')
    clust.fit(rmfscaled)
    scores.append(silhouette_score(rmfscaled, clust.labels_))

i_max = np.argmax(scores)
print("Best no. of clusters:", Ks[i_max])
print("Best Score:", scores[i_max])   

clust = KMeans(n_clusters=Ks[i_max],random_state=24)
clust.fit(rmfscaled)
clust_data = rfm.copy()
clust_data['Clust']=clust.labels_
print(clust_data.groupby('Clust').mean())
print(clust_data['Clust'].value_counts())

g = sns.FacetGrid(clust_data,col='Clust')
g.map(sns.histplot,'revenue')
plt.show()

g = sns.FacetGrid(clust_data,col='Clust')
g.map(sns.histplot,'recency_days')
plt.show()

g = sns.FacetGrid(clust_data,col='Clust')
g.map(sns.boxplot,'number_of_orders')
plt.show()

g = sns.FacetGrid(clust_data,col='Clust')
g.map(plt.scatter,'number_of_orders','revenue')
plt.show()



corr = clust_data.corr()
sns.heatmap(corr,cmap='viridis',annot=True)
plt.show()










