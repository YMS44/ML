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


scaler  = StandardScaler().set_output(transform='pandas')
scaler.fit(milk)
milkscaled = scaler.transform(milk)

clust = KMeans(n_clusters=2,random_state=24)
clust.fit(milkscaled)

print(clust.inertia_)

Ks = [2,3,4,5,6,7,8,9,10]
scores = []
for i in Ks:
    clust = KMeans(n_clusters=i,random_state=24,init='random')
    clust.fit(milkscaled)
    scores.append(clust.inertia_)
    
plt.scatter(Ks,scores,c='red')
plt.plot(Ks,scores)
plt.title('Screen Plot')
plt.xlabel("Clusters")
plt.ylabel("WSS")
plt.show()