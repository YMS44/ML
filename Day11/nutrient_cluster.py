from scipy.cluster.hierarchy import linkage , dendrogram
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.decomposition import PCA
nutrient = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day11\nutrient.csv", index_col=0)
print(nutrient)




scaler = StandardScaler().set_output(transform='pandas')
df_scaled = scaler.fit_transform(nutrient)
link = "ward"
mergings = linkage(df_scaled,method=link)
dendrogram(mergings,
           labels=list(df_scaled.index))
plt.title(link+" linkage")
plt.show()

#####################################################
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

clust = AgglomerativeClustering(n_clusters=4)
clust.fit(df_scaled)

print(clust.labels_)

pca = PCA().set_output(transform='pandas')
principalComponents = pca.fit_transform(df_scaled)

print(pca.explained_variance_ratio_*100)
principalComponents['Clust'] = clust.labels_
principalComponents['Clust'] =principalComponents['Clust'].astype(str)

sns.scatterplot(principalComponents,x='pca0', y='pca1',hue='Clust')
for i in range(0, nutrient.shape[0] ):
    plt.text(principalComponents.values[i,0], 
            principalComponents.values[i,1], 
             list(nutrient.index)[i])
plt.show()


print(silhouette_score(df_scaled, clust.labels_))

Ks = [2,3,4,5]
scores = []
for i in Ks:
    clust = AgglomerativeClustering(n_clusters=i)
    clust.fit(df_scaled)
    scores.append(silhouette_score(df_scaled, clust.labels_))

i_max = np.argmax(scores)
print("Best no. of clusters:", Ks[i_max])
print("Best Score:", scores[i_max])






