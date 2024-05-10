from scipy.cluster.hierarchy import linkage , dendrogram
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


df = pd.read_csv(r'C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day13\monthly-milk-production-pounds-p.csv',index_col=0)
df.index = pd.to_datetime(df.index).to_period('M')

df.plot()
plt.title('Monthly Milk Production')
plt.xlabel('Months')
plt.show()


downsampled  = df.resample('Q').sum()
downsampled.index.rename('Quater',inplace=True)
downsampled.plot()
plt.title('Quaterly Milk Production')
plt.xlabel('Quaters')
plt.show()



