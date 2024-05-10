from scipy.cluster.hierarchy import linkage , dendrogram
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv(r'C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day13\monthly-milk-production-pounds-p.csv')

series = df['Milk']
result = seasonal_decompose(series,model='additive',period=12)

result.plot()
plt.title('Additive Decomposition')
plt.show()

