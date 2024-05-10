from scipy.cluster.hierarchy import linkage , dendrogram
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


df = pd.read_csv(r'C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day13\bike-sharing-demand\train.csv',parse_dates=['datetime'])
df.set_index('datetime',inplace=True)
casual = df['casual']
mon_cas=casual.resample('M').sum()
mon_cas.index.rename('Month',inplace=True)
mon_cas.plot()
plt.title('Monthly casual rentals')
plt.xlabel('Months')
plt.show()

mon_cas=casual.resample('Q').sum()
mon_cas.index.rename('Quarter',inplace=True)
mon_cas.plot()
plt.title('Quarterly casual rentals')
plt.xlabel('Quarters')
plt.show()



