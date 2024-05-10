import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRFRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from warnings import filterwarnings
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

milk = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day10\milk.csv", index_col=0)
print(milk)

scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(milk)
milkscaled = scaler.transform(milk)

pca = PCA().set_output(transform='pandas')
principalComponents = pca.fit_transform(milkscaled)

principalComponents.corr()

print(principalComponents.var())
# Variances of PC columns are eigen values of variance-covariance matrix

values, vectors = np.linalg.eig(milkscaled.cov())

print(pca.explained_variance_)
total_var = np.sum(pca.explained_variance_)
print(pca.explained_variance_/total_var)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_ * 100)

ys = np.cumsum(pca.explained_variance_ratio_*100)
xs = np.arange(1,6)
plt.plot(xs,ys)
plt.show()

iris = sns.load_dataset('iris')
sns.pairplot(iris)
plt.show()

sns.pairplot(data=iris, hue='species')
plt.show()

scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(iris.drop('species', axis=1))
iris_scaled = scaler.transform(iris.drop('species', axis=1))

pca = PCA().set_output(transform='pandas')
p_comps = pca.fit_transform(iris_scaled)

print(np.cumsum(pca.explained_variance_ratio_ * 100))

p_comps['species'] = iris['species']
sns.scatterplot(data=p_comps, x='pca0', y='pca1', hue='species')
plt.show()

from pca import pca
import matplotlib.pyplot as plt

model = pca()

results = model.fit_transform(milkscaled, col_labels=milk.columns, row_labels=list(milk.index))
model.biplot(label=True, legend=True)
for i in np.arange(0, milk.shape[0]):
    plt.text(principalComponents.values[i,0],
             principalComponents.values[i,1],
             list(milk.index)[i])
plt.show()