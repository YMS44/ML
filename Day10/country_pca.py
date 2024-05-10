import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')

country =  pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day10\archive\Country-data.csv",index_col=0)

scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(country)
countryscaled = scaler.transform(country)

pca = PCA().set_output(transform='pandas')
principalComponents = pca.fit_transform(countryscaled)

principalComponents.corr()

print(principalComponents.var())
# Variances of PC columns are eigen values of variance-covariance matrix

values, vectors = np.linalg.eig(countryscaled.cov())

print(pca.explained_variance_)
total_var = np.sum(pca.explained_variance_)
print(pca.explained_variance_/total_var)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_ * 100)

from pca import pca

model = pca()

results = model.fit_transform(countryscaled, 
        col_labels=country.columns, row_labels=list(country.index))
model.biplot(label=True, legend=True)
for i in np.arange(0, country.shape[0]):
    plt.text(principalComponents.values[i,0],
             principalComponents.values[i,1],
             list(country.index)[i])
plt.show()








