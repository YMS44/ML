import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import LabelEncoder

medical = pd.read_csv(r"C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Medical Cost Personal/insurance.csv")

dum_med = pd.get_dummies(medical, drop_first=True)
X = dum_med.drop('charges', axis=1)
y = dum_med['charges']

lr = LinearRegression()
lasso = Lasso()
ridge = Ridge()
elastic = ElasticNet()

kfold = KFold(n_splits=5, shuffle=True, random_state=24)

results = cross_val_score(lr, X, y, cv=kfold)
print(results.mean())
######################## Ridge
params = {'alpha': np.linspace(0.001,100,50)}
gcv = GridSearchCV(ridge, param_grid=params, cv=kfold, scoring='r2')

gcv.fit(X, y)
pd_cv = pd.DataFrame( gcv.cv_results_ )
print(gcv.best_params_)
print(gcv.best_score_)

##################### Lasso

params = {'alpha': np.linspace(0.001,100,50)}
gcv = GridSearchCV(lasso, param_grid=params, cv=kfold, scoring='r2')

gcv.fit(X, y)
pd_cv = pd.DataFrame( gcv.cv_results_ )
print(gcv.best_params_)
print(gcv.best_score_)


##################### ElasticNet


params = {'alpha': np.linspace(0,100,10),'l1_ratio': np.linspace(0,1,10)}
gcv = GridSearchCV(elastic, param_grid=params, cv=kfold, scoring='r2')

gcv.fit(X, y)
pd_cv = pd.DataFrame( gcv.cv_results_ )
print(gcv.best_params_)
print(gcv.best_score_)

