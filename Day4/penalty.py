import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import LabelEncoder

kyphosis = pd.read_csv(r"C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Kyphosis/kyphosis.csv")
le = LabelEncoder()
y = le.fit_transform(kyphosis["Kyphosis"])
X = kyphosis.drop('Kyphosis', axis=1)

lr = LogisticRegression(solver='saga')
lasso = Lasso()
ridge = Ridge()
elastic = ElasticNet()

kfold = KFold(n_splits=5, shuffle=True, random_state=24)


params = {'penalty': ['elasticnet', 'l1', 'l2', None],
          'C':np.linspace(0.01, 10, 5),
          'l1_score': np.linspace(0.001, 1, 4)}
gcv = GridSearchCV(ridge, param_grid=params, cv=kfold, scoring='neg_log_loss')

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

