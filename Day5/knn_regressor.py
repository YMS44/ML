import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression,Ridge,ElasticNet,Lasso
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score,GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor

boston = pd.read_csv(r"C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Boston.csv")
X = boston.drop('medv', axis=1)
y = boston['medv']


knn = KNeighborsRegressor(n_neighbors=4)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=24)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(r2_score(y_test, y_pred))

########################## Gridsearch

knn = KNeighborsRegressor()
kfold = KFold(n_splits=5, shuffle=True, random_state=24)
params = {'n_neighbors':[1,2,3,4,5,6,7,8,9]}
gcv = GridSearchCV(knn,param_grid=params,cv=kfold,scoring='r2')
gcv.fit(X, y)
print(gcv.best_score_)
print(gcv.best_params_)

