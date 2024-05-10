import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression,Ridge,ElasticNet,Lasso
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score,GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline


boston = pd.read_csv(r"C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Boston.csv")
X = boston.drop('medv', axis=1)
y = boston['medv']

kfold = KFold(n_splits=5, shuffle=True, random_state=24)
lr = LinearRegression()

degrees = [1,2,3,4]
scores = []
for i in degrees:
    print(f'########## cross val score POLY {i} #################')
    poly = PolynomialFeatures(degree=i)
    pipe = Pipeline([('POLY', poly), ('LR', lr)])
    # For scoring accuracy is default
    result = cross_val_score(pipe, X, y, cv=kfold)
    scores.append(result.mean())
    print(result.mean())

i_max = np.argmax(scores)
print("Best Degree = ", degrees[i_max])
print("Best Score = ",scores[i_max])

####################### pipeline using GridSearch 
ridge = Ridge()
poly = PolynomialFeatures()
pipe = Pipeline([('POLY', poly), ('LR', ridge)])
print(pipe.get_params())

params = {'POLY__degree':[1,2,3,4,5],
          'LR__alpha': np.linspace(0.001, 5,10)}

gcv = GridSearchCV(pipe, param_grid=params,cv=kfold)

gcv.fit(X, y)
print(gcv.best_score_)
print(gcv.best_params_)

####################### pipeline using lasso
lasso = Lasso()
poly =PolynomialFeatures()
pipe = Pipeline([('POLY',poly),('LR',lasso)])
params = {'POLY__degree':[1,2,3],
          'LR__alpha': np.linspace(0.001, 5,10)}

gcv = GridSearchCV(pipe, param_grid=params,cv=kfold)
gcv.fit(X, y)
print(gcv.best_score_)
print(gcv.best_params_)
best_model = gcv.best_estimator_
best_model.fit(X, y)
print(best_model.named_steps.LR.coef_)
print(best_model.named_steps.LR.intercept_)
####################### pipeline using ElasticNet
elastic = ElasticNet()
poly =PolynomialFeatures()
pipe = Pipeline([('POLY',poly),('LR',elastic)])
params = {'POLY__degree':[1,2,3],
          'LR__alpha': np.linspace(0.001, 5,10),
          'LR__l1_ratio':np.linspace(0,1,5)}

gcv = GridSearchCV(pipe, param_grid=params,cv=kfold)
gcv.fit(X, y)
print(gcv.best_score_)
print(gcv.best_params_)


