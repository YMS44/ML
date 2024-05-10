import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression,Ridge,ElasticNet,Lasso
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score,GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

conc = pd.read_csv("C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Concrete Strength/Concrete_Data.csv")
X = conc.drop('Strength', axis=1)
y = conc['Strength']

kfold = KFold(n_splits=5, shuffle=True, random_state=24)
lr = LinearRegression()
ridge = Ridge()
poly = PolynomialFeatures()
pipe = Pipeline([('POLY', poly), ('LR', ridge)])
print(pipe.get_params())

params = {'POLY__degree':[1,2,3,4,5],
          'LR__alpha': np.linspace(0.001, 5,10)}

gcv_ridge = GridSearchCV(pipe, param_grid=params,cv=kfold)

gcv_ridge.fit(X, y)
print(gcv_ridge.best_score_)
print(gcv_ridge.best_params_)
best_model = gcv_ridge.best_estimator_
best_model.fit(X, y)
print(best_model.named_steps.LR.coef_)
print(best_model.named_steps.LR.intercept_)

####################### pipeline using lasso
lasso = Lasso()
poly =PolynomialFeatures()
pipe = Pipeline([('POLY',poly),('LR',lasso)])
params = {'POLY__degree':[1,2,3],
          'LR__alpha': np.linspace(0.001, 5,10)}

gcv_lasso = GridSearchCV(pipe, param_grid=params,cv=kfold)
gcv_lasso.fit(X, y)
print(gcv_lasso.best_score_)
print(gcv_lasso.best_params_)
best_model = gcv_lasso.best_estimator_
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

gcv_els = GridSearchCV(pipe, param_grid=params,cv=kfold)
gcv_els.fit(X, y)
print(gcv_els.best_score_)
print(gcv_els.best_params_)
best_model = gcv_els.best_estimator_
best_model.fit(X, y)
print(best_model.named_steps.LR.coef_)
print(best_model.named_steps.LR.intercept_)



########################################## Inferencing
########### Incase refit = false

# poly = PolynomialFeatures(degree=3)
# ridge = Ridge(alpha=5)
# best_model = Pipeline([('POLY',poly),('LR',elastic)])





best_model = gcv_ridge.best_estimator_

## Unlabelled Data

tst_conc = pd.read_csv('C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Concrete Strength/testConcrete.csv')
pred_strength = best_model.predict(tst_conc)


