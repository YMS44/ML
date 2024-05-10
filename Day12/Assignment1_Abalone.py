import pandas as pd
import numpy as np
from xgboost import XGBRFRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from warnings import filterwarnings
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor,RandomForestRegressor
import os

filterwarnings('ignore')
os.chdir(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day12\playground-series-s4e4")
train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv")
dum_train = pd.get_dummies(train,drop_first=True)
X_train = dum_train.drop(['Rings'], axis=1)
y_train = train['Rings']
X_test = pd.get_dummies(test,drop_first=True)
X_test = X_test.drop('id',axis=1)

lr = LinearRegression()
rfc = RandomForestRegressor(random_state=24)
xgb = XGBRFRegressor()
l_gbm = LGBMRegressor()
cat = CatBoostRegressor(random_state=24)

################ Linear Regressor ###########
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
y_pred[y_pred<0]=0
y_pred = np.round(y_pred)

submit = pd.DataFrame({'id':test['id'],'Rings':y_pred})
submit.to_csv("Abalone_LR.csv",index=False)

################### XGB ####################
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)
y_pred[y_pred<0]=0
y_pred = np.round(y_pred)

submit = pd.DataFrame({'id':test['id'],'Rings':y_pred})
submit.to_csv("Abalone_XGB.csv",index=False)

###################### Random Forest ####################
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
y_pred[y_pred<0]=0
y_pred = np.round(y_pred)

submit = pd.DataFrame({'id':test['id'],'Rings':y_pred})
submit.to_csv("Abalone_RFC.csv",index=False)

################# Light GBM ##############################
l_gbm.fit(X_train,y_train)
y_pred = l_gbm.predict(X_test)
y_pred[y_pred<0]=0
y_pred = np.round(y_pred)

submit = pd.DataFrame({'id':test['id'],'Rings':y_pred})
submit.to_csv("Abalone_LGBM.csv",index=False)

################# Stacking ###############################
stack  = StackingRegressor([('LR', lr), ('RFC', rfc), ('XGB', xgb),('LGBM',l_gbm),('CAT',cat)], final_estimator=cat)

stack.fit(X_train,y_train)
y_pred = stack.predict(X_test)
y_pred = np.round(y_pred)

submit = pd.DataFrame({'id':test['id'],'Rings':y_pred})
submit.to_csv("Abalone_Regression.csv",index=False)



