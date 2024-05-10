import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor 
from warnings import filterwarnings
filterwarnings('ignore')

train = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day6\train.csv", index_col=0)
print(train.isnull().sum().sum())
test = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day6\test.csv")
print(test.isnull().sum().sum())

X_train = train.drop('FloodProbability', axis=1)
y_train = train['FloodProbability']
X_test = test.drop('id', axis=1)

el = ElasticNet()
lr = LinearRegression()
xgb = XGBRegressor(random_state=24)
l_gbm = LGBMRegressor(random_state=24)

stack = StackingRegressor([('EL', el), ('LR', lr), ('XGB', xgb), ('L_GBM', l_gbm)], final_estimator=l_gbm)
stack.fit(X_train,y_train)
y_pred = stack.predict(X_test)
print(y_pred)

submit = pd.DataFrame({'id':test['id'],'Flood Probability':y_pred})
print(submit)

#submit.to_csv('LGBMRegressor-Flood.csv',index=False)

##############################################################################################################################

# kfold = KFold(n_splits=5, shuffle= True, random_state= 24)

# params = {'EN__alpha': np.linspace(0.001, 3, 5),
#           'XB__max_depth': [None, 2],
#           'XB__gamma': np.linspace(0.001, 3, 5),
#           'passthrough': [True, False]}

# gscv = GridSearchCV(stack, param_grid = params, cv = kfold, scoring='r2')

# gscv.fit(X_train,y_train)
# y_pred = gscv.predict(X_test)

# submit = pd.DataFrame({'id':test['id'],'Flood Probability':y_pred})
# print(submit)
