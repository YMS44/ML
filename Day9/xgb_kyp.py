from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split,StratifiedKFold,KFold
from sklearn.metrics import accuracy_score,log_loss
from sklearn.tree import DecisionTreeClassifier
from warnings import filterwarnings

filterwarnings('ignore')
kyp = pd.read_csv("C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Kyphosis/Kyphosis.csv")

le  = LabelEncoder()
y = le.fit_transform(kyp['Kyphosis'])
X=kyp.drop('Kyphosis',axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=24,stratify=y)

# Learning rate of gbm is 0.1 by default
x_gbm = XGBClassifier(random_state=24)

x_gbm.fit(X_train, y_train)

y_pred = x_gbm.predict(X_test)
y_pred_prob = x_gbm.predict_proba(X_test)
print(accuracy_score(y_test, y_pred))
print(log_loss(y_test, y_pred_prob))

params = {'learning_rate': np.linspace(0.001, 0.9, 10),
          'max_depth': [None, 2, 3, 4],
          'n_estimators':[25, 50, 100]}

kfold= StratifiedKFold(n_splits=5,shuffle=True,random_state=24)

gcv = GridSearchCV(x_gbm, param_grid=params,cv=kfold,scoring='neg_log_loss',n_jobs=-1)

gcv.fit(X,y)
pd_cv = pd.DataFrame(gcv.cv_results_)
print(gcv.best_params_)
print(gcv.best_score_)

