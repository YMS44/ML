import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression,Ridge,ElasticNet,Lasso
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score,GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,accuracy_score,log_loss,roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

kyp = pd.read_csv("C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Kyphosis/Kyphosis.csv")
le  = LabelEncoder()
y = le.fit_transform(kyp['Kyphosis'])
X=kyp.drop('Kyphosis',axis=1)

knn = KNeighborsClassifier(n_neighbors=5)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=24,stratify=y)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_pred_prob = knn.predict_proba(X_test)

print(accuracy_score(y_test,y_pred))
print(roc_auc_score(y_test, y_pred_prob[:,1]) ) 
print(log_loss(y_test, y_pred_prob))

################################### Grid Search


knn = KNeighborsClassifier()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
params = {'n_neighbors':[1,2,3,4,5,6,7,8]}
gcv = GridSearchCV(knn,param_grid=params,cv=kfold,scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_score_)
print(gcv.best_params_)