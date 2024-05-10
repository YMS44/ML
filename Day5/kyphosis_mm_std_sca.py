import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression,Ridge,ElasticNet,Lasso
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score,GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,accuracy_score,log_loss,roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
kyp = pd.read_csv("C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Kyphosis/Kyphosis.csv")
le  = LabelEncoder()
y = le.fit_transform(kyp['Kyphosis'])
X=kyp.drop('Kyphosis',axis=1)

knn = KNeighborsClassifier(n_neighbors=3)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=24,stratify=y)

############ Without scaling

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_pred_prob = knn.predict_proba(X_test)

print(accuracy_score(y_test,y_pred))
print(roc_auc_score(y_test, y_pred_prob[:,1]) ) 
print(log_loss(y_test, y_pred_prob))


############### Standard Scaling

knn = KNeighborsClassifier(n_neighbors=3)
std_scaler = StandardScaler()

X_scl_trn = std_scaler.fit_transform(X_train)
X_scl_tst  = std_scaler.transform(X_test)

knn.fit(X_scl_trn , y_train)
y_pred = knn.predict(X_scl_tst)
y_pred_prob = knn.predict_proba(X_scl_tst)

print(accuracy_score(y_test,y_pred))
print(roc_auc_score(y_test, y_pred_prob[:,1]) ) 
print(log_loss(y_test, y_pred_prob))


########################  MinMax Scaler
knn = KNeighborsClassifier(n_neighbors=3)
scl_mm  = MinMaxScaler()

scl_mm_trn=scl_mm.fit_transform(X_train)
scl_mm_tst=scl_mm.transform(X_test)

knn.fit(scl_mm_trn , y_train)
y_pred = knn.predict(scl_mm_tst)
y_pred_prob = knn.predict_proba(scl_mm_tst)

print(accuracy_score(y_test,y_pred))
print(roc_auc_score(y_test, y_pred_prob[:,1]) ) 
print(log_loss(y_test, y_pred_prob))











