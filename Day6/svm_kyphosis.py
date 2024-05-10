import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import  accuracy_score, log_loss

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC

kyp = pd.read_csv("C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Kyphosis/Kyphosis.csv")
le  = LabelEncoder()
y = le.fit_transform(kyp['Kyphosis'])
X=kyp.drop('Kyphosis',axis=1)

svm = SVC(C=0.5, kernel = 'linear', probability=True,random_state=24)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=24, stratify=y)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
y_pred_prob = svm.predict_proba(X_test)
print(accuracy_score(y_test, y_pred))
print(log_loss(y_test, y_pred_prob))


#################################################  Grid Search 
params = {'C':[0.1,2,0.5,1,3]}
svm = SVC(C=0.5, kernel = 'linear', probability=True,random_state=24)
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=24)

gcv = GridSearchCV(svm, param_grid=params,cv=kfold,scoring='neg_log_loss')

gcv.fit(X,y)
pd_cv = pd.DataFrame(gcv.cv_results_)
print(pd_cv)
print(gcv.best_params_)
print(gcv.best_score_)


#################################################### Std , Min max Scaler -  Scaling


std_scl = StandardScaler()
scl_mm = MinMaxScaler()
svm = SVC(kernel = 'linear', probability=True,random_state=24)


pipe = Pipeline([('SCL',None),('SVM',svm)])
params = {'SVM__C': np.linspace(0.001,5,30),'SCL': [std_scl, scl_mm, None]}
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=24)
gcv = GridSearchCV(pipe, param_grid = params, cv = kfold, scoring = 'neg_log_loss')
gcv.fit(X, y)

print(gcv.best_params_)
print(gcv.best_score_)

########################################### Polynomial 

std_scl = StandardScaler()
scl_mm = MinMaxScaler()
svm = SVC(kernel = 'poly', probability=True,random_state=24)


pipe = Pipeline([('SCL',None),('SVM',svm)])
params = {'SVM__C': np.linspace(0.001,5,20),'SCL': [std_scl, scl_mm, None]
          ,'SVM__degree':[2,3],'SVM__coef0':np.linspace(0,3,5)}
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=24)
gcv = GridSearchCV(pipe, param_grid = params, cv = kfold, scoring = 'neg_log_loss',verbose=2)
gcv.fit(X, y)

print(gcv.best_params_)
print(gcv.best_score_)


########################################## Radial svm


std_scl = StandardScaler()
scl_mm = MinMaxScaler()
svm = SVC(kernel = 'rbf', probability=True,random_state=24)


pipe = Pipeline([('SCL',None),('SVM',svm)])
params = {'SVM__C': np.linspace(0.001,5,20),'SCL': [std_scl, scl_mm, None]
          ,'SVM__gamma':np.linspace(0.001,5,5)}
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=24)
gcv = GridSearchCV(pipe, param_grid = params, cv = kfold, scoring = 'neg_log_loss',verbose=2)
gcv.fit(X, y)

print(gcv.best_params_)
print(gcv.best_score_)



































