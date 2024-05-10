from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, StackingClassifier
import numpy as np
import pandas as pd
from sklearn.svm import SVC
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


lr = LogisticRegression()
svm = SVC(kernel='linear', probability=True, random_state=24)
dtc = DecisionTreeClassifier(random_state=24)
rfc = RandomForestClassifier(random_state=24)

stack = StackingClassifier([('LR', lr), ('SVM', svm), ('TREE', dtc)], final_estimator=rfc)
stack.fit(X_train, y_train)

y_pred = stack.predict(X_test)
y_pred_prob = stack.predict_proba(X_test)

print(accuracy_score(y_test, y_pred))
print(log_loss(y_test, y_pred_prob))

################### With pass through
stack = StackingClassifier([('LR', lr), ('SVM', svm), ('TREE', dtc)], final_estimator=rfc,passthrough=True)
stack.fit(X_train, y_train)

y_pred = stack.predict(X_test)
y_pred_prob = stack.predict_proba(X_test)

print(accuracy_score(y_test, y_pred))
print(log_loss(y_test, y_pred_prob))

######################## Grid Search CV

print(stack.get_params())
stack = StackingClassifier([('LR', lr), ('SVM', svm), ('TREE', dtc)], final_estimator=rfc)
params = {
    'LR__C':np.linspace(0.01,3,5),
    'SVM__C':np.linspace(0.01,3,5),
    'TREE__max_depth':[None,2,3,4],
    'final_estimator__max_features':[2,3],
    'passthrough':[False,True]
    }

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
gcv = GridSearchCV(stack, param_grid=params, cv=kfold, scoring='neg_log_loss',n_jobs=-1)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
