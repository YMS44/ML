import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split,StratifiedKFold,KFold
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline



sonar = pd.read_csv(r"C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Sonar/Sonar.csv")
le = LabelEncoder()
y=le.fit_transform(sonar["Class"])
X = sonar.drop('Class',axis=1)

svm_l = SVC(kernel='linear',probability=True,random_state=24)
std_scaler = StandardScaler()
pipe_l = Pipeline([('SCL',std_scaler),('SVM',svm_l)])

svm_r = SVC(kernel='rbf',probability=True,random_state=24)
std_scaler = StandardScaler()
pipe_r = Pipeline([('SCL',std_scaler),('SVM',svm_r)])


lr = LogisticRegression()
lda = LinearDiscriminantAnalysis()
dtc = DecisionTreeClassifier(random_state=24)

voting=VotingClassifier([('LR',lr),('SVM_L',pipe_l),('SVM_R',pipe_r),('LDA',lda),('TREE',dtc)],voting='soft')

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=24,stratify=y)

voting.fit(X_train,y_train)

y_pred = voting.predict(X_test)

y_pred_prob = voting.predict_proba(X_test)[:,1]

print(accuracy_score(y_test, y_pred))
print(log_loss(y_test, y_pred_prob))

#################################################### Gridsearch

kfold= StratifiedKFold(n_splits=5,shuffle=True,random_state=24)
print(voting.get_params())

params = {'SVM_L__SVM__C':np.linspace(0.001,3,10),
          'SVM_R__SVM__C':np.linspace(0.001,3,10),
          'SVM_R__SVM__gamma':np.linspace(0.001,3,10),
          'LR__C':np.linspace(0.001,3,10),
          'TREE__max_depth':[None,3,2]}
gcv = GridSearchCV(voting, param_grid=params,cv=kfold,scoring='neg_log_loss',n_jobs=-1)

gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)