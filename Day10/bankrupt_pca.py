import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')
bank = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Cases\Bankruptcy\Bankruptcy.csv")
X = bank.drop(['D','NO'], axis=1)
y = bank['D']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=24,stratify=y)

scaler  = StandardScaler().set_output(transform='pandas')
prcomp = PCA(n_components=11).set_output(transform='pandas')
X_trn_scl = scaler.fit_transform(X_train)
X_trn_pca = prcomp.fit_transform(X_trn_scl)

lr = LogisticRegression()


pipe = Pipeline([("PCA", prcomp),('LR',lr)])
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
y_pred_proba = pipe.predict_proba(X_test)
print(accuracy_score(y_test, y_pred))
print(log_loss(y_test, y_pred_proba))

##############################  GridSearch CV

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)
# print(pipe.get_params())
params = {'PCA__n_components': np.arange(6,12),
           'LR__C':np.linspace(0.001,3,5)}

gsv = GridSearchCV (pipe, param_grid = params, cv = kfold, scoring = 'neg_log_loss') #, verbose=2 shows calculation
# gsv = GridSearchCV (svm, param_grid = params1, cv = kfold, scoring = 'neg_log_loss')
gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)