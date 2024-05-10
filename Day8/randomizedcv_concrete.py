import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split,StratifiedKFold,KFold, RandomizedSearchCV
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss,accuracy_score,r2_score
from sklearn.linear_model import LogisticRegression,LinearRegression,Ridge,Lasso
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import Pipeline
from warnings import filterwarnings

filterwarnings('ignore')

conc = pd.read_csv("C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Concrete Strength/Concrete_Data.csv")
X = conc.drop('Strength', axis=1)
y = conc['Strength']
 
lr = LinearRegression()
ridge =Ridge()
lasso = Lasso()
dtr = DecisionTreeRegressor(random_state=24)

voting =VotingRegressor([('LR',lr),('RIDGE',ridge),('LASSO',lasso),('TREE',dtr)])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=24)

lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
r2_lr=r2_score(y_test, y_pred)

ridge.fit(X_train,y_train)
y_pred = ridge.predict(X_test)
r2_ridge=r2_score(y_test, y_pred)

lasso.fit(X_train,y_train)
y_pred =lasso.predict(X_test)
r2_lasso=r2_score(y_test, y_pred)

dtr.fit(X_train,y_train)
y_pred =  dtr.predict(X_test)
r2_dtr=r2_score(y_test, y_pred)

voting.fit(X_train,y_train)
y_pred = voting.predict(X_test)
r2_voting=r2_score(y_test, y_pred)

print("LR:",r2_lr)
print("Ridge:",r2_ridge)
print("Lasso:",r2_lasso)
print("Tree:",r2_dtr)
print("Voting:",r2_voting)

#######################################  Weighted voting average

# voting =VotingRegressor([('LR',lr),('RIDGE',ridge),('LASSO',lasso),('TREE',dtr)],weights=[r2_lr,r2_ridge,r2_lasso,
#                                                                                           r2_dtr])


# voting.fit(X_train,y_train)
# y_pred = voting.predict(X_test)
# r2_voting=r2_score(y_test, y_pred)

# print("Weighted Voting:",r2_voting)

################################## Parameter Tunnig 


voting =VotingRegressor([('LR',lr),('RIDGE',ridge),('LASSO',lasso),('TREE',dtr)])

kfold= KFold(n_splits=5,shuffle=True,random_state=24)
print(voting.get_params())

params = {'RIDGE__alpha':np.linspace(0.001,3,10),
          'LASSO__alpha':np.linspace(0.001,3,10),
          'TREE__max_depth':[None,3,4,5],
          'TREE__min_samples_leaf':[2,4, 5, 8, 10],
          'TREE__min_samples_split':[1,4, 5, 8, 10]}
rgcv = RandomizedSearchCV(voting,param_distributions=params,
                          cv=kfold,scoring='r2',n_jobs=-1, 
                          n_iter=20, random_state=24)

rgcv.fit(X,y)
print(rgcv.best_params_)
print(rgcv.best_score_)
















