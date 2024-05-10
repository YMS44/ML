import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split,StratifiedKFold,KFold
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss,accuracy_score,r2_score
from sklearn.linear_model import LogisticRegression,LinearRegression,Ridge,Lasso
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import Pipeline

boston = pd.read_csv(r"C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Boston.csv")
X = boston.drop('medv', axis=1)
y = boston['medv']
 
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

voting =VotingRegressor([('LR',lr),('RIDGE',ridge),('LASSO',lasso),('TREE',dtr)],weights=[r2_lr,r2_ridge,r2_lasso,
                                                                                          r2_dtr])


voting.fit(X_train,y_train)
y_pred = voting.predict(X_test)
r2_voting=r2_score(y_test, y_pred)

print("Weighted Voting:",r2_voting)

















