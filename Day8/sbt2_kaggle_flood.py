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
from sklearn.linear_model import LogisticRegression,LinearRegression,Ridge,Lasso, ElasticNet
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import Pipeline
from warnings import filterwarnings
from sklearn.ensemble import BaggingRegressor


filterwarnings('ignore')

train = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day6\train.csv", index_col=0)
print(train.isnull().sum().sum())
test = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day6\test.csv")
print(test.isnull().sum().sum())

X_train = train.drop('FloodProbability', axis=1)
y_train = train['FloodProbability']
X_test = test.drop('id', axis=1)
 
lr = LinearRegression()
std_slr = StandardScaler()
ridge = Ridge()
lasso = Lasso()
dtr = DecisionTreeRegressor(random_state=24)

voting = VotingRegressor([('LR',lr), ('Ridge', ridge), ('Lasso',lasso),('TREE',dtr)])

kfold = KFold(n_splits=5, random_state=24, shuffle=True)

params = {'Ridge__alpha': np.linspace(0.001, 3, 5),
          'Lasso__alpha': np.linspace(0.001, 3, 5),
          'TREE__max_depth': [None, 3, 4, 5],
          'TREE__min_samples_leaf': [2, 4, 5, 8],
          'TREE__min_samples_split': [1, 4, 5, 8]}

rcv = RandomizedSearchCV(voting, param_distributions = params, random_state=24,
                         cv = kfold,scoring = 'r2', n_jobs = -1, n_iter = 20)

rcv.fit(X_train,y_train)
print("\nUsing RSCV:")
print("R2 Score:",rcv.best_score_)
print("Best Parameters:\n",rcv.best_params_)

y_pred = rcv.predict(X_test)

submit = pd.DataFrame({'id':test['id'],
                       'FloodProbability': y_pred})


submit.to_csv("sbt_l2.csv",index=False)

########################################Linear Bagging##########################

lr = LinearRegression()
bagg = BaggingRegressor(lr,n_estimators=25,random_state=24,n_jobs=-1)


bagg.fit(X_train,y_train)

y_pred = bagg.predict(X_test)


submit = pd.DataFrame({'id':test['id'],
                       'FloodProbability': y_pred})


submit.to_csv("sbt_lr_bagging.csv",index=False)


################################################ElasticNet#####################

el = ElasticNet()
kfold = KFold(n_splits=5,shuffle=True,random_state=24)
bag = BaggingRegressor(el,n_estimators=25,random_state=24)
print(bag.get_params())
params={'estimator__alpha':np.linspace(0.001,5,5),
        'estimator__l1_ratio':np.linspace(0,1,8)}
gsv = GridSearchCV(bag, param_grid=params, cv=kfold, n_jobs=-1)
gsv.fit(X_train,y_train)
best_model = gsv.best_estimator_
y_pred = best_model.predict(X_test)
submit = pd.DataFrame({'id':test['id'],
                       'FloodProbability': y_pred})


submit.to_csv("sbt_elasticnet_bagging.csv",index=False)















