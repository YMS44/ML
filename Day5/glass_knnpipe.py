import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, log_loss

import warnings
warnings.filterwarnings("ignore")

label_glass = pd.read_csv("C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Glass Identification/Glass.csv")
print(label_glass)

le = LabelEncoder()

X = label_glass.drop('Type', axis = 1)
y = le.fit_transform(label_glass['Type'])

kfold = KFold(n_splits = 5, shuffle = True, random_state = 24)

std_scl = StandardScaler()
scl_mm = MinMaxScaler()
knn = KNeighborsRegressor()

pipe = Pipeline([('SCL',None),('KNN',knn)])

params = {'KNN__n_neighbors': [1,2,3,4,5,6,7,8,9,10],'SCL': [std_scl, scl_mm, None]}

gcv = GridSearchCV(pipe, param_grid = params, cv = kfold, scoring = 'r2')
gcv.fit(X, y)

print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_
print(best_model)


