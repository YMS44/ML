import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import make_column_transformer,make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold,KFold
from sklearn.metrics import r2_score
from sklearn.tree import plot_tree

house = pd.read_csv("Housing.csv")

ohc = OneHotEncoder(sparse_output = False, drop = 'first')
ct = make_column_transformer((ohc, make_column_selector(dtype_include=object)),
                             ("passthrough", make_column_selector(dtype_include=['int64', 'float64'])),
                             verbose_feature_names_out = False).set_output(transform='pandas')

dum_pd = pd.get_dummies(house, drop_first=True)

X = dum_pd.drop('price', axis=1)
X = ct.fit_transform(X)
y = dum_pd['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

dtc = DecisionTreeRegressor(random_state=24, max_depth=2)
dtc.fit(X_train, y_train)

#plotting tree
plt.figure(figsize=(15,10))
plot_tree(dtc,feature_names=list(X.columns),
               class_names=['left', 'not_left'],
               filled=True,fontsize=6);




#gridsearch to find best max_depth, min_samples_leaf, min_samples_split
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state=24)

params = {'min_samples_split' : np.arange(2, 35, 5), 'min_samples_leaf':np.arange(1, 35, 5), 'max_depth': [None, 4, 3, 2, 6, 8, 10]}
gcv = GridSearchCV(dtc, param_grid = params, cv = kfold, scoring = 'neg_mean_squared_error')
gcv.fit(X, y)
print(gcv.best_score_)
print(gcv.best_params_)



#plotting best tree
best_tree = gcv.best_estimator_
plt.figure(figsize=(35,10))
plot_tree(best_tree,feature_names=list(X.columns),
               class_names=['left', 'not_left'],
               filled=True,fontsize= 8);

y_pred = dtc.predict(X_test)
print(r2_score(y_test, y_pred))

#################################

dtr = DecisionTreeRegressor()
dtr.fit(X, y)

#gridsearch to find best max_depth, min_samples_leaf, min_samples_split
kfold = KFold(n_splits = 5, shuffle = True, random_state=24)

params = {'min_samples_split' : np.arange(2, 35, 5),
          'min_samples_leaf':np.arange(1, 35, 5),
          'max_depth': [None, 4, 3, 2, 6, 8, 10]}
gcv = GridSearchCV(dtc, param_grid = params, cv = kfold, scoring = 'r2')
gcv.fit(X, y)
print(gcv.best_score_)
print(gcv.best_params_)

