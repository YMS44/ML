import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import make_column_transformer,make_column_selector


hr = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Cases\human-resources-analytics\HR_comma_sep.csv")
ohc = OneHotEncoder(sparse_output = False, drop = 'first')
ct = make_column_transformer((ohc, make_column_selector(dtype_include=object)),
                             ("passthrough", make_column_selector(dtype_include=['int64', 'float64'])), verbose_feature_names_out = False).set_output(transform='pandas')
X =hr.drop('left', axis=1)
X = ct.fit_transform(X)
y = hr['left']



dtc = DecisionTreeClassifier()
dtc.fit(X, y)

#plotting tree
plt.figure(figsize=(15,10))
plot_tree(dtc,feature_names=list(X.columns),
               class_names=['left', 'not_left'],
               filled=True,fontsize=6);




#gridsearch to find best max_depth, min_samples_leaf, min_samples_split
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state=24)

params = {'min_samples_split' : np.arange(2, 35, 5), 'min_samples_leaf':np.arange(1, 35, 5), 'max_depth': [None, 4, 3, 2, 6, 8, 10]}
gcv = GridSearchCV(dtc, param_grid = params, cv = kfold, scoring = 'neg_log_loss')
gcv.fit(X, y)
print(gcv.best_score_)
print(gcv.best_params_)



#plotting best tree
best_tree = gcv.best_estimator_
plt.figure(figsize=(15,10))
plot_tree(best_tree,feature_names=list(X.columns),
               class_names=['left', 'not_left'],
               filled=True,fontsize= 8);

## Infereencing
tst_hr = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Cases\human-resources-analytics\HR_comma_sep.csv")
dum_tst_hr = ct.transform(tst_hr)

best_model = gcv.best_estimator_
best_model.prediction(dum_tst_hr)