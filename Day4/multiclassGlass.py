import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import LabelEncoder

glass = pd.read_csv(r"C:/Users/Administrator.DAI-PC2\Desktop/MachineLearning/Cases/Glass Identification/Glass.csv")
y = glass['Type']
X = glass.drop('Type', axis=1)

lr = LogisticRegression(solver='saga', random_state=24)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)

params = {'penalty': ['elasticnet', 'l1', 'l2', None],
          'C':np.linspace(0.01, 10, 5),
          'l1_ratio': np.linspace(0.001, 1, 4),
          'multi_class':['ovr', 'multinomial']}

gcv = GridSearchCV(lr, param_grid=params, cv = kfold, scoring = 'neg_log_loss')

gcv.fit(X, y)
pd_cv = pd.DataFrame(gcv.cv_results_)
print(gcv.best_params_)
print(gcv.best_score_)