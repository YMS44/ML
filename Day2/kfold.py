# k folds cross validation
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

sonar = pd.read_csv(r"C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Sonar/Sonar.csv")
le = LabelEncoder()
y=le.fit_transform(sonar["Class"])
X = sonar.drop('Class',axis=1)
print(le.classes_)

gaussian = GaussianNB()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)

# For scoring accuracy is default
result = cross_val_score(gaussian, X, y, cv=kfold)

# Scoring by ROC AUC
result2 = cross_val_score(gaussian, X, y, cv=kfold, scoring='roc_auc')

#Mean Roc Score
print(result.mean())
print(result2.mean())


##############################################
lr = LogisticRegression()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)

# For scoring accuracy is default in stratified kfold
result = cross_val_score(lr, X, y, cv=kfold)

# Scoring by ROC AUC
result2 = cross_val_score(lr, X, y, cv=kfold, scoring='roc_auc')

#Mean Roc Score
print(result.mean())
print(result2.mean())