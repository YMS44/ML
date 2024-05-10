# k folds cross validation
import pandas as pd
 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

cancer = pd.read_csv(r"C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Cancer/Cancer.csv")
dum_can = pd.get_dummies(cancer, drop_first=True)
y = dum_can['Class_recurrence-events']
X = dum_can.drop(['Class_recurrence-events', 'subjid'], axis=1)
nb = BernoulliNB()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=24, stratify=y)

nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
y_pred_prob = nb.predict_proba(X_test)

print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred_prob[:,1]))
print(log_loss(y_test, y_pred_prob))



#######################################################################
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)

# For scoring accuracy is default
result = cross_val_score(nb, X, y, cv=kfold)

# Scoring by ROC AUC
result2 = cross_val_score(nb, X, y, cv=kfold, scoring='roc_auc')

result3 = cross_val_score(nb, X, y, cv=kfold, scoring='neg_log_loss')

#Mean Roc Score
print(result.mean())
print(result2.mean())
print(result3.mean())
