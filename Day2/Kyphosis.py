# k folds cross validation
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

kyphosis = pd.read_csv(r"C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Kyphosis/Kyphosis.csv")
le = LabelEncoder()
y = le.fit_transform(kyphosis['Kyphosis'])
X = kyphosis.drop('Kyphosis', axis=1)

lr = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24, stratify=y)


lr.fit(X_train, y_train)

print(lr.coef_)
print(lr.intercept_)
y_pred = lr.predict(X_test)
y_pred_prob = lr.predict_proba(X_test)

print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred_prob[:, 1]))
print(log_loss(y_test, y_pred_prob))