import pandas as pd
from sklearn.naive_bayes import GaussianNB
import os

train = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day6\train.csv", index_col=0)
print(train.isnull().sum().sum())
test = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day6\test.csv")
print(test.isnull().sum().sum())

X_train = train.drop('target', axis=1)
y_train = train['target']
X_test = test.drop('id', axis=1)

nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred_prob = nb.predict_proba(X_test)[:, 1]

submit = pd.DataFrame({'id':test['id'],'target':y_pred_prob})

submit.to_csv("sbt_nb.csv", index=False)
