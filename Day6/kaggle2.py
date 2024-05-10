import pandas as pd
from sklearn.linear_model import LinearRegression
import os

train = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day6\train.csv", index_col=0)
print(train.isnull().sum().sum())
test = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day6\test.csv")
print(test.isnull().sum().sum())

X_train = train.drop('FloodProbability', axis=1)
y_train = train['FloodProbability']
X_test = test.drop('id', axis=1)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

submit = pd.DataFrame({'id':test['id'],'FloodProbability':y_pred})
submit.to_csv("sbt_lr.csv", index=False)

