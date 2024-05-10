import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

con = pd.read_csv(r"C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Concrete Strength/Concrete_Data.csv")

y=con['Strength']
X = con.drop('Strength',axis=1)

lr = LinearRegression()

# Dont't take stratify for Regression
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=24)

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print(r2_score(y_test, y_pred))

kfold = KFold(n_splits=5, shuffle=True, random_state=24)

# r2_score is defualt for scoring in kfold
result = cross_val_score(lr, X, y,cv=kfold, scoring="r2")

print(result)
print(result.mean())