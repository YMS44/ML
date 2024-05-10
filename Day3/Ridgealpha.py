import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

con = pd.read_csv(r"C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Concrete Strength/Concrete_Data.csv")

y=con['Strength']
X = con.drop('Strength',axis=1)

ridge = Ridge(alpha=0.02)

# Dont't take stratify for Regression
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=24)

ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)

print(r2_score(y_test, y_pred))

kfold = KFold(n_splits=5, shuffle=True, random_state=24)
lambdas = np.linspace(0.001, 100,40)
score=[]


# r2_score is defualt for scoring in kfold
for i in lambdas:
    ridge = Ridge(alpha=i)
    result = cross_val_score(ridge, X, y,cv=kfold)
    score.append(result.mean())
i_max = np.argmax(score)
print(score)
print('Best alpha =',lambdas[i_max])

###################################

from sklearn.model_selection import GridSearchCV

params = {'alpha'}
