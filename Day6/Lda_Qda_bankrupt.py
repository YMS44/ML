import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression 
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold


from sklearn.model_selection import cross_val_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis



brupt = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Cases\Bankruptcy\Bankruptcy.csv")

y = brupt['D']
X = brupt.drop(['NO','D'],axis=1)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=24, stratify=y)

lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

########################## Linear Discriminant

lda.fit(X_train, y_train)

y_pred = lda.predict(X_test)
y_pred_prob = lda.predict_proba(X_test)

print(accuracy_score(y_test, y_pred))
print(log_loss(y_test, y_pred_prob))

######################### Quadratic Discriminant

qda.fit(X_train, y_train)

y_pred = qda.predict(X_test)
y_pred_prob = qda.predict_proba(X_test)

print(accuracy_score(y_test, y_pred))
print(log_loss(y_test, y_pred_prob))


################################ 

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)

result = cross_val_score(lda, X,y,cv=kfold,scoring='neg_log_loss') # Default it takes scoring as accuracy
print(result.mean())


result = cross_val_score(qda, X,y,cv=kfold,scoring='neg_log_loss') # Default it takes scoring as accuracy
print(result.mean())