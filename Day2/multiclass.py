import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

glass = pd.read_csv(r"C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Glass Identification/Glass.csv")
le = LabelEncoder()
y=le.fit_transform(glass["Type"])
X = glass.drop(['Type'],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=24, stratify=y)

####################################################
lr=LogisticRegression(multi_class='ovr')
lr.fit(X_train, y_train)
print(lr.coef_)
print(lr.intercept_)

y_pred = lr.predict(X_test)
y_pred_prob = lr.predict_proba(X_test)
############## Model Evaluation ##############
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred_prob,multi_class='ovr'))

##############################################

#print(roc_auc_score(y_test, y_pred_prob))
print(log_loss(y_test, y_pred_prob))