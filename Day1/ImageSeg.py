import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

imgseg = pd.read_csv(r"C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Image Segmentation/Image_Segmentation.csv")
print(imgseg)
le = LabelEncoder()
y=le.fit_transform(imgseg["Class"])
X = imgseg.drop(['Class'],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=2021, stratify=y)

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_test)
y_pred_prob = gaussian.predict_proba(X_test)
############## Model Evaluation ##############
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

#############################################

#print(roc_auc_score(y_test, y_pred_prob))
print(log_loss(y_test, y_pred_prob))
####################################################
lr=LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_pred_prob = lr.predict_proba(X_test)
############## Model Evaluation ##############
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

##############################################
print(log_loss(y_test, y_pred_prob))