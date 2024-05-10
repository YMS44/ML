from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split,StratifiedKFold,KFold
from sklearn.metrics import accuracy_score,log_loss,r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import make_column_selector
import matplotlib.pyplot as plt
import os 
from warnings import filterwarnings
filterwarnings('ignore')

os.chdir(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day8\cirrosis multi class outcomes")

train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)
X_train = pd.get_dummies(train.drop('Status', axis=1), 
                         drop_first=True)
le = LabelEncoder()
y_train = le.fit_transform(train['Status'])
print(le.classes_)
# cat = list(X_train.select_dtypes(include=object).columns)
params = {'learning_rate': np.linspace(0.001, 0.9, 10),
          'max_depth': [None, 2, 3, 4],
          'n_estimators':[25, 50, 100]}

x_gbm = XGBClassifier(random_state=24)
kfold= KFold(n_splits=5,shuffle=True,random_state=24)

gcv = GridSearchCV(x_gbm, param_grid=params,cv=kfold,n_jobs=-1)

gcv.fit(X_train,y_train)
pd_cv = pd.DataFrame(gcv.cv_results_)
print(pd_cv)
print(gcv.best_params_)
print(gcv.best_score_)


best_tree = gcv.best_estimator_


df_imp = pd.DataFrame({'Feature':list(X_train.columns),
                       'Importance':best_tree.feature_importances_})

plt.barh(df_imp['Feature'],
        df_imp['Importance'])
plt.title("Feature Importances")
plt.show()

# ### Inferencing
# dum_tst = pd.get_dummies(test, drop_first=True)
# # y_pred_prob = best_tree.predict_proba(dum_tst)

# submit = pd.DataFrame({'id':list(test.index),
#                        'Status_C':y_pred_prob[:,0],
#                        'Status_CL':y_pred_prob[:,1],
#                        'Status_D':y_pred_prob[:,2]})
# submit.to_csv("catboostreg_.csv", index=False)