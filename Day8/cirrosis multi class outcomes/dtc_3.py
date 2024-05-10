import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.tree import plot_tree 
import os 
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

os.chdir(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day8\cirrosis multi class outcomes")

train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)

X_train = pd.get_dummies(train.drop('Status', axis=1), 
                         drop_first=True)
le = LabelEncoder()
y_train = le.fit_transform(train['Status'])
print(le.classes_)


###################################### Decision tree #########################

params = {'min_samples_split':np.arange(2,35,5),
          'min_samples_leaf':np.arange(1, 35, 5),
          'max_depth':[None, 4, 3, 2]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)

dtc = DecisionTreeClassifier(random_state=24)
gcv = GridSearchCV(dtc, param_grid=params, verbose=3,
                   cv=kfold, scoring="neg_log_loss")
gcv.fit(X_train, y_train)
print(gcv.best_params_)
print(gcv.best_score_)

best_tree = gcv.best_estimator_
plt.figure(figsize=(50,20))
plot_tree(best_tree,feature_names=list(X_train.columns),
               class_names=list(le.classes_),
               filled=True,fontsize=25)
plt.title("Best Tree")
plt.show() 

df_imp = pd.DataFrame({'Feature':list(X_train.columns),
                       'Importance':best_tree.feature_importances_})

plt.barh(df_imp['Feature'],
        df_imp['Importance'])
plt.title("Feature Importances")
plt.show()

### Inferencing
dum_tst = pd.get_dummies(test, drop_first=True)
y_pred_prob = best_tree.predict_proba(dum_tst)

submit = pd.DataFrame({'id':list(test.index),
                       'Status_C':y_pred_prob[:,0],
                       'Status_CL':y_pred_prob[:,1],
                       'Status_D':y_pred_prob[:,2]})
submit.to_csv("sbt_dtc.csv", index=False)


###################################### Bagging Decision tree #########################

params = {'estimator__min_samples_split':np.arange(2,35,5),
          'estimator__min_samples_leaf':np.arange(1, 35, 5),
          'estimator__max_depth':[None, 4, 3, 2]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)

dtc = DecisionTreeClassifier(random_state=24)
bag = BaggingClassifier(dtc,random_state=24)
gcv = GridSearchCV(bag, param_grid=params, verbose=3,
                   cv=kfold, scoring="neg_log_loss")
gcv.fit(X_train, y_train)
print(gcv.best_params_)
print(gcv.best_score_)


# best_tree = gcv.best_estimator_
# plt.figure(figsize=(50,20))
# plot_tree(best_tree,feature_names=list(X_train.columns),
#                class_names=list(le.classes_),
#                filled=True,fontsize=25)
# plt.title("Best Tree")
# plt.show()  

"""Error: InvalidParameterError: The 'decision_tree' parameter of plot_tree must be an instance of 'sklearn.tree._classes.DecisionTreeClassifier' or an instance of 'sklearn.tree._classes.DecisionTreeRegressor'. Got BaggingClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=31,
#                                                    random_state=24),
#                   random_state=24) instead.

# <Figure size 3600x1440 with 0 Axes>""" 

# df_imp = pd.DataFrame({'Feature':list(X_train.columns),
#                        'Importance':best_tree.feature_importances_})

# plt.barh(df_imp['Feature'],
#         df_imp['Importance'])
# plt.title("Feature Importances")
# plt.show()

# AttributeError: 'BaggingClassifier' object has no attribute 'feature_importances_'

### Inferencing
dum_tst = pd.get_dummies(test, drop_first=True)
y_pred_prob = best_tree.predict_proba(dum_tst)

submit = pd.DataFrame({'id':list(test.index),
                       'Status_C':y_pred_prob[:,0],
                       'Status_CL':y_pred_prob[:,1],
                       'Status_D':y_pred_prob[:,2]})
submit.to_csv("sbt_dtc_baggingClass.csv", index=False)



##################################### Random Forest Classifier #################


params = {'min_samples_split':np.arange(2,35,10),
          'min_samples_leaf':np.arange(1, 35, 10),
          'max_depth':[None, 4, 3, 2],
          'max_features': [3,4,5,6,7]}

kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)

rfc = RandomForestClassifier(n_estimators = 25,random_state=24)
# dtc = DecisionTreeClassifier(random_state=24)
gcv = GridSearchCV(rfc, param_grid=params, verbose=3,
                   cv=kfold, scoring="neg_log_loss")
gcv.fit(X_train, y_train)
print(gcv.best_params_)
print(gcv.best_score_)

best_tree = gcv.best_estimator_


df_imp = pd.DataFrame({'Feature':list(X_train.columns),
                       'Importance':best_tree.feature_importances_})

plt.barh(df_imp['Feature'],
        df_imp['Importance'])
plt.title("Feature Importances")
plt.show()

### Inferencing
dum_tst = pd.get_dummies(test, drop_first=True)
y_pred_prob = best_tree.predict_proba(dum_tst)

submit = pd.DataFrame({'id':list(test.index),
                       'Status_C':y_pred_prob[:,0],
                       'Status_CL':y_pred_prob[:,1],
                       'Status_D':y_pred_prob[:,2]})
submit.to_csv("sbt_rfc_params.csv", index=False)
