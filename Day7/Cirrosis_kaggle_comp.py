import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day7\train.csv", index_col=0)
test = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day7\test.csv", index_col=0)

test_dum=pd.get_dummies(test,drop_first=True)

X=train.drop(['Status'],axis=1)
le=LabelEncoder()
y =le.fit_transform(train['Status'])

X_dum=pd.get_dummies(X,drop_first=True)

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=24)

dtc = DecisionTreeClassifier(random_state=24)
dtc.fit(X_dum, y)

# For Neg log loss is best for classification


params = {'min_samples_split': [2,4,6,10,20],
          'min_samples_leaf' : [1,5,10,15],
          'max_depth' : [None, 4, 3, 2]}
gcv = GridSearchCV(dtc, param_grid=params, cv=kfold, scoring='neg_log_loss')
gcv.fit(X_dum,y)

print(gcv.best_score_)
print(gcv.best_params_)

best_tree = gcv.best_estimator_

plt.figure(figsize=(15,10))                  #dtc tree with max_depth=None, and direct fitting 
                                             #without using gcv search
plot_tree(best_tree,feature_names=list(X_dum.columns),
               class_names=['Status_C', 'Status_CL', 'Status_D'],
               filled=True,fontsize=9)
plt.show()

print(best_tree.feature_importances_)

df_imp1 = pd.DataFrame({'Feature': list(X_dum.columns), 'Importance':best_tree.feature_importances_ })

df_imp1.plot(kind='barh',x='Feature')

y_pred_prob=gcv.predict_proba(test_dum)

submit=pd.DataFrame({'id':test_dum.index, 'Status_C': y_pred_prob[:,0] , 'Status_CL': y_pred_prob[:,1], 'Status_D': y_pred_prob[:,2]})

submit.to_csv("Cirrosis_sub.csv", index=False)