import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

kyp = pd.read_csv("C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Kyphosis/Kyphosis.csv")
le  = LabelEncoder()
y = le.fit_transform(kyp['Kyphosis'])
X=kyp.drop('Kyphosis',axis=1)


params = {'min_samples_split':[2,4,6,10,20],
          'min_samples_leaf':[1,5,10,15],
          'max_depth':[None,4,3,2]}

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=24)
dtc = DecisionTreeClassifier(random_state=24)
gcv = GridSearchCV(dtc, param_grid=params,cv=kfold,scoring='neg_log_loss')
gcv.fit(X,y)

print(gcv.best_params_)
print(gcv.best_score_)
########################################
best_tree = gcv.best_estimator_


dtc = DecisionTreeClassifier(random_state=24,min_samples_split=4)
dtc.fit(X, y)
plt.figure(figsize=(25,20))
plot_tree(dtc,feature_names=list(X.columns),
               class_names=['0','1'],
               filled=True,fontsize=18)
plt.show() 

