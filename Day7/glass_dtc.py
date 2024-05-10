import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Cases\Glass Identification\Glass.csv")
le = LabelEncoder()
y = le.fit_transform(df['Type']) # y = df.iloc[:,-1]  # Dependent Variable
X = df.drop('Type', axis=1)  # Independent Variable


params = {'min_samples_split':np.arange(2,35,5),
          'min_samples_leaf':np.arange(1,35,5),
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
               class_names=list(le.classes_),
               filled=True,fontsize=18)
plt.show() 
###################################################

best_tree = gcv.best_estimator_
dtc = DecisionTreeClassifier(random_state=24,min_samples_split=4)
dtc.fit(X, y)
plt.figure(figsize=(25,20))
plot_tree(best_tree,feature_names=list(X.columns),
               class_names=list(le.classes_),
               filled=True,fontsize=25)
plt.title("Best Tree")
plt.show() 

########################################################


print(best_tree.feature_importances_)
print(X.columns)
df_imp = pd.DataFrame({'Feature':list(X.columns),
                      'Importance':best_tree.feature_importances_})

plt.bar(df_imp['Feature'],df_imp['Importance'])

plt.title('Feature Importance')
plt.show()

m_left, m_right = 183,31
g_left,g_right =0.679,0.287
m=214
ba_split = (m_left/m)*g_left+(m_right/m)*g_right
ba_reduction=0.737-ba_split


m_left, m_right = 113,70
g_left,g_right =0.6,0.584
m=183

al_split = (m_left/m)*g_left+(m_right/m)*g_right
al_reduction=0.679-al_split


















