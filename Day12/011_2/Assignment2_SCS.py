import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os


os.chdir(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day12\santander-customer-satisfaction")
train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv")

# Separate features and target
X_train = train.drop(['TARGET'], axis=1)
y_train = train['TARGET']
X_test = test.drop('ID',axis=1)

scaler  = StandardScaler().set_output(transform='pandas')
sc=scaler.fit_transform(X_train)


# Perform PCA to find number of components explaining 90% variance
pca = PCA()
pca.fit_transform(sc)

cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_*100)
n_components = (cumulative_variance_ratio >90).sum()

print(f"Number of components explaining more than 90% variance: {n_components}")

########################### Random Forest ########################
# Create pipelines for each model with PCA transformation
rf_pipeline = Pipeline([('scaler',scaler),
    ('pca', PCA(n_components=n_components)),
    ('rf', RandomForestClassifier())
])
rf_pipeline.fit(X_train,y_train)
y_pred = rf_pipeline.predict(X_test)
y_pred_proba = rf_pipeline.predict_proba(X_test)
submit = pd.DataFrame({'ID':test['ID'],'target':y_pred_proba[:,1]})
submit.to_csv('santander_RFC.csv',index=False)


################ XGBoost ###############################################

xgb_pipeline = Pipeline([('scaler',scaler),
    ('pca', PCA(n_components=n_components)),
    ('xgb', XGBClassifier())
])
xgb_pipeline.fit(X_train,y_train)
y_pred = xgb_pipeline.predict(X_test)
y_pred_proba = xgb_pipeline.predict_proba(X_test)
submit = pd.DataFrame({'ID':test['ID'],'target':y_pred_proba[:,1]})
submit.to_csv('santander_XGB.csv',index=False)

########################## CatBoost ###################################

catboost_pipeline = Pipeline([('scaler',scaler),
    ('pca', PCA(n_components=n_components)),
    ('catboost', CatBoostClassifier())
])
catboost_pipeline.fit(X_train,y_train)
y_pred = catboost_pipeline.predict(X_test)
y_pred_proba = catboost_pipeline.predict_proba(X_test)
submit = pd.DataFrame({'ID':test['ID'],'target':y_pred_proba[:,1]})
submit.to_csv('santander_CatBoost.csv',index=False)


######################### LightGBM ################################################

lgbm_pipeline = Pipeline([('scaler',scaler),
    ('pca', PCA(n_components=n_components)),
    ('lgbm', LGBMClassifier())
])
lgbm_pipeline.fit(X_train,y_train)
y_pred = lgbm_pipeline.predict(X_test)
y_pred_proba = lgbm_pipeline.predict_proba(X_test)
submit = pd.DataFrame({'ID':test['ID'],'target':y_pred_proba[:,1]})
submit.to_csv('santander_LGBM.csv',index=False)

################################ All in single pipeline #############################

stack  = StackingClassifier([('RFC',RandomForestClassifier()),('Cat',CatBoostClassifier()),
                            ('XGB',XGBClassifier()),('LGBM',LGBMClassifier())],passthrough=True,
                            final_estimator = XGBClassifier(),verbose=4)
stack.fit(X_train,y_train)
y_pred = stack.predict(X_test)
y_pred_proba = stack.predict_proba(X_test)
submit = pd.DataFrame({'ID':test['ID'],'target':y_pred_proba[:,1]})
submit.to_csv('santander_AllModels.csv',index=False)