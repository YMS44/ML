import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Cases\Glass Identification\Glass.csv")

le = LabelEncoder()
y = le.fit_transform(df['Type']) # y = df.iloc[:,-1]  # Dependent Variable
X = df.drop('Type', axis=1)  # Independent Variable

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=24, stratify=y)

svc = SVC(kernel='linear', probability=True, random_state=24)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print('Accuracy Score: ',accuracy_score(y_test, y_pred))

params = {'C': np.linspace(0.1,5,20)}
kfold = StratifiedKFold(n_splits=5,shuffle=True, random_state=24)

gcv = GridSearchCV(svc, param_grid=params, cv=kfold)
gcv.fit(X,y)

print(gcv.best_params_)
print("Best Accuracy: ",gcv.best_score_)

########################################
std_scaler= StandardScaler()
std_mm= MinMaxScaler()
pipe = Pipeline([('scalar', None), ('SVC', svc)])
print(pipe.get_params())
params = {'scalar':[std_scaler, std_mm, None],'SVC__C': np.linspace(0.001,5,20), 
          'SVC__decision_function_shape': ['ovr'], 'SVC__gamma': ['scale', 'auto']}

gcv_linear = GridSearchCV(pipe, param_grid=params, cv=kfold, scoring='neg_log_loss')
gcv_linear.fit(X,y)

print(gcv_linear.best_params_)
print(gcv_linear.best_score_)


###########################################################
svc = SVC(kernel='poly', probability=True, random_state=24)

std_scaler= StandardScaler()
std_mm= MinMaxScaler()

pipe = Pipeline([('scalar', None), ('SVC', svc)])
print(pipe.get_params())
params = {'scalar':[std_scaler, std_mm, None],'SVC__C': np.linspace(0.001,5,20),
          'SVC__degree': [2,3], 'SVC__coef0': np.linspace(0, 3, 5), 
          'SVC__decision_function_shape': ['ovr']}

gcv_poly = GridSearchCV(pipe, param_grid=params, cv=kfold, scoring='neg_log_loss')

gcv_poly.fit(X,y)

print(gcv_poly.best_params_)
print(gcv_poly.best_score_)


######################################################
svc = SVC(probability=True, random_state=24)

std_scaler= StandardScaler()
std_mm= MinMaxScaler()

pipe = Pipeline([('scalar', None), ('SVC', svc)])
print(pipe.get_params())
params = {'scalar':[std_scaler, std_mm, None],'SVC__C': np.linspace(0.001,5,20), 
          'SVC__gamma': np.linspace(0.001,5,5), 'SVC__decision_function_shape': ['ovr']}

gcv_rbf = GridSearchCV(pipe, param_grid=params, cv=kfold, scoring='neg_log_loss')

gcv_rbf.fit(X,y)

print(gcv_rbf.best_params_)
print(gcv_rbf.best_score_)


##############################################################
svc = SVC(kernel='sigmoid', probability=True, random_state=24)

std_scaler= StandardScaler()
std_mm= MinMaxScaler()

pipe = Pipeline([('scalar', None), ('SVC', svc)])
print(pipe.get_params())
params = {'scalar':[std_scaler, std_mm, None],'SVC__C': np.linspace(0.001,5,20), 
          'SVC__gamma': np.linspace(0.001,5,5), 'SVC__coef0': np.linspace(0, 3, 5), 
          'SVC__decision_function_shape': ['ovr']}

gcv_sigmoid = GridSearchCV(pipe, param_grid=params, cv=kfold, scoring='neg_log_loss')

gcv_sigmoid.fit(X,y)

print(gcv_sigmoid.best_params_)
print(gcv_sigmoid.best_score_)