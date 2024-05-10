import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')

import warnings
warnings.filterwarnings("ignore")

label_glass = pd.read_csv("C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Glass Identification/Glass.csv")
print(label_glass)

le = LabelEncoder()

X = label_glass.drop('Type', axis = 1)
y = le.fit_transform(label_glass['Type'])

scaler  = StandardScaler().set_output(transform='pandas')
prcomp = PCA().set_output(transform='pandas')
lr = LogisticRegression()

pipe = Pipeline([('SCL',scaler),("PCA", prcomp),('LR',lr)])
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)
# print(pipe.get_params())
params = {'PCA__n_components': [5,6,7,8,9],
           'LR__C':np.linspace(0.001,3,5),
           'LR__multi_class':['ovr','multinomial']}
gsv = GridSearchCV (pipe, param_grid = params, cv = kfold, scoring = 'neg_log_loss') #, verbose=2 shows calculation
# gsv = GridSearchCV (svm, param_grid = params1, cv = kfold, scoring = 'neg_log_loss')
gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)





##################################  with GaussianNB


gaussian = GaussianNB()
pipe = Pipeline([('SCL',scaler),("PCA", prcomp),('GNB',gaussian)])
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)
#print(pipe.get_params())
params = {'PCA__n_components': [5,6,7,8,9]}
gsv = GridSearchCV (pipe, param_grid = params, cv = kfold, scoring = 'neg_log_loss') #, verbose=2 shows calculation
# gsv = GridSearchCV (svm, param_grid = params1, cv = kfold, scoring = 'neg_log_loss')
gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)

#############################################  with Random forest

rf = RandomForestClassifier(random_state=24)
pipe = Pipeline([('SCL',scaler),("PCA", prcomp),('RF',rf)])
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)
#print(pipe.get_params())
params = {'PCA__n_components': [5,6,7,8,9]}
gsv = GridSearchCV (pipe, param_grid = params, cv = kfold, scoring = 'neg_log_loss') #, verbose=2 shows calculation
# gsv = GridSearchCV (svm, param_grid = params1, cv = kfold, scoring = 'neg_log_loss')
gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)
###################### TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2,random_state=24,perplexity=20).set_output(transform='pandas')
embed_tsne = tsne.fit_transform(X)

embed_tsne['Type'] = le.fit_transform(label_glass['Type'])
embed_tsne['Type'] = embed_tsne['Type'].astype(str)
sns.scatterplot(data=embed_tsne,x='tsne0',y='tsne1',hue='Type')
plt.show()

lr = LogisticRegression()

params = {}

gcv = GridSearchCV(lr, param_grid = params, cv = kfold, scoring = 'neg_log_loss')
tsne = TSNE(n_components=2,random_state=24,perplexity=20).set_output(transform='pandas')
embed_tsne = tsne.fit_transform(X)
gcv.fit(embed_tsne, y)

print(gcv.best_params_)
print(gcv.best_score_)


