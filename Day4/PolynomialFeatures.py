import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

pizza = pd.read_csv('C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/pizza.csv')
X = pizza[['Promote']]
y = pizza['Sales']

poly = PolynomialFeatures(degree=2).set_output(transform=None)

X_poly = poly.fit_transform(X)

lr = LinearRegression()
lr.fit(X_poly, y)
print(lr.coef_, lr.intercept_)

########################################################################################

insure = pd.read_csv(r"C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Insure_auto.csv")
dum_med = pd.get_dummies(insure, drop_first=True)
X = dum_med.drop('Operating_Cost', axis=1)
y = dum_med['Operating_Cost']

poly = PolynomialFeatures(degree=3).set_output(transform=None)

X_poly = poly.fit_transform(X)

lr = LinearRegression()
lr.fit(X_poly, y)
print(lr.coef_, lr.intercept_)