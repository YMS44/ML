import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

pizza = pd.read_csv('C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/pizza.csv')
X = pizza[['Promote']]
y = pizza['Sales']

poly = PolynomialFeatures(degree=2).set_output(transform='pandas')

X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=24)

lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.coef_, lr.intercept_)

print('##################### Poly 1 #####################')

boston = pd.read_csv(r"C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Boston.csv")
X = boston.drop('medv', axis=1)
y = boston['medv']

poly = PolynomialFeatures(degree=1).set_output(transform='pandas')

X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=24)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
# print(lr.coef_, lr.intercept_)

print(r2_score(y_test, y_pred))

print('##################### Poly 2 #####################')

poly = PolynomialFeatures(degree=2).set_output(transform='pandas')

X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=24)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
# print(lr.coef_, lr.intercept_)

print(r2_score(y_test, y_pred))

print('##################### Poly 3 #####################')

poly = PolynomialFeatures(degree=3).set_output(transform='pandas')

X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=24)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
# print(lr.coef_, lr.intercept_)

print(r2_score(y_test, y_pred))