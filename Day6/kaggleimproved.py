import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
train = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day6\train.csv", index_col=0)
test = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day6\test.csv")

# Handle missing values
train = train.fillna(train.mean())
test = test.fillna(test.mean())

# Feature engineering
# (Add your feature engineering code here)

# Feature selection
# (Add your feature selection code here)

# Split the data
X_train = train.drop('FloodProbability', axis=1)
y_train = train['FloodProbability']
X_test = test.drop('id', axis=1)

# Try different models
models = {
    'LinearRegression': LinearRegression(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
}

# Perform cross-validation
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"{name} R-squared: {scores.mean():.2f}")

# Hyperparameter tuning (example for RandomForestRegressor)
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}
rf_grid = GridSearchCV(RandomForestRegressor(), rf_params, cv=5)
rf_grid.fit(X_train, y_train)
print(f"Best RandomForestRegressor parameters: {rf_grid.best_params_}")

# Train the best model
best_model = rf_grid.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Create the submission file
submit = pd.DataFrame({'id': test['id'], 'FloodProbability': y_pred})
submit.to_csv("sbt_improved.csv", index=False)