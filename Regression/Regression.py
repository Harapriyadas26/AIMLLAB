import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Larger sample dataset (10 rows instead of 5)
data = {
    'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'X2': [2, 1, 4, 3, 5, 6, 7, 8, 9, 10],
    'Y':  [3, 4, 6, 8,10,12,14,16,18,20]
}

df = pd.DataFrame(data)

# Independent and dependent variables
X = df[['X1', 'X2']]
y = df['Y']

# Split data (30% test for safer evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("-" * 30)

# Function to evaluate models
def evaluate(name, y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(name)
    print("R² Score:", round(r2, 2))
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("-" * 30)

# Simple Linear Regression
X_train_simple = X_train[['X1']]
X_test_simple = X_test[['X1']]

simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train)
y_pred_simple = simple_model.predict(X_test_simple)

# Multiple Linear Regression
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)
y_pred_multi = multi_model.predict(X_test)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# Evaluate all models
evaluate("Simple Linear Regression", y_test, y_pred_simple)
evaluate("Multiple Linear Regression", y_test, y_pred_multi)
evaluate("Polynomial Regression (Degree 2)", y_test, y_pred_poly)
