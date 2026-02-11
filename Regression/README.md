START

1. Import required libraries
      - pandas
      - numpy
      - train_test_split
      - LinearRegression
      - PolynomialFeatures
      - evaluation metrics (R2, MAE, RMSE)

2. Create dataset
      - Define input features X1 and X2
      - Define target variable Y
      - Store data in a table (DataFrame)

3. Separate variables
      - X ← columns X1 and X2
      - y ← column Y

4. Split dataset
      - Divide X and y into training and testing sets
      - Use 70% training and 30% testing

5. Define evaluation function
      FUNCTION evaluate(model_name, actual_values, predicted_values)
            Calculate R2 score
            Calculate MAE
            Calculate RMSE
            Print model name
            Print R2, MAE, RMSE
      END FUNCTION

6. Simple Linear Regression
      - Use only X1 as input feature
      - Train LinearRegression model using training data
      - Predict output for test data
      - Call evaluate function

7. Multiple Linear Regression
      - Use X1 and X2 as input features
      - Train LinearRegression model
      - Predict output for test data
      - Call evaluate function

8. Polynomial Regression
      - Convert input features into polynomial features (degree = 2)
      - Train LinearRegression model using transformed training data
      - Predict output for transformed test data
      - Call evaluate function

END
