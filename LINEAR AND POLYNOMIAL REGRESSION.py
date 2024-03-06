# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Generating some random data for demonstration
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 5 * X + 3 * X**2 + np.random.randn(100, 1)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Polynomial Regression
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# Plotting the results
plt.figure(figsize=(10, 6))

# Plotting the training data
plt.scatter(X_train, y_train, color='blue', label='Training Data')

# Plotting the linear regression line
plt.plot(X_train, linear_reg.predict(X_train), color='red', label='Linear Regression')

# Plotting the polynomial regression line
plt.plot(X_train, poly_reg.predict(X_train_poly), color='green', label='Polynomial Regression')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear vs Polynomial Regression')
plt.legend()
plt.show()

# Evaluating the models
linear_train_mse = mean_squared_error(y_train, linear_reg.predict(X_train))
poly_train_mse = mean_squared_error(y_train, poly_reg.predict(X_train_poly))

print("Linear Regression Training MSE:", linear_train_mse)
print("Polynomial Regression Training MSE:", poly_train_mse)

linear_test_mse = mean_squared_error(y_test, linear_reg.predict(X_test))
poly_test_mse = mean_squared_error(y_test, poly_reg.predict(X_test_poly))

print("Linear Regression Testing MSE:", linear_test_mse)
print("Polynomial Regression Testing MSE:", poly_test_mse)
