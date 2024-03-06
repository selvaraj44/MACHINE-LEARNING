# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
sales_data = pd.read_csv(r'C:\Users\Admin\Downloads\futuresale prediction.csv')

# Display the first few rows of the dataframe
print(sales_data.head())

# Splitting the dataset into independent and dependent variables
X = sales_data[['TV', 'Radio', 'Newspaper']].values  # independent variables
y = sales_data['Sales'].values  # dependent variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating and fitting the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Model evaluation
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error:', mean_squared_error(y_test, y_pred))
print('Coefficient of determination (R^2):', r2_score(y_test, y_pred))
