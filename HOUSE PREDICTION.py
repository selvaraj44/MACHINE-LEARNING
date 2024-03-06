# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
house_data = pd.read_csv(r'C:\Users\Admin\Downloads\HOUSEPREDICT.csv')

# Display the first few rows of the dataset to check its structure
print(house_data.head())

# Check the column names in the dataset
print(house_data.columns)

# Ensure that 'Price' is in the column names
if 'Price' not in house_data.columns:
    print("Error: 'Price' column not found in the dataset.")
    exit()

# Data preprocessing
# You may need to handle missing values, encode categorical variables, etc.

# Split the dataset into features and target variable
X = house_data.drop('Price', axis=1) # Features
y = house_data['Price'] # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('R-squared Score:', r2)
