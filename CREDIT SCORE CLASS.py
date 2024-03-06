# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset, replace 'your_dataset.csv' with the actual filename
# Make sure your dataset has features and a target variable (credit score)
try:
    dataset = pd.read_csv(r'C:\Users\Admin\Downloads\CREDITSCORE.csv.')

    # Check if 'credit_score' column exists in the dataset
    if 'credit_score' in dataset.columns:
        # Split the dataset into features and target variable
        X = dataset.drop('credit_score', axis=1)  # Features
        y = dataset['credit_score']  # Target variable

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the decision tree classifier
        clf = DecisionTreeClassifier()

        # Fit the classifier to the training data
        clf.fit(X_train, y_train)

        # Predict the credit scores for the test set
        y_pred = clf.predict(X_test)

        # Evaluate the classifier
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    else:
        print("Error: 'credit_score' column not found in the dataset.")
except FileNotFoundError:
    print("Error: File not found. Please provide the correct path to your dataset.")
except Exception as e:
    print("An error occurred:", e)
