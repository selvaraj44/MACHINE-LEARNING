from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
import numpy as np

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the Decision Tree Classifier using ID3 algorithm
clf = DecisionTreeClassifier(criterion='entropy')

# Train the Decision Tree Classifier
clf.fit(X_train, y_train)

# Print the decision tree rules
tree_rules = export_text(clf, feature_names=iris.feature_names)
print("Decision Tree Rules:\n", tree_rules)

# Predict using the trained classifier
def predict_sample(sample):
    prediction = clf.predict([sample])
    species = iris.target_names[prediction][0]
    return species

# Example of a new sample
new_sample = [6.7, 3.0, 5.2, 2.3]  # Sample features (sepal length, sepal width, petal length, petal width)

# Predict the species of the new sample
predicted_species = predict_sample(new_sample)
print("\nPredicted Species for the new sample:", predicted_species)
