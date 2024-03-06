import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, verbose=False):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.theta = None
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __compute_cost(self, X, y):
        m = len(y)
        predictions = np.dot(X, self.theta)
        square_error = (predictions - y) ** 2
        cost = (1 / (2 * m)) * np.sum(square_error)
        return cost
    
    def fit(self, X, y):
        X = self.__add_intercept(X)
        n_samples, n_features = X.shape
        self.theta = np.zeros((n_features, 1))  # Adjusting the shape of theta

        for _ in range(self.num_iterations):
            predictions = np.dot(X, self.theta)
            error = predictions - y
            gradient = np.dot(X.T, error) / n_samples
            self.theta -= self.learning_rate * gradient.reshape(-1, 1)  # Reshaping gradient
            if self.verbose and _ % 100 == 0:
                cost = self.__compute_cost(X, y)
                print(f"Iteration {_}: Cost = {cost}")
    
    def predict(self, X):
        X = self.__add_intercept(X)
        return np.dot(X, self.theta)

# Example usage:
if __name__ == "__main__":
    # Generate synthetic data for demonstration
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    # Instantiate the Linear Regression model
    model = LinearRegression(learning_rate=0.01, num_iterations=1000, verbose=True)

    # Fit the model
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)

    # Print the coefficients
    print("Coefficients:", model.theta)
