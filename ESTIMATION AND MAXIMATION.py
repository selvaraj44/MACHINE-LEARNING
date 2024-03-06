import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, k, max_iter=100, tol=1e-6):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        # Initialization
        n_samples, n_features = X.shape
        self.weights = np.ones(self.k) / self.k
        self.means = X[np.random.choice(n_samples, self.k, replace=False)]
        self.covariances = [np.cov(X.T) for _ in range(self.k)]
        self.prev_likelihood = 0  # Initialize prev_likelihood here

        # EM iterations
        for _ in range(self.max_iter):
            # E-step: compute responsibilities
            responsibilities = self._estimate_responsibilities(X)

            # M-step: update parameters
            self._update_parameters(X, responsibilities)

            # Check for convergence
            if self._converged(X, responsibilities):
                break

    def _estimate_responsibilities(self, X):
        responsibilities = np.zeros((len(X), self.k))
        for i in range(self.k):
            probabilities = multivariate_normal.pdf(X, mean=self.means[i], cov=self.covariances[i])
            responsibilities[:, i] = self.weights[i] * probabilities
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def _update_parameters(self, X, responsibilities):
        # Update weights
        self.weights = responsibilities.mean(axis=0)

        # Update means
        weighted_sum = responsibilities.T @ X
        self.means = weighted_sum / responsibilities.sum(axis=0)[:, None]

        # Update covariances
        for i in range(self.k):
            diff = X - self.means[i]
            weighted_sum = (responsibilities[:, i, None] * diff).T @ diff
            self.covariances[i] = weighted_sum / responsibilities[:, i].sum()

    def _converged(self, X, responsibilities):
        # Calculate log-likelihood
        likelihood = 0
        for i in range(self.k):
            likelihood += multivariate_normal.pdf(X, mean=self.means[i], cov=self.covariances[i]) * self.weights[i]
        likelihood = np.log(likelihood).sum()

        # Check for convergence
        if abs(likelihood - self.prev_likelihood) < self.tol:
            return True
        self.prev_likelihood = likelihood
        return False


# Example usage:
if __name__ == '__main__':
    # Generate some sample data
    np.random.seed(0)
    x1 = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=100)
    x2 = np.random.multivariate_normal(mean=[5, 5], cov=[[1, 0.5], [0.5, 1]], size=100)
    X = np.vstack([x1, x2])

    # Fit Gaussian Mixture Model with 2 components
    gmm = GMM(k=2)
    gmm.fit(X)

    print("Estimated weights:", gmm.weights)
    print("Estimated means:", gmm.means)
    print("Estimated covariances:", gmm.covariances)
