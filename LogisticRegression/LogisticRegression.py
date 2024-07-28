import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, epochs=1000, alpha=0.001):
        self.epochs = epochs
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            linear = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum((y_pred - y))

            self.weights -= self.alpha * dw
            self.bias -= self.alpha * db

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred