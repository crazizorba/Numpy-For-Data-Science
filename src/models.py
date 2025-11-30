import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.05, num_iterations=5000, lambda_reg=0.001):
        """
        Logistic Regression from scratch using NumPy.
        - learning_rate: Alpha for gradient descent
        - num_iterations: Number of GD steps
        - lambda_reg: Regularization strength (L2)
        """
        self.lr = learning_rate
        self.iters = num_iterations
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, z):
        """Sigmoid function - vectorized."""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """Fit the model using gradient descent."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)

            # Gradient for weights (vectorized)
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y)) + (self.lambda_reg / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(predictions - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Calculate loss
            loss = - (1 / n_samples) * np.sum(y * np.log(predictions + 1e-8) + (1 - y) * np.log(1 - predictions + 1e-8)) + (self.lambda_reg / (2 * n_samples)) * np.sum(self.weights**2)
            self.losses.append(loss)

    def predict_proba(self, X):
        """Predict probabilities."""
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)

    def predict(self, X, threshold=0.38):
        """Predict binary labels."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

def accuracy(y_true, y_pred):
    """Accuracy metric."""
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    """Precision metric."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp + 1e-8)

def recall(y_true, y_pred):
    """Recall metric."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))  
    return tp / (tp + fn + 1e-8)

def f1_score(y_true, y_pred):
    """F1 score metric."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-8)

def roc_auc(y_true, y_proba):
    """ROC AUC score from scratch."""
    sort_idx = np.argsort(-y_proba)
    y_true = y_true[sort_idx]
    tpr = np.cumsum(y_true) / np.sum(y_true)
    fpr = np.cumsum(1 - y_true) / np.sum(1 - y_true)
    auc = np.sum(np.diff(fpr) * tpr[1:])
    return auc

