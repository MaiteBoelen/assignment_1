import numpy as np

class MultipleLinearRegression:
    def __init__(self):
        self.intercept = 0
        self.coefficients = None

    def train(self, X, y):
        # Add a column of ones to X for the intercept term
        X = np.c_[np.ones(X.shape[0]), X]

        # Calculate coefficients using the normal equation (X^T * X)^-1 * X^T * y
        X_transpose = np.transpose(X)
        X_transpose_X_inv = np.linalg.inv(np.dot(X_transpose, X))
        self.coefficients = np.dot(np.dot(X_transpose_X_inv, X_transpose), y)

        # Set intercept and coefficients
        self.intercept = self.coefficients[0]
        self.slope = self.coefficients[1:]

    def predict(self, X):
        # Make sure X is a 2D array
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # Add a column of ones to X for the intercept term
        X = np.c_[np.ones(X.shape[0]), X]
        # Use the learned coefficients to make predictions
        return np.dot(X, self.coefficients)
    

if __name__ == "__main__":
    model = MultipleLinearRegression()

    # Example dataset with multiple features
    X = np.array([[1, 4], [2, 5], [3, 2], [4, 3], [5, 1]])
    y = np.array([0, 1, 2, 3, 4])

    print(f"MultipleLinearRegression coefficients -- intercept {model.intercept} -- coefficients {model.coefficients}")
    
    # Train the model
    model.train(X, y)

    # Make predictions
    y_pred = model.predict(X)

    print("Ground truth and predicted values:", y, y_pred, sep="\n")
