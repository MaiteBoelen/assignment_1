import numpy as np

class MultipleLinearRegression:
    def __init__(self):
        self.intercept = 0
        self.coefficients = None

    def train(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]

        X_transpose = np.transpose(X)
        X_transpose_X_inv = np.linalg.inv(np.dot(X_transpose, X))
        self.coefficients = np.dot(np.dot(X_transpose_X_inv, X_transpose), y)

        self.intercept = self.coefficients[0]
        self.slope = self.coefficients[1:]

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.coefficients)
    

if __name__ == "__main__":
    model = MultipleLinearRegression()

    X = np.array([[1, 4], [2, 5], [3, 2], [4, 3], [5, 1]])
    y = np.array([0, 1, 2, 3, 4])

    print(f"MultipleLinearRegression coefficients -- intercept {model.intercept} -- coefficients {model.coefficients}")
    
    model.train(X, y)

    y_pred = model.predict(X)

    print("Ground truth and predicted values:", y, y_pred, sep="\n")
