import numpy as np 

class MultipleLinearRegressor:
    def __init__(self):
        self.intercept = 0
        self.slope = 0

model = MultipleLinearRegressor()

def train(self, x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    slope_numerator = np.sum((x - x_mean) * (y - y_mean))
    slope_denominator = np.sum((x - x_mean)**2)
    self.slope = slope_numerator / slope_denominator
    self.intercept = y_mean - self.slope * x_mean

def predict(self, x):

    return self.slope * x + self.intercept

if __name__ == "__main__":
    model = MultipleLinearRegressor()
    x = np.array([1,2,3,4,5,6])
    y = np.array([0,1,2,3,4,5])
    x += np.random.rand(*x.shape)
    print(f"SimpleLinerRegressor coefficients -- intercept {model.intercept}-- slope {model.slope}")
    model.train(x, y)
    y_pred = model.predict(x)
    print("Ground truth and predicted values:", y, y_pred, sep="\n")

