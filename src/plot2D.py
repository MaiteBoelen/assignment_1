import numpy as np
from multiple_linear_regression import MultipleLinearRegression
from regression_plotter import RegressionPlotter

X = np.array([[1], [2], [3], [4], [5]])  # Single feature
y = np.array([2, 3, 5, 7, 11])           # Target values

model = MultipleLinearRegression()
model.train(X, y)

plotter = RegressionPlotter(model, X, y)
plotter.plot()  


