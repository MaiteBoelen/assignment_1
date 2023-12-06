import numpy as np
from multiple_linear_regression import MultipleLinearRegression
from regression_plotter import RegressionPlotter

X = np.array([[1, 2], [2, 4], [3, 1], [4, 3], [5, 5]])  
y = np.array([1, 2, 2, 3, 4])                           

model = MultipleLinearRegression()
model.train(X, y)

plotter = RegressionPlotter(model, X, y)
plotter.plot(plot_all_features=True)  