import numpy as np
from multiple_linear_regression import MultipleLinearRegression
from regression_plotter import RegressionPlotter

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3, 3]])  
y = np.array([2, 3, 3, 5, 5])                           

model = MultipleLinearRegression()
model.train(X, y)

plotter = RegressionPlotter(model, X, y)
plotter.plot()  

