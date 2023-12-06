import numpy as np

# Assume MultipleLinearRegression is your custom regression class
# from multiple_linear_regression.py
from multiple_linear_regression_test import MultipleLinearRegression

# Import RegressionPlotter from regression_plotter.py
from regression_plotter import RegressionPlotter

# Example data for 2D regression
X = np.array([[1], [2], [3], [4], [5]])  # Single feature
y = np.array([2, 3, 5, 7, 11])           # Target values

# Initialize the MultipleLinearRegression model and train it
model = MultipleLinearRegression()
model.train(X, y)

# Create an instance of RegressionPlotter and plot 2D regression
plotter = RegressionPlotter(model, X, y)
plotter.plot()  # This will generate a 2D plot as we have only one feature


