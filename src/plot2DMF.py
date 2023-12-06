import numpy as np

# Assume MultipleLinearRegression is your custom regression class
# from multiple_linear_regression.py
from multiple_linear_regression import MultipleLinearRegressor

# Import RegressionPlotter from regression_plotter.py
from regression_plotter import RegressionPlotter

# Updated example data for regression with multiple features
X = np.array([[1, 2], [2, 4], [3, 1], [4, 3], [5, 5]])  # Two features
y = np.array([1, 2, 2, 3, 4])                           # Target values

# Initialize the MultipleLinearRegression model and train it
model = MultipleLinearRegressor()
model.train(X, y)

# Create an instance of RegressionPlotter and plot for multiple features
plotter = RegressionPlotter(model, X, y)
plotter.plot(plot_all_features=True)  # This will generate a plot for each feature
