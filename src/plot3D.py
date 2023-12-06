import numpy as np

# Assume MultipleLinearRegression is your custom regression class
# from multiple_linear_regression.py
from multiple_linear_regression import MultipleLinearRegressor

# Import RegressionPlotter from regression_plotter.py
from regression_plotter import RegressionPlotter

# Example data for 3D regression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3, 3]])  # Two features
y = np.array([2, 3, 3, 5, 5])                           # Target values

# Initialize the MultipleLinearRegression model and train it
model = MultipleLinearRegressor()
model.train(X, y)

# Create an instance of RegressionPlotter and plot 3D regression
plotter = RegressionPlotter(model, X, y)
plotter.plot()  # This will generate a 3D plot as we have two features


