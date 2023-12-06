from multiple_linear_regression import MultipleLinearRegression
from model_saver import ModelSaver
import numpy as np
import json

model = MultipleLinearRegression()

# Sample data for training
X = np.array([[1, 4], [2, 5], [3, 2], [4, 3], [5, 1]])
y = np.array([0, 1, 2, 3, 4])

# Train the model
model.train(X, y)

# Now save the model parameters
saver = ModelSaver(format_type='json')
saver.save_parameters(model, 'saved_model.json')

