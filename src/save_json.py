from multiple_linear_regression import MultipleLinearRegression
from model_saver import ModelSaver
import json

model = MultipleLinearRegression
# Create an instance of ModelSaver
saver = ModelSaver(format_type='json')

# Save the model parameters
saver.save_parameters(model, 'saved_model.json')

# Create a new instance of SimpleModel
new_model = MultipleLinearRegression()

# Load the model parameters to the new model
saver.load_parameters(new_model, 'saved_model.json')

# Check if the parameters are successfully loaded
print("Original Model Parameters -- Intercept:", model.intercept, "Coefficients:", model.coefficients)
print("Loaded Model Parameters -- Intercept:", new_model.intercept, "Coefficients:", new_model.coefficients)

