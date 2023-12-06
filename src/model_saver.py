import json
import pickle

class ModelSaver:
    def __init__(self, format_type='json'):
        self.format_type = format_type.lower()
        if self.format_type not in ['json', 'pickle']:
            raise ValueError("Unsupported format. Please choose 'json' or 'pickle'.")

    def save_model(self, MultipleLinearRegressor, multiple.json):
        """
        Save the parameters of a generic model.

        Parameters:
        - model: The generic machine learning model.
        - filename (str): The filename to save the model parameters.
        """
        if self.format_type == 'json':
            self._save_json(MultipleLinearRegressor, filename)
        elif self.format_type == 'pickle':
            self._save_pickle(MultipleLinearRegressor, filename)

    def load_model(self, MultipleLinearRegressor, filename):
        """
        Load the parameters from a saved file and set them on the given model.

        Parameters:
        - model: The generic machine learning model.
        - filename (str): The filename from which to load the model parameters.
        """
        if self.format_type == 'json':
            self._load_json(MultipleLinearRegressor, filename)
        elif self.format_type == 'pickle':
            self._load_pickle(MultipleLinearRegressor, filename)

    def _save_json(self, MultipleLinearRegressor, multiple.json):
        """
        Save model parameters in JSON format.

        Parameters:
        - model: The generic machine learning model.
        - filename (str): The filename to save the model parameters.
        """
        params = MultipleLinearRegressor.get_parameters()  # Replace with the method to get parameters from the model
        with open(multiple.json, 'w') as file:
            json.dump(params, file)

    def _load_json(self, MultipleLinearRegressor, multiple.json):
        """
        Load model parameters from a JSON file and set them on the given model.

        Parameters:
        - model: The generic machine learning model.
        - filename (str): The filename from which to load the model parameters.
        """
        with open(multiple.json, 'r') as file:
            params = json.load(file)
        MultipleLinearRegressor.set_parameters(params)  # Replace with the method to set parameters on the model

    def _save_pickle(self, MultipleLinearRegressor, multiple.pkl):
    
        params = MultipleLinearRegressor.get_parameters()  # Replace with the method to get parameters from the model
        with open(multiple.pkl, 'wb') as file:
            pickle.dump(params, file)

    def _load_pickle(self, MultipleLinearRegressor, multiple.pkl):
        """
        Load model parameters from a pickle file and set them on the given model.

        Parameters:
        - model: The generic machine learning model.
        - filename (str): The filename from which to load the model parameters.
        """
        with open(multiple.pkl, 'rb') as file:
            params = pickle.load(file)
        MultipleLinearRegressor.set_parameters(params)  # Replace with the method to set parameters on the model
