import json
import pickle
from multiple_linear_regression import MultipleLinearRegression


class ModelSaver:
    def __init__(self, format_type='json'):
        self.format_type = format_type.lower()

        if self.format_type not in ['json', 'pickle']:
            raise ValueError("Unsupported format. Choose from 'json' or 'pickle'.")

    def save_parameters(self, model, file_path):
        """
        Save the parameters of a generic model to a file.

        Parameters:
        - model: The generic machine learning model.
        - file_path: The file path where parameters will be saved.
        """
        if self.format_type == 'json':
            self._save_json(model, file_path)
        elif self.format_type == 'pickle':
            self._save_pickle(model, file_path)
        else:
            raise ValueError("Unsupported format.")

    def load_parameters(self, model, file_path):
        """
        Load parameters from a file and set them on the given model.

        Parameters:
        - model: The generic machine learning model.
        - file_path: The file path from which parameters will be loaded.
        """
        if self.format_type == 'json':
            self._load_json(model, file_path)
        elif self.format_type == 'pickle':
            self._load_pickle(model, file_path)
        else:
            raise ValueError("Unsupported format.")

    def _save_json(self, model, file_path):
        parameters = model.get_parameters()  # Assume the model has a method to get parameters
        with open(file_path, 'w') as json_file:
            json.dump(parameters, json_file)

    def _load_json(self, model, file_path):
        with open(file_path, 'r') as json_file:
            parameters = json.load(json_file)
            model.set_parameters(parameters)  # Assume the model has a method to set parameters

    def _save_pickle(self, model, file_path):
        parameters = model.get_parameters()  # Assume the model has a method to get parameters
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(parameters, pickle_file)

    def _load_pickle(self, model, file_path):
        with open(file_path, 'rb') as pickle_file:
            parameters = pickle.load(pickle_file)
            model.set_parameters(parameters)  # Assume the model has a method to set parameters
