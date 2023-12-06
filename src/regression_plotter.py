import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class RegressionPlotter:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def plot(self, plot_all_features=False):
        num_features = self.X.shape[1]

        if num_features == 1:
            self._plot_2d_regression()
        elif num_features == 2:
            self._plot_3d_regression()
        elif plot_all_features:
            self._plot_all_features()

    def _plot_2d_regression(self):
        plt.scatter(self.X[:, 0], self.y, color='blue', label='Data')
        line_x = np.linspace(self.X[:, 0].min(), self.X[:, 0].max(), 100)
        line_y = self.model.predict(line_x.reshape(-1, 1))
        plt.plot(line_x, line_y, color='red', label='Regression Line')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.legend()
        plt.show()

    def _plot_3d_regression(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.X[:, 0], self.X[:, 1], self.y, color='blue', label='Data')

        # Create a meshgrid for the plane
        x0, x1 = np.meshgrid(np.linspace(self.X[:, 0].min(), self.X[:, 0].max(), 10),
                             np.linspace(self.X[:, 1].min(), self.X[:, 1].max(), 10))
        y_plane = self.model.predict(np.c_[x0.ravel(), x1.ravel()]).reshape(x0.shape)

        ax.plot_surface(x0, x1, y_plane, color='red', alpha=0.3)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Target')
        plt.show()

    def _plot_all_features(self):
        num_features = self.X.shape[1]
        fig, axs = plt.subplots(1, num_features, figsize=(num_features * 5, 4))

        for i in range(num_features):
            axs[i].scatter(self.X[:, i], self.y, color='blue', label=f'Feature {i+1} vs Target')
        
        # Create a grid for the feature being plotted
            line_x = np.linspace(self.X[:, i].min(), self.X[:, i].max(), 100).reshape(-1, 1)

        # Construct a full feature set with the other features set to their mean values
            full_feature_set = np.full((100, num_features), np.mean(self.X, axis=0))
            full_feature_set[:, i] = line_x.ravel()

        # Predict using the full feature set
            line_y = self.model.predict(full_feature_set)

            axs[i].plot(line_x, line_y, color='red', label='Regression Line')
            axs[i].set_xlabel(f'Feature {i+1}')
            axs[i].set_ylabel('Target')
            axs[i].legend()

        plt.tight_layout()
        plt.show()

