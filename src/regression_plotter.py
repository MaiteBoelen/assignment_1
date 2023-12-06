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
        elif num_features == 2 and not plot_all_features:
            self._plot_3d_regression()
        else:  # This will handle more than two features or when plot_all_features is True
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
        for i in range(num_features):
            plt.figure()
            plt.scatter(self.X[:, i], self.y, color='blue', label=f'Data Feature {i+1}')
            
            # Generate a sequence of values for the current feature
            line_x = np.linspace(self.X[:, i].min(), self.X[:, i].max(), 100)
            
            # Prepare a feature array for predictions. Use the mean values for other features
            X_temp = np.ones((100, num_features)) * np.mean(self.X, axis=0)
            X_temp[:, i] = line_x
            
            # Make predictions
            line_y = self.model.predict(X_temp)
            
            plt.plot(line_x, line_y, color='red', label='Regression Line')
            plt.xlabel(f'Feature {i+1}')
            plt.ylabel('Target')
            plt.title(f'Regression Line for Feature {i+1}')
            plt.legend()
            plt.show()

