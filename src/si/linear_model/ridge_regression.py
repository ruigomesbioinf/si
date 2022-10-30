import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src/si')
# print(sys.path)

from data.dataset import Dataset
from metrics.mse import mse


class RidgeRegression:
    """
    The RidgeRegression is a linear model using the L2 regularization.
    This model solves the linear regression problem using an adapted Gradient Descent technique

    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    alpha: float
        The learning rate
    max_iter: int
        The maximum number of iterations

    Attributes
    ----------
    theta: np.array
        The model parameters, namely the coefficients of the linear model.
        For example, x0 * theta[0] + x1 * theta[1] + ...
    theta_zero: float
        The model parameter, namely the intercept of the linear model.
        For example, theta_zero * 1
    """
    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000):
        """

        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter
        alpha: float
            The learning rate
        max_iter: int
            The maximum number of iterations
        """
        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter

        # attributes
        self.theta = None
        self.theta_zero = None
        self.cost_history = {}

    def fit(self, dataset: Dataset) -> 'RidgeRegression':
        """
        Fit the model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: RidgeRegression
            The fitted model
        """
        m, n = dataset.get_shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0
        
        
        threshold = 1
        # gradient descent
        for i in range(self.max_iter):
            # computes cost and updates cost_history
            self.cost_history[i] = self.cost(dataset=dataset)
                
            if i > 1 and (self.cost_history[i - 1] - self.cost_history[i] < threshold):
                break
            else:
                # predicted y
                y_pred = np.dot(dataset.X, self.theta) + self.theta_zero

                # computing and updating the gradient with the learning rate
                gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)

                # computing the penalty
                penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

                # updating the model parameters
                self.theta = self.theta - gradient - penalization_term
                self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)
        return self

    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of

        Returns
        -------
        predictions: np.array
            The predictions of the dataset
        """
        return np.dot(dataset.X, self.theta) + self.theta_zero

    def score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Square Error of the model on the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on

        Returns
        -------
        mse: float
            The Mean Square Error of the model
        """
        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using L2 regularization

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost function on

        Returns
        -------
        cost: float
            The cost function of the model
        """
        y_pred = self.predict(dataset)
        return (np.sum((y_pred - dataset.y) ** 2) + (self.l2_penalty * np.sum(self.theta ** 2))) / (2 * len(dataset.y))

    def plot_cost_history(self):
        """
        Plots the cost history using matplotlib with x axis as number of iterations and y axis as cost value.
        """
        plt.plot(self.cost_history.keys(), self.cost_history.values(), "-k")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()        

