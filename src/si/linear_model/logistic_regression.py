# -*- coding: utf-8 -*-
# ---------------------------
# Created By  : Rui Gomes
# Created Date: 28-10-2022
# ---------------------------

import sys
sys.path.insert(0, 'src/si')
# print(sys.path)
import numpy as np
import matplotlib.pyplot as plt
from data.dataset import Dataset
from statistic.sigmoid_function import sigmoid_function
from metrics.accuracy import accuracy



class LogisticRegression:
    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000):
        """
        Logistic Regression is a linear model that uses L2 regularization and a sigmoid function.
        This model uses the linear regression problem using an adapted Gradient Descent algorithm.

        Args:
            l2_penalty (float, optional): The L2 regularization parameter. Defaults to 1.
            alpha (float, optional): The learning rate. Defaults to 0.001.
            max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        """
        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        
        # attributes
        self.theta = None
        self.theta_zero = None
        self.cost_history = {}
        
    def fit(self, dataset: Dataset) -> "LogisticRegression":
        """
        Fits the model to the dataset

        Args:
            dataset (Dataset): Dataset object.
        Returns:
            LogisticRegression: The fitted model.
        """
        m, n = dataset.get_shape()
        
        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0
        
        threshold = 0.0001
        
        # gradient descent
        for i in range(int(self.max_iter)):
            # computes the cost history and updates it with iteration and cost
            self.cost_history[i] = self.cost(dataset=dataset)
            
            if i > 1 and (self.cost_history[i - 1] - self.cost_history[i] < threshold):
                break
            else:
                # predicted y
                y_pred = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
                
                # computes the gradient with the learning rate
                gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)
                
                # computes the l2 regularization penalty
                penalization_term = self.alpha * (self.l2_penalty / m) * self.theta
                
                # updates the model parameters theta and theta_zero
                self.theta = self.theta - gradient - penalization_term
                self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)
                
        return self
    
    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output dataset

        Args:
            dataset (Dataset): Dataset object.

        Returns:
            np.array: The predictions of the dataset.
        """
        y_pred = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        y_pred = [1 if x >= 0.5 else 0 for x in y_pred]
        return y_pred
    
    def score(self, dataset: Dataset) -> float:
        """
        Computes the accuracy of the model on the dataset.

        Args:
            dataset (Dataset): Dataset object.

        Returns:
            float: The accuracy value.
        """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)
    
    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using L2 regularization
        
        Args:
            dataset (Dataset): Dataset object.

        Returns:
            float: The cost function value.
        """
        m, n = dataset.get_shape()
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        cost = (- dataset.y * np.log(predictions)) - ((1 - dataset.y) * np.log(1 - predictions))
        cost = np.sum(cost) / n
        
        # regularization term
        cost = cost + (self.l2_penalty * np.sum(self.theta ** 2) / (2 * n))
        
        return cost
    
    def plot_cost_history(self):
        """
        Plots the cost history using matplotlib with x axis as number of iterations and y axis as cost value.
        """
        plt.plot(self.cost_history.keys(), self.cost_history.values())
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()