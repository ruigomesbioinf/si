import sys
sys.path.insert(0, 'src/si')
# print(sys.path)


import numpy as np
from statistic.sigmoid_function import sigmoid_function

class Dense:
    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the dense layer.

        Args:
            input_size (int): Number of inputs the layer will receive.
            output_size (int): Number of outputs the layer will retrieve.
        """
        self.input_size = input_size
        self.output_size = output_size
        
        # attributes
        self.x = None
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        
    def forward(self, X: np.ndarray):
        """
        Computes the forward pass of our layer.
        
        Args:
            X (np.ndarray): The input to the layer.
        """
        self.x = X
        
        return np.dot(X, self.weights) + self.bias # our input_data needs to be a matrix with columns == features, to multiply these two matrixes the number of columns of input_data == number of rows of weights
    
    def backward(self, error: np.ndarray, learning_rate: float = 0.01):
        """
        Computes the backward pass of our layer.

        Args:
            error (np.ndarray): error values for the loss function
            learning_rate (float, optional): Learning rate. Defaults to 0.01.
        """
        
        error_to_propagate = np.dot(error, self.weights.T)
        
        # update weights and bias
        self.weights -= learning_rate * np.dot(self.x.T, error) # due to matrix multiplication rules x.T is used
        
        self.bias -= learning_rate * np.sum(error, axis = 0) # sum because bias has the dimensions of nodes and error has the dimensions of samples + nodes
        
        return error_to_propagate
    
    

        
    