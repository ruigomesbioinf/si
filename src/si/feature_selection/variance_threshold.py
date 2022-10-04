# -*- coding: utf-8 -*-
# ---------------------------
# Created By  : Rui Gomes
# Created Date: 03-10-2022
# ---------------------------

import sys
sys.path.insert(0, 'src/si')
# print(sys.path)

import warnings
from data.dataset import Dataset
import numpy as np
import pandas as pd

class VarianceThreshold:
    
    def __init__(self, threshold: int = 0) -> None:
        """ The variance threshold represents a baseline approach for feature selection.
            It removes all features which variance doesn't meet a threshold given by the user.

        Args:
            threshold (int): Non negative threshold given by the user. Defaults to 0.
        """
        if threshold < 0:
            warnings.warn("Your threshold must be a non-negative integer.")
        self.threshold = threshold
        self.variance = None
        
    def fit(self, dataset):
        """Estimates/calculate the variance of each feature in a dataset.

        Args:
            dataset (_type_): dataset object
        """
        X = dataset
        variance = X.get_variance()
        self.variance = variance
        return self
    
    def transform(self, dataset):
        """Selects all the features with variance higher than the threshold and returns a new dataset 
        with the selected features

        Args:
            dataset (_type_): dataset object
        """
        X = dataset.X
        
        features_mask = self.variance > self.threshold
        X = X[:, features_mask]
        features = np.array(dataset.features)[features_mask]
        return Dataset(X=X, y=dataset.y, features=list(features), label=None)
        
        

if __name__ == "__main__":
    dataset = Dataset(X= np.array([[1, 4, 12], [1, 1, 1], [1, 1, 1]]),
                      y = np.array([1, 2, 3]),
                      features=["A", "B", "C"],
                      label="class")
    a = VarianceThreshold(0)
    a.fit(dataset=dataset)
    print(a.variance)
    print(a.threshold)
    b = a.transform(dataset)
    print(b.print_dataframe())
    