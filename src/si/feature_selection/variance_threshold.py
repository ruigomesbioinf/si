# -*- coding: utf-8 -*-
# ---------------------------
# Created By  : Rui Gomes
# Created Date: 03-10-2022
# ---------------------------

import sys
sys.path.insert(0, 'src/si')
# print(sys.path)

from data.dataset import Dataset
import numpy as np
import pandas as pd

class VarianceThreshold:
    
    def __init__(self, threshold: float = 0.0) -> None:
        """ The variance threshold represents a baseline approach for feature selection.
            It removes all features which variance doesn't meet a threshold given by the user.

        Args:
            threshold (float): Non negative threshold given by the user. Defaults to 0.
        """
        if threshold < 0:
            raise ValueError("Your threshold must be a non-negative integer.")
        self.threshold = threshold
        self.variance = None
        
    def fit(self, dataset: Dataset) -> "VarianceThreshold":
        """Estimates/calculate the variance of each feature in a dataset.

        Args:
            dataset (_type_): dataset object
        """
        X = dataset
        variance = X.get_variance()
        self.variance = variance
        return self
    
    def transform(self, dataset: Dataset) -> Dataset:
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
        
    def fit_transform(self, dataset: Dataset) -> None:
        """Pre-processing method to run the fit and transform methods automatically by the user.

        Args:
            dataset (Dataset): Dataset object.
        """
        self.fit(dataset)
        return self.transform(dataset)
        

if __name__ == '__main__':
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

    # a = VarianceThreshold()
    # a = a.fit(dataset)
    # dataset = a.transform(dataset)
    # print(dataset.features)

    b = VarianceThreshold()
    b = b.fit_transform(dataset)
    # print(b.print_dataframe())
    print(b.features)
    