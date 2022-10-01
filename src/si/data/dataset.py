# -*- coding: utf-8 -*-
# ---------------------------
# Created By  : Rui Gomes
# Created Date: 26-09-2022
# ---------------------------

from typing import Tuple, List
import numpy as np
import pandas as pd


class Dataset:
    
    def __init__(self, X:np.ndarray = None, y:np.ndarray = None, features:List = None, label:str = None):
        """Dataset implementation from scratch"""
        
        self.X = X
        self.y = y
        self.features = features
        self.label = label
        
    def get_shape(self) -> Tuple:
        """Return dataset dimensions"""
        
        return self.X.shape
    
    def has_label(self):
        """Returns True if dataset has labels"""
        
        if self.y is not None:
            return True
        else:
            return False
    
    def get_classes(self) -> List[int]:
        """Return the label classes as a list"""
        
        if self.y is None:
            raise Exception("You've an unsupervised dataset")
        else:
            return np.unique(self.y)
        
    def get_mean(self):
        """Return mean of features"""
        return np.mean(self.X, axis=0)
    
    def get_variance(self):
        """Return variance of features"""
        return np.var(self.X, axis=0)
    
    def get_median(self):
        """Return median of features"""
        return np.median(self.X, axis=0)
    
    def get_min(self):
        """Return the minimum value of features"""
        return np.min(self.X, axis=0)
    
    def get_max(self):
        """Return the maximum value of features"""
        return np.max(self.X, axis=0)
    
    def summary(self):
        return pd.DataFrame(
            {"mean": self.get_mean(),
             "median": self.get_median(),
             "variance": self.get_variance(),
             "min": self.get_min(),
             "max": self.get_max()}
        )
        
    def print_dataframe(self):
        return pd.DataFrame(self.X, columns=self.features, index=self.y)
    
    
if __name__ == "__main__":
    x = np.array([[1, 2, 8], [1, 2, 7]])
    y = np.array([1, 2])
    features = ["A", "B", "C"]
    label = "y"
    a = Dataset(X=x, y=y, features=features, label=label)
    b = Dataset()
    c = Dataset(X=x, y=y, features=features, label=None)
    #print(a.get_shape())
    print(c.has_label())
    #print(a.get_classes())
    print(c.print_dataframe())