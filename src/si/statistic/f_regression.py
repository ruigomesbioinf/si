# -*- coding: utf-8 -*-
# ---------------------------
# Created By  : Rui Gomes
# Created Date: 08-10-2022
# ---------------------------

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

from scipy import stats
from data.dataset import Dataset
from typing import Tuple, Union
import numpy as np

def f_regression(dataset: Dataset) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]:
    
    
    degrees_freedom = dataset.get_shape()[0] - 2
    correlation_coefficients = np.array([stats.pearsonr(dataset.X[:, i], dataset.y)[0] for i in range(dataset.get_shape()[1])])
    squared_correlation_coefficients = correlation_coefficients ** 2
    F = (squared_correlation_coefficients / (1 - squared_correlation_coefficients)) * degrees_freedom
    p = stats.f.sf(F, 1, degrees_freedom)
    return F, p
    
    
    
if __name__ == "__main__":
    a = Dataset(X=np.array([[1, 15, 3], [4, 5, 6]]), y=[1, 2], features=["A", "B", "C"], label="y")
    #print(f_regression(a))