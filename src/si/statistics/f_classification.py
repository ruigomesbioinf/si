# -*- coding: utf-8 -*-
# ---------------------------
# Created By  : Rui Gomes
# Created Date: 03-10-2022
# ---------------------------

import sys
sys.path.insert(0, 'src/si')
# print(sys.path)

from scipy import stats
from data.dataset import Dataset
from typing import Tuple, Union
import numpy as np

def f_classification(dataset: Dataset) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]:
    """The scoring function for classification. It groups samples by classes and computes ANOVA's for the sammples
    returning F and P values.   

    Args:
        dataset (Dataset): A dataset object 

    Returns:
        Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]: Tuple of np arrays with F scores and p-values
    """
    classes = dataset.get_classes()
    groups = [dataset.X[dataset.y == c] for c in classes]
    F, p = stats.f_oneway(*groups)
    return F, p

