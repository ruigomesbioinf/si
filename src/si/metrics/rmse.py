# -*- coding: utf-8 -*-
# ---------------------------
# Created By  : Rui Gomes
# Created Date: 18-10-2022
# ---------------------------

import sys
import numpy as np
sys.path.insert(0, 'src/si')
# print(sys.path)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It calculates the Root Mean Squared Error metric

    Args:
        y_true (np.ndarray): Real values
        y_pred (np.ndarray): Predicted values

    Returns:
        float: RMSE between real and predicted values
    """
    return np.sqrt(((y_true - y_pred) ** 2).mean())