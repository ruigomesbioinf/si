# -*- coding: utf-8 -*-
# ---------------------------
# Created By  : Rui Gomes
# Created Date: 24-10-2022
# ---------------------------

import sys
import numpy as np
sys.path.insert(0, 'src/si')
# print(sys.path)

def sigmoid_function(x: np.ndarray) -> float:
    """
    Used for predicting probabilities since the probability is always between 0 and 1.

    Args:
        x (_type_): Entry values

    Returns:
        float: probability of values being equal to 1 (sigmoid function)
    """
    return 1 / (1 + np.exp(-x))