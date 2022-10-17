# -*- coding: utf-8 -*-
# ---------------------------
# Created By  : Rui Gomes
# Created Date: 10-10-2022
# ---------------------------

import sys
sys.path.insert(0, 'src/si')
# print(sys.path)

import numpy as np

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """It computes the euclidean distance of a point(x) to a set of points y.

    Args:
        x (np.ndarray): Point.
        y (np.ndarray): Set of points

    Returns:
        np.ndarray: Euclidean distance for each point in y.
    """
    distance = np.sqrt(((x - y) ** 2).sum(axis=1))
    return distance