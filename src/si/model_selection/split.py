# -*- coding: utf-8 -*-
# ---------------------------
# Created By  : Rui Gomes
# Created Date: 17-10-2022
# ---------------------------

import sys
sys.path.insert(0, 'src/si')
# print(sys.path)

from data.dataset import Dataset
import numpy as np

def train_test_split (dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Random splits a dataset into a train and a test set.

    Args:
        dataset (Dataset): Dataset object
        test_size (float, optional): size of the test set. Defaults to 0.2.
        random_state (int, optional): seed to feed the random permutations. Defaults to 42.
    """
    np.random.seed(random_state)
    
    len_samples = dataset.get_shape()[0]
    len_test = int(test_size * len_samples)
    permutations = np.random.permutation(len_samples)
    test_split = permutations[:len_test]
    train_split = permutations[len_test:]
    train = Dataset(X=dataset.X[train_split], y=dataset.y[train_split], features=dataset.features, label=dataset.label)
    test = Dataset(X=dataset.X[test_split], y=dataset.y[test_split], features=dataset.features, label=dataset.label)
    return train, test
