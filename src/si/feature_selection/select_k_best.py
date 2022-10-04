# -*- coding: utf-8 -*-
# ---------------------------
# Created By  : Rui Gomes
# Created Date: 04-10-2022
# ---------------------------

import sys
sys.path.insert(0, 'src/si')
# print(sys.path)


from typing import Callable
import numpy as np

# imports necessários do package
from data.dataset import Dataset
from si.statistics.f_classification import f_classification

class SelectKBest:
    
    def __init__(self, score_func: Callable = f_classification, k: int = 10) -> None:
        """Select features according to the k highest scores.
            The scores are computed using ANOVA F-values between label/feature.

        Args:
            score_func (Callable): Function that takes a dataset and returns the scores and p-values. Defaults to f_classification.
            k (int, optional): Number of top features to select. Defaults to 10.
        """
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None
        
        
    def fit(self, dataset: Dataset) -> "SelectKBest":
        """Fits SelectKBest to compute the F scores and p_values.

        Args:
            dataset (Dataset): A dataset object

        Returns:
            SelectKBest: returns self
        """
        self.F, self.p = self.score_func(dataset)
        return self
    
    def transform(self, dataset: Dataset) -> Dataset:
        """It transforms the dataset by selecting k highest score features.

        Args:
            dataset (Dataset): A dataset object

        Returns:
            Dataset: A dataset object with the k highest score features.
        """
        idxs = np.argsort(self.F)[-self.k:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)
    
    def fit_transform(self, dataset: Dataset) -> Dataset:
        """It fits SelectKBest and then transforms the dataset by selecting the k highest score features.

        Args:
            dataset (Dataset): A dataset object

        Returns:
            Dataset: A dataset object with the k highest score features.
        """
        self.fit(dataset)
        return self.transform(dataset)
    
    