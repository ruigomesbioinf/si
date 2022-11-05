# -*- coding: utf-8 -*-
# ---------------------------
# Created By  : Rui Gomes
# Created Date: 18-10-2022
# ---------------------------

import sys
from typing import Callable
from statistic.euclidean_distance import euclidean_distance
from data.dataset import Dataset
from metrics.accuracy import accuracy
import numpy as np

sys.path.insert(0, 'src/si')
# print(sys.path)


class KNNClassifier:
    
    def __init__(self, k: int, distance: Callable = euclidean_distance) -> None:
        """
        This algorithm predicts the class for a sample using the k most similar examples.

        Args:
            k (int): number of examples to consider
            distance (Callable, optional): euclidean distance function. Defaults to euclidean_distance.
        """
        self.k = k
        self.distance = distance
        self.dataset = None
        
    def fit(self, dataset: Dataset) -> "KNNClassifier":
        self.dataset = dataset # dataset de treino para o modelo
        return self
    
    def _get_closest_label(self, sample):
        # compute the distance between the sample and the dataset
        distances = self.distance(sample, self.dataset.X)

        # get the k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # get the labels of the k nearest neighbors
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]

        # get the most common label
        labels, counts = np.unique(k_nearest_neighbors_labels, return_counts=True)
        return labels[np.argmax(counts)]
    
    def predict(self, dataset: Dataset):
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)        
    
    def score(self, dataset: Dataset) -> float:
        prediction = self.predict(dataset)
        return accuracy(dataset.y, prediction)