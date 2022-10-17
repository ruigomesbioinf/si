# -*- coding: utf-8 -*-
# ---------------------------
# Created By  : Rui Gomes
# Created Date: 10-10-2022
# ---------------------------

import sys
sys.path.insert(0, 'src/si')
# print(sys.path)

from typing import Callable
from statistic.euclidean_distance import euclidean_distance
from data.dataset import Dataset
import numpy as np

class KMeans:
    
    def __init__(self, k: int, max_iter: int = 1000, distance: Callable = euclidean_distance) -> None:
        
        # parameters
        self.k = k
        self.max_iter = max_iter
        self.distance = distance
        
        # attributes
        self.centroids = None
        self.labels = None
        
    def _init_centroids(self, dataset: Dataset):
        
        seeds = np.random.permutation(dataset.get_shape()[0])[:self.k]
        self.centroids = dataset.X[seeds]
    
    def _get_closest_centroid(self, x: np.ndarray) -> np.ndarray:
        """Get the closest centroid to each data point.

        Args:
            x (np.ndarray): A sample

        Returns:
            np.ndarray: The closest centroid to each data point.
        """
        distance = self.distance(x, self.centroids)
        closest_index = np.argmin(distance) #retorna o centroid que está mais proximo do x (o index do menor valor)
        return closest_index
    
    
    def _dar_nome(self, dataset: Dataset):
        
        
        difference = False # variable to check differences between 
        i = 0 # variable to check max_iter
        labels = np.zeros(dataset.get_shape()[0]) #initialize array with zeros for labels
        
        while not difference and i < self.max_iter:
            new_labels = np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.X)
            centroids =[]
            
            for j in range(self.k):
                centroid = np.mean(dataset.X[new_labels == j], axis = 0)
                centroids.append(centroid)
                
            self.centroids =np.array(centroids)
            
            # check if centroids had changed
            difference = np.any(labels != new_labels)
            labels = new_labels
            
            i += 1
        self.labels = labels
        return self
    
    def _get_distances(self, sample: np.ndarray):
        return self.distance(sample, self.centroids)
    
    
    def transform(self):
        
        centroid_distances = np.apply_along_axis(self._get_distances, axis = 1, arr = dataset.X)
        return centroid_distances
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        """It predicts the labels of the dataset.

        Args:
            dataset (Dataset): Dataset object

        Returns:
            np.ndarray: Labels of the dataset
        """

        return np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.X)
    
        
        
        
        
        
if __name__ == "__main__":
    a = KMeans(2, 10)
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    a.init_centroids(dataset=dataset)
    print(a.get_closest_centroid([[1, 2, 3, 4], [2, 3, 7, 9]]))