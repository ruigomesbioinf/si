# -*- coding: utf-8 -*-
# ---------------------------
# Created By  : Rui Gomes
# Created Date: 28-10-2022
# ---------------------------

import sys
from typing import List
from data.dataset import Dataset
from metrics.accuracy import accuracy
import numpy as np


sys.path.insert(0, 'src/si')
# print(sys.path)


class VotingClassifier:
    def __init__(self, models: List):
        """Ensemble classifier that uses the majority of votes to predict the class of the labels.

        Args:
            models (List): A list of models for the ensemble
        """
        # parameters
        self.models = models
        
    def fit(self, dataset: Dataset) -> "VotingClassifier":
        """Fits the model according to the given training dataset

        Args:
            dataset (Dataset): A Dataset object

        Returns:
            VotingClassifier: The fitted model
        """
        for model in self.models:
            model.fit(dataset)
            
        return self
        
    def predict(self, dataset: Dataset):
        """PRedict class labels for samples in dataset

        Args:
            dataset (Dataset): Dataset object
        """
        
        def _get_majority_vote(pred: np.ndarray) -> int:
            """Auxiliary function that returns the majority vote of the given predictions

            Args:
                pred (np.ndarray): The predictions to get the majority vote from 
            """
            # get the most common vote
            labels, counts = np.unique(pred, return_counts=True)
            return labels[np.argmax(counts)]
        
        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
        return np.apply_along_axis(_get_majority_vote, axis=1, arr=predictions)
        
    def score(self, dataset: Dataset) -> float:
        
        return accuracy(dataset.y, self.predict(dataset))
        
        
        