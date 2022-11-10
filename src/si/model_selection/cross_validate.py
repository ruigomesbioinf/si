# -*- coding: utf-8 -*-
# ---------------------------
# Created By  : Rui Gomes
# Created Date: 28-10-2022
# ---------------------------

import sys
sys.path.insert(0, 'src/si')
# print(sys.path)

from data.dataset import Dataset
from model_selection.split import train_test_split
from typing import Callable, Dict, List
import numpy as np
import random



def cross_validate(model, dataset: Dataset, scoring: Callable = None, cv: int = 3, test_size: float = 0.2) -> Dict[str, List[float]]:
    
    
    scores = {
        "seed": [],
        "train": [],
        "test": []
    }
    
    for i in range(cv):
        # get random state seed
        seed = random.randint(0, 1000)
        scores["seed"].append(seed)
        
        # split dataset 
        train, test = train_test_split(dataset=dataset, test_size=test_size, random_state=seed)
        
        # fit the model to the train set
        model.fit(train)
        
        # score the model on the test dataset
        if scoring is None:
            
            # store the train score
            scores["train"].append(model.score(train))
            
            # store the test score
            scores["test"].append(model.score(test))
            
        else:
            y_train = train.y 
            y_test = test.y
            
            # store the train score
            scores["train"].append(scoring(y_train, model.predict(train)))
            
            # store the test score
            scores["test"].append(scoring(y_test, model.predict(test)))
            
    return scores