# -*- coding: utf-8 -*-
# ---------------------------
# Created By  : Rui Gomes
# Created Date: 26-09-2022
# ---------------------------

import sys
sys.path.insert(0, 'src/si')
# print(sys.path)

from typing import Optional
import pandas as pd
import numpy as np

from data.dataset import Dataset

def read_csv(filename:str, sep:str = ",", features: Optional[bool] = True, label: Optional[bool] = False) -> Dataset:
    """Function that reads csv file and returns a Dataset object of that file.

    Args:
        filename (str): name/path of file
        sep (str): separator between values. Defaults to , .
        features (Optional[bool], optional): If the csv file has feature names. Defaults to True.
        label (int): If the dataset has defined labels. Defaults to False
    Returns:
        Dataset: The dataset object
    """
    data = pd.read_csv(filename, sep)
    if features and label: 
        y = data.iloc[:, -1] 
        label = data.columns[-1]
        data = data.iloc[:, :-1]
        features = data.columns

    elif features and label is False:  
        features = data.columns
        y = None

    elif features is False and label:  
        y = data.iloc[:, -1]
        label = data.columns[-1]
        data = data.iloc[:, :-1]

    else: 
        y = None

    return Dataset(data, y, features, label)

        
    
def write_csv(dataset: Dataset, filename: str, sep: str = ",", features: Optional[bool] = True, label: Optional[bool] = True) -> None:
    """Writes a csv file from a dataset object

    Args:
        dataset (_type_): Dataset to save on csv format
        filename (str): Name of the csv file that will be saved
        sep (str, optional): Separator of values. Defaults to ",".
        features (Optional[bool], optional): Boolean value that tells if the dataset object has feature names. Defaults to True.
        label (Optional[bool], optional): Boolean value that tells if the dataset object has label names Defaults to True.
    """
    csv = pd.DataFrame(data=dataset.X)
    
    if features:
        csv.columns = dataset.features
    
    if label:
        csv.insert(loc=0, column=dataset.label, value=dataset.y)
        # csv[dataset.label] = dataset.y
        
    csv.to_csv(filename, sep = sep, index=False)
    
    
if __name__ == "__main__":
    pass
    # TESTING
    # file = r"/home/rui/Desktop/SIB/si/datasets/iris.csv"
    # a = read_csv(filename=file, sep = ",", features=True, label=4)
    # print(a.print_dataframe())
    # print(a.summary())
    # write_csv(a, "csv_write_test1.csv", features=True, label=False)
    
    # TESTING MISSING VALUES METHODS ON DATASET CLASS
    # file = r"/home/rui/Desktop/SIB/si/datasets/iris_missing_data.csv"
    # a = read_csv(filename=file, sep = ",", features=True, label=4)
    # print(a.dropna())
    # print(a.fillna(100))
