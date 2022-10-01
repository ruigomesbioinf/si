# -*- coding: utf-8 -*-
# ---------------------------
# Created By  : Rui Gomes
# Created Date: 26-09-2022
# ---------------------------

import sys
sys.path.insert(0, '/home/rui/Desktop/SIB/si/src/si')

from typing import Optional, Union
import pandas as pd
import numpy as np

from data.dataset import Dataset

def read_csv(filename:str, sep:str = ",", features: Optional[bool] = True, label: Union[None, int]= None):
    """Function that reads csv file and returns a Dataset object of that file.

    Args:
        filename (str): name/path of file
        sep (str): separator between values. Defaults to , .
        features (Optional[bool], optional): If the csv file has feature names. Defaults to True.
        label (int): If the dataset has defined labels receives an integer value that tells the column of the labels. Defaults to None.
    """
    imported_data = pd.read_csv(filepath_or_buffer=filename, sep=sep)
    data = imported_data.values.tolist()
    headers = list(imported_data.columns)
    header_label = headers[label]
    
    if features:
        if label is not None:
            del headers[label]
    else:
        headers = None
    
    if label is not None:
        y = list(imported_data.iloc[:, label])
        imported_data = imported_data.drop(imported_data.columns[label], axis=1)
        data = imported_data.values.tolist()
    else: y = None
    
    return Dataset(X=data, y=y, features = headers, label = header_label)
        
    
def write_csv(dataset, filename: str, sep: str = ",", features: Optional[bool] = True, label: Optional[bool] = True):
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
        
    csv.to_csv(filename, sep = sep, index=False)
    
    
if __name__ == "__main__":
    file = r"/home/rui/Desktop/SIB/si/datasets/iris.csv"
    a = read_csv(filename=file, sep = ",", features=True, label=4)
    # print(a.print_dataframe())
    # print(a.summary())
    write_csv(a, "csv_write_test1.csv", features=True, label=False)
