# -*- coding: utf-8 -*-
# ---------------------------
# Created By  : Rui Gomes
# Created Date: 01-10-2022
# ---------------------------

import sys
from typing import Optional, Union
from matplotlib import dates
sys.path.insert(0, '/home/rui/Desktop/SIB/si/src/si')

import numpy as np

from data.dataset import Dataset

def read_data_file(filename: str, sep: str = ",", label: Union[None, int]= None):
    """Read a data file and returns a dataset object.

    Args:
        filename (str): File name of path.
        sep (str, optional): Separator between values. Defaults to ",".
        label (Union[None, int], optional): Where are the labels. Defaults to None.
    """
    
    if label is not None:
        data = np.genfromtxt(fname = filename, delimiter=sep, skip_header=1, usecols=range(0, label))
        y = np.genfromtxt(fname=filename, delimiter=sep, skip_header=1, usecols=label, encoding = None, dtype = None)
    else:
        data = np.genfromtxt(fname = filename, delimiter=sep, usecols=range(0, label))
        y = None
        
    return Dataset(X=data, y=y)
    
    
    
def write_data_file(dataset, filename: str, sep: str = ",", label: Union[None, int] = None):
    """Writes a dataset 

    Args:
        dataset (_type_): The dataset that will be written.
        filename (str): The filename or path for the file that will be written
        sep (str, optional): The separator between values. Defaults to ",".
        label (Union[None, int], optional): Where the defined labels are. Defaults to None.
    """
    
    if label is not None:
        dataset = np.append(dataset.X, dataset.y[:, None], axis = 1)
    else:
        dataset = dataset.X

    np.savetxt(fname = filename, X = dataset, delimiter=sep, fmt="%s") #fmt="%s" is there to save numbers with only one decimal number
        
if __name__ == "__main__":
    file = r"/home/rui/Desktop/SIB/si/datasets/iris.csv"
    a = read_data_file(file, label=4)
    write_data_file(a, "write_data_file1.csv", label=4)


