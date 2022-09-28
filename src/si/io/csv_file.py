# -*- coding: utf-8 -*-
# ---------------------------
# Created By  : Rui Gomes
# Created Date: 26-09-2022
# ---------------------------

from typing import Optional
import pandas as pd
import numpy as np

from ..data.dataset import Dataset

def read_csv(filename:str, sep:str, features: Optional[bool] = True, label: Optional[bool] = True):
    """Function that reads csv file and returns a Dataset object of that file.

    Args:
        filename (str): name/path of file
        sep (str): separator between values
        features (Optional[bool], optional): If the csv file has feature names. Defaults to True.
        label (Optional[bool], optional): If label names exist. Defaults to True.
    """
    df = pd.read_csv(filename, sep = sep)
    return Dataset(X=df.to_numpy(), y=np.arange(0, len(df)), features=list(df.columns), label = None)
    
def write_csv(filename:str, dataset, sep:str, features: bool, label: bool):
    pass
    
if __name__ == "__main__":
    file = r"/home/rui/Desktop/SIB/si/datasets/iris.csv"
    a = read_csv(file, sep=",")
    print(a.print_dataframe())