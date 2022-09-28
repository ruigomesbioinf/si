# -*- coding: utf-8 -*-
# ---------------------------
# Created By  : Rui Gomes
# Created Date: 26-09-2022
# ---------------------------

import pandas as pd
import numpy as np

from ..data.dataset import Dataset

def read_csv(filename:str, sep:str, features:bool = True, label:bool = True):
    df = pd.read_csv(filename, sep = sep)
    return Dataset(X=df.to_numpy(), y=np.arange(0, len(df)), features=list(df.columns), label = None)
    
    
if __name__ == "__main__":
    file = r"/home/rui/Desktop/SIB/si/datasets/iris.csv"
    a = read_csv(file, sep=",")
    print(a.print_dataframe())