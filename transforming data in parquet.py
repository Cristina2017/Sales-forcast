# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:21:51 2023

@author: crist
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src import DATADIR, DATARAW, ROOTDIR

# #data = pd.read_csv('C:\\Users\\crist\\Rossmann\\data\\raw\\train.csv', low_memory=False)

# We are goint to read the files and to merge it to create an unique one. 

data = pd.read_csv(DATARAW/'train.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, low_memory=False)     
store = pd.read_csv(DATARAW/'store.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, low_memory=False)    


def merging_data(data, store):
    df = data.merge(store, on='Store', how='right')
    #df = df.drop(['StoreType_x', 'Assortment_x',
     #   'CompetitionDistance_x', 'CompetitionOpenSinceMonth_x',
     #   'CompetitionOpenSinceYear_x', 'Promo2_x', 'Promo2SinceWeek_x',
     #   'Promo2SinceYear_x', 'PromoInterval_x'], axis = 1)

    c = [column for column in df.columns]

    c = [column[:-2] if "_y" in column else column for column in c]

    df.columns = c
    df['year'] = df['Date'].apply(lambda x: x[:4]).astype('int')
    df['month'] = df['Date'].apply(lambda x: x[5:7]).astype('int')
    return df

if __name__ == "__main__":
   

    df = merging_data(data, store)
    df.to_parquet(DATADIR / "df.parquet")

  