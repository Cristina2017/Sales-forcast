# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 14:01:30 2023

@author: crist
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src import DATADIR, DATARAW, ROOTDIR

# #data = pd.read_csv('C:\\Users\\crist\\Rossmann\\data\\raw\\train.csv', low_memory=False)
data = pd.read_csv(DATARAW/'train.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, low_memory=False)        
#                    #encoding='latin-1')
# #test= pd.read_csv('C:\\Users\\crist\\Rossmann\\data\\raw\\test.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, low_memory=False)   

# store = pd.read_csv('C:\\Users\\crist\\Rossmann\\data\\raw\\store.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, low_memory=False)      

# df=data.merge(store, on='Store', how='right')
# df = df.drop(['StoreType_x', 'Assortment_x',
#        'CompetitionDistance_x', 'CompetitionOpenSinceMonth_x',
#        'CompetitionOpenSinceYear_x', 'Promo2_x', 'Promo2SinceWeek_x',
#        'Promo2SinceYear_x', 'PromoInterval_x'], axis = 1)

# c = [column for column in df.columns]

# c = [column[:-2] if "_y" in column else column for column in c]

# df.columns = c

df =  pd.read_parquet(DATADIR/'df.parquet', engine='auto')

years = [2013, 2014, 2015]
df['year'] = df['Date'].apply(lambda x: x[:4]).astype('int')
# for year in years:
#     globals()[year] = df[df['year']==year]. groupby('DayOfWeek').agg({"Sales":"mean"}).reset_index()