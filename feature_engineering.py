# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:48:31 2023

@author: crist
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src import DATADIR, DATARAW, ROOTDIR

df = pd.read_parquet(DATADIR / "df.parquet", engine="auto")

years = [2013, 2014, 2015]


def Sales_month_mean(df):
    """
    We will add to our dataframe a column with the mean of the monthly sales

    """
    sales = df.groupby(["Store", "year", "month"]).agg({"Sales": "mean"}).reset_index()
    d = sales.groupby(["Store", "year", "month"])["Sales"].apply(list).to_dict()
    df["mean_monthly_sales"] = df[["Store", "year", "month"]].apply(
        lambda x: d[(x["Store"], x["year"], x["month"])][0]
        if (x["Store"], x["year"], x["month"]) in d
        else 0,
        axis=1,
    )
    return df

def std_month_mean(df):
    
    """
    We will add to our dataframe a column with the mean of the monthly sales

    """
    sales = df.groupby(["Store", "year", "month"]).agg({"Sales": "std"}).reset_index()
    d = sales.groupby(["Store", "year", "month"])["Sales"].apply(list).to_dict()
    df["std_montly_sales"] = df[["Store", "year", "month"]].apply(
        lambda x: d[(x["Store"], x["year"], x["month"])][0]
        if (x["Store"], x["year"], x["month"]) in d
        else 0,
        axis=1,
    )
    return df


def Sales_monthly_dayWeek(df):

    """
    We know that there are some days in the week where the sales
    are higher than others. Hence, we will create a feature with
    a lag of a month for each day of the week
    """
    sales = (
        df.groupby(["Store", "year", "month", "DayOfWeek"])
        .agg({"Sales": ["mean", "std"]})
        .reset_index()
    )
    sales.columns = ["Store", "year", "month", "DayOfWeek", "Sales_mean_monthly", "Sales_std_monthly"]
    d = (
        sales.groupby(["Store", "year", "month", "DayOfWeek"])["Sales_mean_monthly"]
        .apply(list)
        .to_dict()
    )
    d1 = (
        sales.groupby(["Store", "year", "month", "DayOfWeek"])["Sales_std_monthly"]
        .apply(list)
        .to_dict()
    )
    df["DayWeekMonthMean"] = df[["Store", "year", "month", "DayOfWeek"]].apply(
        lambda x: d[(x["Store"], x["year"], x["month"], x["DayOfWeek"])][0]
        if (x["Store"], x["year"], x["month"], x["DayOfWeek"]) in d
        else 0,
        axis=1,
    )
    df["DayWeekMonthstd"] = df[["Store", "year", "month", "DayOfWeek"]].apply(
        lambda x: d[(x["Store"], x["year"], x["month"], x["DayOfWeek"])][0]
        if (x["Store"], x["year"], x["month"], x["DayOfWeek"]) in d1
        else 0,
        axis=1,
    )
    return df


def Sales_bimonthly_mean(df):
    """
    We will add to our dataframe a column with the mean of the monthly sales

    """
    def func(x):
        if x ==1 or x == 2:
            sol= 1
        elif x == 3 or x==4:
            sol = 2
        elif x == 5 or x==6:
            sol=3
        elif x==7 or x==8:
            sol=4
        elif x==9 or x==10:
            sol=5
        elif x==11 or x==12:
            sol = 6
        return sol

    df['bimonth']=df['month'].apply(func)
    
    sales = df.groupby(["Store", "year", "bimonth"]).agg({"Sales": "mean"}).reset_index()
    d = sales.groupby(["Store", "year", "bimonth"])["Sales"].apply(list).to_dict()
    df["mean_bimonthly_sales"] = df[["Store", "year", "bimonth"]].apply(
        lambda x: d[(x["Store"], x["year"], x["bimonth"])][0]
        if (x["Store"], x["year"], x["bimonth"]) in d
        else 0,
        axis=1,
    )
    return df

def std_bimonthly_mean(df):
    
    """
    We will add to our dataframe a column with the mean of the monthly sales

    """
    def func(x):
        if x ==1 or x == 2:
            sol= 1
        elif x == 3 or x==4:
            sol = 2
        elif x == 5 or x==6:
            sol=3
        elif x==7 or x==8:
            sol=4
        elif x==9 or x==10:
            sol=5
        elif x==11 or x==12:
            sol = 6
        return sol

    df['bimonth']=df['month'].apply(func)
    
    sales = df.groupby(["Store", "year", "bimonth"]).agg({"Sales": "std"}).reset_index()
    d = sales.groupby(["Store", "year", "bimonth"])["Sales"].apply(list).to_dict()
    df["std_bimontly_sales"] = df[["Store", "year", "bimonth"]].apply(
        lambda x: d[(x["Store"], x["year"], x["bimonth"])][0]
        if (x["Store"], x["year"], x["bimonth"]) in d
        else 0,
        axis=1,
    )
    return df

def Sales_biMonthly_dayWeek(df):

    """
    We know that there are some days in the week where the sales
    are higher than others. Hence, we will create a feature with
    a lag of two months for each day of the week
    """
    
    def func(x):
        if x ==1 or x == 2:
            sol= 1
        elif x == 3 or x==4:
            sol = 2
        elif x == 5 or x==6:
            sol=3
        elif x==7 or x==8:
            sol=4
        elif x==9 or x==10:
            sol=5
        elif x==11 or x==12:
            sol = 6
        return sol

    df['bimonth']=df['month'].apply(func)
    sales = (
        df.groupby(["Store", "year", "bimonth", "DayOfWeek"])
        .agg({"Sales": ["mean", "std"]})
        .reset_index()
    )
    sales.columns = ["Store", "year", "bimonth", "DayOfWeek", "Sales_mean_bimonthly", "Sales_std_bimonthly"]
    d = (
        sales.groupby(["Store", "year", "bimonth", "DayOfWeek"])["Sales_mean_bimonthly"]
        .apply(list)
        .to_dict()
    )
    d1 = (
        sales.groupby(["Store", "year", "bimonth", "DayOfWeek"])["Sales_std_bimonthly"]
        .apply(list)
        .to_dict()
    )
    df["DayWeekBiMonthlyMean"] = df[["Store", "year", "bimonth", "DayOfWeek"]].apply(
        lambda x: d[(x["Store"], x["year"], x["bimonth"], x["DayOfWeek"])][0]
        if (x["Store"], x["year"], x["bimonth"], x["DayOfWeek"]) in d
        else 0,
        axis=1,
    )
    df["DayWeekBiMonthystd"] = df[["Store", "year", "bimonth", "DayOfWeek"]].apply(
        lambda x: d[(x["Store"], x["year"], x["bimonth"], x["DayOfWeek"])][0]
        if (x["Store"], x["year"], x["bimonth"], x["DayOfWeek"]) in d1
        else 0,
        axis=1,
    )
    return df

def Sales_trimonthly_mean(df):
    
    """
    We will add to our dataframe a column with the mean of the monthly sales

    """
    def func(x):
        if x ==1 or x == 2 or x == 3:
            sol= 1
        elif x == 4 or x == 5 or x == 6:
            sol = 2
        elif x == 7 or x == 8 or x == 9:
            sol=3
        elif x == 10 or x == 11 or x == 12:
            sol=4
        return sol

    df['trimonth']=df['month'].apply(func)
    
    sales = df.groupby(["Store", "year", "trimonth"]).agg({"Sales": "mean"}).reset_index()
    d = sales.groupby(["Store", "year", "trimonth"])["Sales"].apply(list).to_dict()
    df["mean_trimonthly_sales"] = df[["Store", "year", "trimonth"]].apply(
        lambda x: d[(x["Store"], x["year"], x["trimonth"])][0]
        if (x["Store"], x["year"], x["trimonth"]) in d
        else 0,
        axis=1,
    )
    return df

def std_trimonthly_mean(df):
    
    """
    We will add to our dataframe a column with the mean of the monthly sales

    """
    def func(x):
        if x ==1 or x == 2 or x == 3:
            sol= 1
        elif x == 4 or x == 5 or x == 6:
            sol = 2
        elif x == 7 or x == 8 or x == 9:
            sol=3
        elif x == 10 or x == 11 or x == 12:
            sol=4
        return sol

    df['trimonth']=df['month'].apply(func)
    
    sales = df.groupby(["Store", "year", "trimonth"]).agg({"Sales": "std"}).reset_index()
    d = sales.groupby(["Store", "year", "trimonth"])["Sales"].apply(list).to_dict()
    df["std_trimontly_sales"] = df[["Store", "year", "trimonth"]].apply(
        lambda x: d[(x["Store"], x["year"], x["trimonth"])][0]
        if (x["Store"], x["year"], x["trimonth"]) in d
        else 0,
        axis=1,
    )
    return df

def Sales_triMonthly_dayWeek(df):

    """
    We know that there are some days in the week where the sales
    are higher than others. Hence, we will create a feature with
    a lag of two months for each day of the week
    """
    
    def func(x):
        if x ==1 or x == 2 or x == 3:
            sol= 1
        elif x == 4 or x == 5 or x == 6:
            sol = 2
        elif x == 7 or x == 8 or x == 9:
            sol=3
        elif x == 10 or x == 11 or x == 12:
            sol=4
        return sol

    df['trimonth']=df['month'].apply(func)
    sales = (
        df.groupby(["Store", "year", "trimonth", "DayOfWeek"])
        .agg({"Sales": ["mean", "std"]})
        .reset_index()
    )
    sales.columns = ["Store", "year", "trimonth", "DayOfWeek", "Sales_mean_trimonthly", "Sales_std_trimonthly"]
    d = (
        sales.groupby(["Store", "year", "trimonth", "DayOfWeek"])["Sales_mean_trimonthly"]
        .apply(list)
        .to_dict()
    )
    d1 = (
        sales.groupby(["Store", "year", "trimonth", "DayOfWeek"])["Sales_std_trimonthly"]
        .apply(list)
        .to_dict()
    )
    df["DayWeektriMonthlyMean"] = df[["Store", "year", "trimonth", "DayOfWeek"]].apply(
        lambda x: d[(x["Store"], x["year"], x["trimonth"], x["DayOfWeek"])][0]
        if (x["Store"], x["year"], x["trimonth"], x["DayOfWeek"]) in d
        else 0,
        axis=1,
    )
    df["DayWeektriMonthystd"] = df[["Store", "year", "trimonth", "DayOfWeek"]].apply(
        lambda x: d[(x["Store"], x["year"], x["trimonth"], x["DayOfWeek"])][0]
        if (x["Store"], x["year"], x["trimonth"], x["DayOfWeek"]) in d1
        else 0,
        axis=1,
    )
    return df

def Sales_quarter_mean(df):
    
    """
    We will add to our dataframe a column with the mean of the monthly sales

    """
    def func(x):
        if x ==1 or x == 2 or x == 3 or x == 4:
            sol= 1
        elif x == 5 or x == 6 or x == 7 or x == 8:
            sol = 2
        elif x == 9 or x == 10 or x == 11 or x == 12:
            sol=3
        return sol

    df['quarter']=df['month'].apply(func)
    
    sales = df.groupby(["Store", "year", "quarter"]).agg({"Sales": "mean"}).reset_index()
    d = sales.groupby(["Store", "year", "quarter"])["Sales"].apply(list).to_dict()
    df["mean_quarter_sales"] = df[["Store", "year", "quarter"]].apply(
        lambda x: d[(x["Store"], x["year"], x["quarter"])][0]
        if (x["Store"], x["year"], x["quarter"]) in d
        else 0,
        axis=1,
    )
    return df

def std_quarter_mean(df):
    
    """
    We will add to our dataframe a column with the mean of the monthly sales

    """
    def func(x):
        if x ==1 or x == 2 or x == 3 or x == 4:
            sol= 1
        elif x == 5 or x == 6 or x == 7 or x == 8:
            sol = 2
        elif x == 9 or x == 10 or x == 11 or x == 12:
            sol=3
        return sol

    df['quarter']=df['month'].apply(func)
    
    sales = df.groupby(["Store", "year", "quarter"]).agg({"Sales": "std"}).reset_index()
    d = sales.groupby(["Store", "year", "quarter"])["Sales"].apply(list).to_dict()
    df["std_quarter_sales"] = df[["Store", "year", "quarter"]].apply(
        lambda x: d[(x["Store"], x["year"], x["quarter"])][0]
        if (x["Store"], x["year"], x["quarter"]) in d
        else 0,
        axis=1,
    )
    return df

def Sales_quarter_dayWeek(df):

    """
    We know that there are some days in the week where the sales
    are higher than others. Hence, we will create a feature with
    a lag of two months for each day of the week
    """
    
    def func(x):
        if x ==1 or x == 2 or x == 3 or x == 4:
            sol= 1
        elif x == 5 or x == 6 or x == 7 or x == 8:
            sol = 2
        elif x == 9 or x == 10 or x == 11 or x == 12:
            sol=3
        return sol

    df['quarter']=df['month'].apply(func)
    sales = (
        df.groupby(["Store", "year", "quarter", "DayOfWeek"])
        .agg({"Sales": ["mean", "std"]})
        .reset_index()
    )
    sales.columns = ["Store", "year", "quarter", "DayOfWeek", "Sales_mean_quarter", "Sales_std_quarter"]
    d = (
        sales.groupby(["Store", "year", "quarter", "DayOfWeek"])["Sales_mean_quarter"]
        .apply(list)
        .to_dict()
    )
    d1 = (
        sales.groupby(["Store", "year", "quarter", "DayOfWeek"])["Sales_std_quarter"]
        .apply(list)
        .to_dict()
    )
    df["DayWeekquarterMean"] = df[["Store", "year", "quarter", "DayOfWeek"]].apply(
        lambda x: d[(x["Store"], x["year"], x["quarter"], x["DayOfWeek"])][0]
        if (x["Store"], x["year"], x["quarter"], x["DayOfWeek"]) in d
        else 0,
        axis=1,
    )
    df["DayWeekquarterstd"] = df[["Store", "year", "quarter", "DayOfWeek"]].apply(
        lambda x: d[(x["Store"], x["year"], x["quarter"], x["DayOfWeek"])][0]
        if (x["Store"], x["year"], x["quarter"], x["DayOfWeek"]) in d1
        else 0,
        axis=1,
    )
    return df

if __name__ == "__main__":

    df = pd.read_parquet(DATADIR / "df.parquet", engine="auto").reset_index()
    
    steps = [Sales_month_mean, std_month_mean, Sales_monthly_dayWeek, Sales_bimonthly_mean, std_bimonthly_mean, Sales_biMonthly_dayWeek,
             Sales_trimonthly_mean, std_trimonthly_mean, Sales_triMonthly_dayWeek, Sales_quarter_mean, std_quarter_mean, Sales_quarter_dayWeek]
    
    for step in steps:
        df = step(df)
        df.to_parquet(DATADIR / (step.__name__ + ".parquet"))
        
        


# def Sales_mean_next_month(df):
    
#     """
#     We have seen that there is a relationship between the distance of the
#     competitor and the dicrease of sales next month
#     we create a feature with the sales next month

#     """
#     sales = df.groupby(["Store", "year", "month"]).agg({"Sales": "mean"}).reset_index()
#     # Sales2 = sales['Sales'].values
#     Sales2 = sales["Sales"].values
#     Sales2 = np.insert(Sales2, 0, 0)
#     Sales2 = Sales2[:-1]
#     sales["Sales2"] = Sales2
#     d = sales.groupby(["Store", "year", "month"])["Sales2"].apply(list).to_dict()
#     df["mean_nextMonth"] = df[["Store", "year", "month"]].apply(
#         lambda x: d[(x["Store"], x["year"], x["month"])][0]
#         if (x["Store"], x["year"], x["month"]) in d
#         else 0,
#         axis=1
#     )
#     return df

