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


def Sales_mean(df):
    """
    We will add to our dataframe a column with the mean of the monthly sales

    """
    sales = df.groupby(["Store", "year", "month"]).agg({"Sales": "mean"}).reset_index()
    d = sales.groupby(["Store", "year", "month"])["Sales"].apply(list).to_dict()
    df["mean_sales"] = df[["Store", "year", "month"]].apply(
        lambda x: d[(x["Store"], x["year"], x["month"])][0]
        if (x["Store"], x["year"], x["month"]) in d
        else 0,
        axis=1,
    )
    return df


def Sales_mean_next_month(df):
    
    """
    We have seen that there is a relation between the distance of the
    competitor and the dicrease of sales next month
    we create a feature with the sales next month

    """
    sales = df.groupby(["Store", "year", "month"]).agg({"Sales": "mean"}).reset_index()
    # Sales2 = sales['Sales'].values
    Sales2 = sales["Sales"].values
    Sales2 = np.insert(Sales2, 0, 0)
    Sales2 = Sales2[:-1]
    sales["Sales2"] = Sales2
    d = sales.groupby(["Store", "year", "month"])["Sales2"].apply(list).to_dict()
    df["mean_nextMonth"] = df[["Store", "year", "month"]].apply(
        lambda x: d[(x["Store"], x["year"], x["month"])][0]
        if (x["Store"], x["year"], x["month"]) in d
        else 0,
        axis=1
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
    sales.columns = ["Store", "year", "month", "DayOfWeek", "Sales_mean", "Sales_std"]
    d = (
        sales.groupby(["Store", "year", "month", "DayOfWeek"])["Sales_mean"]
        .apply(list)
        .to_dict()
    )
    d1 = (
        sales.groupby(["Store", "year", "month", "DayOfWeek"])["Sales_std"]
        .apply(list)
        .to_dict()
    df["DayWeekMonthMean"] = df[["Store", "year", "month", "DayOfWeek"]].apply(
        lambda x: d[(x["Store"], x["year"], x["month"], x["DayOfWeek"])][0]
        if (x["Store"], x["year"], x["month"], x["DayOfWeek"]) in d
        else 0,
        axis=1,
    )
    df["DayWeekMonthstd"] = df[["Store", "year", "month", "DayOfWeek"]].apply(
        lambda x: d[(x["Store"], x["year"], x["month"], x["DayOfWeek"])][0]
        if (x["Store"], x["year"], x["month"], x["DayOfWeek"]) in d
        else 0,
        axis=1,
    )
    return df

if __name__ == "__main__":

    df = pd.read_parquet(DATADIR / "df.parquet", engine="auto").reset_index()
    
    steps = [Sales_mean, Sales_mean_next_month, Sales_monthly_dayWeek]
    
    for step in steps:
        df = step(df)
        df.to_parquet(DATADIR / (step.__name__ + ".parquet"))
