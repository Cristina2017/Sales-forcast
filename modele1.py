# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:02:42 2023

@author: crist
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src import DATADIR, DATARAW, ROOTDIR
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_parquet(DATADIR / "Sales_quarter_dayWeek.parquet", engine="auto")

def data_treatment(df):
    
    """ Convert columns into numeric ones
    
    """
    df = df.fillna(0)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%d%m%Y")
    df["StoreType"] = df["StoreType"].astype('category')
    df["StoreType"] = df["StoreType"].cat.codes
    df["Assortment"] = df["Assortment"].astype('category')
    df["Assortment"] = df["Assortment"].cat.codes
    df["StateHoliday"] = df["StateHoliday"].astype('category')
    df["StateHoliday"] = df["StateHoliday"].cat.codes
    df['day']=df['Date'].apply(lambda x: x[:2]).astype('int')
    return df

    
data = df.drop(['Sales', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'Date'], axis=1) # We eliminate this variables because they have a lot of missing values
target = df.Sales
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state= 101)


# lightGBM

# https://practicaldatascience.co.uk/machine-learning/how-to-tune-an-xgbregressor-model-with-optuna

import lightgbm as lgb
import optuna

def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        #'gamma': trial.suggest_float('gamma', 0.01, 1.0),
        'subsample': trial.suggest_float('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        'random_state': trial.suggest_int('random_state', 1, 1000)
    }
    model = lgb.LGBMRegressor(**param)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return mean_squared_error(y_test, y_pred)
    
study = optuna.create_study(direction='minimize', study_name='regression')
study.optimize(objective, n_trials=50)

# print('Best parameters', study.best_params)

model = lgb.LGBMRegressor(**study.best_params)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print('MSE: ', mean_squared_error(y_test, y_pred))
print('RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred)))

# import xgboost as xgb

# def objective(trial):
#     param = {
#         'max_depth': trial.suggest_int('max_depth', 1, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
#         'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         #'gamma': trial.suggest_float('gamma', 0.01, 1.0),
#         'subsample': trial.suggest_float('subsample', 0.01, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
#         'random_state': trial.suggest_int('random_state', 1, 1000)
#     }
#     model = xgb.XGBRegressor(**param)
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
#     return mean_squared_error(y_test, y_pred)
    
# study = optuna.create_study(direction='minimize', study_name='regression')
# study.optimize(objective, n_trials=30)

# print('Best parameters', study.best_params)

# model = xgb.XGBRegressor(**study.best_params)
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)

# We have a lot of rows, hence we can't use random forest

# Random forest is not possible in this case because of the processing time
# rf = RandomForestRegressor(n_estimators = 50, max_features = 'sqrt', max_depth = 5, random_state = 18).fit(x_train, y_train)

# prediction = rf.predict(x_test)
# mse = mean_squared_error(y_test, prediction)
# rmse = mse**.5
# print(mse)
# print(rmse)

# from sklearn.model_selection import RandomizedSearchCV
# grid_param = {
#     'n_estimators':[100,150,200],
#     #'criterion': ['gini', 'entropy'],
#     'max_depth': range(2,10,1),
#     'min_samples_leaf': range(1,5,1),
#     'min_samples_split': range(2,5,1),
#     'max_features': ['auto', 'log2']
#     }
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = grid_param, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# rf_random.fit(x_train, y_train)