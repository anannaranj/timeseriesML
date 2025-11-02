from cleaning import cleaning, asFreq, seasonalDecomposition
from models import XGB, RF, eval
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# get cleaned Dataframe
df = cleaning()


# Make it an hour frequency

# That is what the machine is going to learn
# as it will take ages to learn it by minutes
df = asFreq(df, "h")
df = df.to_frame()

df["active_energy"] = np.where(
    df["active_energy"] > 125,
    np.nan, df["active_energy"])
df["active_energy"] = np.where(
    (df.index >= "2008-8") & (df.index <= "2008-9"),
    np.nan, df["active_energy"])
df.dropna(inplace=True)
# df.interpolate(inplace=True)


# Create some features
df["year"] = df.index.year
df["dayofyear"] = df.index.dayofyear
# df["quarter"] = df.index.quarter
df["season"] = (df.index.month % 12 + 3) // 3
df["harshweather"] = (df["season"] + 1) % 2
df["month"] = df.index.month
df["dayofweek"] = df.index.dayofweek
df["hour"] = df.index.hour

df["notaholiday"] = ~(
    (
        (df.index.month == 7) |
        ((df.index.month == 8) & (df.index.day <= 15)) |
        ((df.index.month == 12) & (df.index.day >= 20)) |
        ((df.index.month == 1) & (df.index.day <= 5))
    ) | (df.index.dayofweek >= 5)
)
df["workhour"] = ((df.index.hour >= 9) & (df.index.hour <= 17)
                  ) & df["notaholiday"]

# in a timeseries forecasting project the only
# column you need to train on is the datetime
train = df[df.index <= "2010-5"]
test = df[df.index > "2010-5"]


# I just didn't figure out how to do seasonal decomposition lol!
# decomposition = seasonalDecomposition(train)

# train['trend'] = decomposition.trend
# train['seasonal'] = decomposition.seasonal
# train['resid'] = decomposition.resid
# train = train.dropna()
# test['trend'] = train["trend"].mean()
# test['seasonal'] = test.index.hour.map(
#     train.groupby(train.index.hour)['seasonal'].mean())
# test['resid'] = pd.NA

# Training and evaluating the models
vars = list(set(train.columns) - set(["active_energy"]))

xtrain = train[vars]
ytrain = train["active_energy"]

xtest = test[vars]
ytest = test["active_energy"]

week1 = test[(test.index >= "2010-5-15") & (test.index <= "2010-5-22")]
xweek1 = week1[vars]
yweek1 = week1["active_energy"]


# try1(xtrain, ytrain, xweek1, yweek1)
# try2(xtrain, ytrain, xweek1, yweek1)

# # # # finding the hyper params:
# xg_reg = xgb.XGBRegressor(eval_metric='rmse')
# param_grid = {
#     'max_depth': [3, 4, 5, 6, 7, 8],
#     'learning_rate': [0.01, 0.1, 0.2, 0.5],
#     'n_estimators': [100, 200, 500, 1000, 2000, 5000, 10000]
# }
#
# grid_search = GridSearchCV(
#     estimator=xg_reg, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=3)
#
# grid_search.fit(xtrain, ytrain)
# print("Best parameters: ", grid_search.best_params_)
# # # # {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 500}
# print("Best RMSE score: ", -grid_search.best_score_)
# # # # 14.734954667237204

# rf_reg = RandomForestRegressor()
# param_grid = {
#     'max_depth': [None, 5, 10, 15, 20],
#     'n_estimators': [100, 200, 500],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2],
#     'bootstrap': [True, False]
# }
#
# grid_search = GridSearchCV(
#     estimator=rf_reg, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
#
# grid_search.fit(xtrain, ytrain)
# print("Best parameters: ", grid_search.best_params_)
# # # # Best parameters:  {'bootstrap': True, 'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 500}
# print("Best RMSE score: ", -grid_search.best_score_)
# # # # Best RMSE score:  15.203679142531392

# xgbmodel = XGB(xtrain, ytrain)
# eval(xgbmodel, xweek1, yweek1)
rfmodel = RF(xtrain, ytrain)

# Evaluation:
# eval(xgbmodel, xtrain, ytrain)
# eval(xgbmodel, xtest, ytest)

eval(rfmodel, xtrain, ytrain)
eval(rfmodel, xtest, ytest)
