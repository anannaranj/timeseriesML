from cleaning import cleaning, asFreq, seasonalDecomposition
from xgb import try1, try2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# get cleaned Dataframe
df = cleaning()


# Make it an hour frequency

# That is what the machine is going to learn
# as it will take ages to learn it by minutes
df = asFreq(df, "h")
df = df["active_energy"]
df = df.to_frame()

df["active_energy"] = np.where(
    df["active_energy"] > 125, np.nan, df["active_energy"])
df.interpolate(inplace=True)

df["year"] = df.index.year
df["dayofyear"] = df.index.dayofyear
df["quarter"] = df.index.quarter
df["month"] = df.index.month
df["dayofweek"] = df.index.dayofweek
df["hour"] = df.index.hour

df["weekend"] = df.index.dayofweek >= 5
df["holiday"] = ((df.index.month == 7) |
                 ((df.index.month == 8) & (df.index.day <= 15)) |
                 ((df.index.month == 12) & (df.index.day >= 20)) |
                 ((df.index.month == 1) & (df.index.day <= 5)))

# in a timeseries forecasting project the only
# column you need to train on is the datetime
train = df[df.index <= "2010-5"]
test = df[df.index > "2010-5"]

vars = list(set(df.columns) - set(["active_energy"]))

xtrain = train[vars]
ytrain = train["active_energy"]

xtest = test[vars]
ytest = test["active_energy"]

week1 = df[(df.index >= "2010-5-15") & (df.index <= "2010-5-22")]
xweek1 = week1[vars]
yweek1 = week1["active_energy"]

# seasonalDecomposition(df)

# try1(xtrain, ytrain, xweek1, yweek1)
# try2(xtrain, ytrain, xweek1, yweek1)
