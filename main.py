from cleaning import cleaning, asFreq
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

# get cleaned Dataframe
df = cleaning()


# Make it an hour frequency

# That is what the machine is going to learn
# as it will take ages to learn it by minutes
df = asFreq(df, "h")
df = df["active_energy"]
df = df.to_frame()

df["year"] = df.index.year
df["dayofyear"] = df.index.dayofyear
df["quarter"] = df.index.quarter
df["month"] = df.index.month
df["dayofweek"] = df.index.dayofweek
df["hour"] = df.index.hour

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


# # # # try number 1:
# model = xgb.XGBRegressor()
# model.fit(xtrain, ytrain)
# result = model.predict(xweek1)
#
# plt.plot(yweek1)
# plt.plot(pd.DataFrame(result, index=yweek1.index))
# plt.show()


# # # # try number 2:
# model = xgb.XGBRegressor(n_estimators=4500, learning_rate=0.8,
#                          random_state=1)
# model.fit(xtrain, ytrain, eval_set=[(xtrain, ytrain)],  verbose=True)
# result = model.predict(xweek1)
#
# plt.plot(yweek1)
# plt.plot(pd.DataFrame(result, index=yweek1.index))
# plt.show()
# # # # best rmse is 1.666
