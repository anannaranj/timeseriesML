import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def XGB(xtrain, ytrain):
    # # # # {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 500}
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01,
                             random_state=42, early_stopping_rounds=10,
                             max_depth=7)
    model.fit(xtrain, ytrain, eval_set=[(xtrain, ytrain)], verbose=10)
    return model


def RF(xtrain, ytrain):
    # # # # Best parameters:  {
    # 'bootstrap': True, 'max_depth': 10,
    # 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 500
    # }
    model = RandomForestRegressor(n_estimators=500, max_depth=10,
                                  min_samples_leaf=2, min_samples_split=5)
    model.fit(xtrain, ytrain)
    return model


def eval(model, x, y):
    predict = model.predict(x)
    mse = mean_squared_error(y, predict)
    print("MSE:", mse)
    print("RMSE:", np.sqrt(mse))
    plt.plot(y)
    plt.plot(pd.DataFrame(predict, index=y.index))
    plt.show()

# def try1(xtrain, ytrain, xdata, ydata):
#
#     # # # try number 1:
#     model = xgb.XGBRegressor()
#     model.fit(xtrain, ytrain)
#     result = model.predict(xdata)
#
#     plt.plot(ydata)
#     plt.plot(pd.DataFrame(result, index=ydata.index))
#     plt.show()
#
#
# def try2(xtrain, ytrain, xdata, ydata):
#
#     # # # try number 2:
#     model = xgb.XGBRegressor(n_estimators=4500, learning_rate=0.8,
#                              random_state=1)
#     model.fit(xtrain, ytrain, eval_set=[(xtrain, ytrain)],  verbose=True)
#     result = model.predict(xdata)
#
#     plt.plot(ydata)
#     plt.plot(pd.DataFrame(result, index=ydata.index))
#     plt.show()
#     # # # best train rmse is 1.666
