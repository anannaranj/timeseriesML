import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd


def try1(xtrain, ytrain, xdata, ydata):

    # # # try number 1:
    model = xgb.XGBRegressor()
    model.fit(xtrain, ytrain)
    result = model.predict(xdata)

    plt.plot(ydata)
    plt.plot(pd.DataFrame(result, index=ydata.index))
    plt.show()


def try2(xtrain, ytrain, xdata, ydata):

    # # # try number 2:
    model = xgb.XGBRegressor(n_estimators=4500, learning_rate=0.8,
                             random_state=1)
    model.fit(xtrain, ytrain, eval_set=[(xtrain, ytrain)],  verbose=True)
    result = model.predict(xdata)

    plt.plot(ydata)
    plt.plot(pd.DataFrame(result, index=ydata.index))
    plt.show()
    # # # best rmse is 1.666


def XGB(xtrain, ytrain):
    model = xgb.XGBRegressor(n_estimators=50000, learning_rate=0.2,
                             random_state=1, early_stopping_rounds=50)
    model.fit(xtrain, ytrain, eval_set=[(xtrain, ytrain)],  verbose=True)
    return model
