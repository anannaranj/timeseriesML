import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns

# Read data
df = pd.read_csv(
    "./data/household_power_consumption.txt",
    sep=";",
    low_memory=False,
    na_values="?")

df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True)

del df["Date"]
del df["Time"]

df = df.set_index("datetime")
# make it in KWh
df["active_energy"] = df["Global_active_power"] * 1000 / 60
del df["Global_active_power"]

# df.ffill(inplace=True)

# df.bfill(inplace=True)

# df.interpolate(inplace=True)

# si = SimpleImputer(strategy="mean")
# df[:] = si.fit_transform(df[:])

# si = SimpleImputer(strategy="median")
# df[:] = si.fit_transform(df[:])

# plot random ranges
# df[(df.index > "2009-5") & (df.index < "2009-6")
#    ]["active_energy"].plot(color="#35567e")
# plt.show()
# df[(df.index > "2010-1") & (df.index < "2010-2")
#    ]["active_energy"].plot(color="#35567e")
# plt.show()

df.drop(index=df[(df["active_energy"].isnull()) & (
    df["active_energy"].shift().isnull())].index, inplace=True)
df.ffill(inplace=True)

days = df.asfreq("D")
days.interpolate(inplace=True)
daysResult = seasonal_decompose(days["active_energy"])
# plt.plot(days["active_energy"], label="Days", color="#35567e")
# plt.legend()
# plt.show()
# plt.plot(daysResult.trend, label="Days Trend", color="#35567e")
# plt.legend()
# plt.show()
# plt.plot(daysResult.seasonal, label="Days Seasonal", color="#35567e")
# plt.legend()
# plt.show()


weeks = df.asfreq("W")
weeks.interpolate(inplace=True)
weeksResult = seasonal_decompose(weeks["active_energy"])
# plt.plot(weeks["active_energy"], label="Weeks", color="#35567e")
# plt.legend()
# plt.show()
# plt.plot(weeksResult.trend, label="Weeks Trend", color="#35567e")
# plt.legend()
# plt.show()
# plt.plot(weeksResult.seasonal, label="Weeks Seasonal", color="#35567e")
# plt.legend()
# plt.show()


months = df.asfreq("ME")
months.interpolate(inplace=True)
monthsResult = seasonal_decompose(months["active_energy"])
# plt.plot(months["active_energy"], label="Months", color="#35567e")
# plt.legend()
# plt.show()
# plt.plot(monthsResult.trend, label="Months Trend", color="#35567e")
# plt.legend()
# plt.show()
# plt.plot(monthsResult.seasonal, label="Months Seasonal", color="#35567e")
# plt.legend()
# plt.show()

# sns.heatmap(pd.DataFrame(
#     days.corr()["active_energy"]), annot=True, cmap="BrBG")
# plt.title("Days correlation")
# plt.show()
# sns.heatmap(pd.DataFrame(
#     weeks.corr()["active_energy"]), annot=True, cmap="BrBG")
# plt.title("Weeks correlation")
# plt.show()
# sns.heatmap(pd.DataFrame(
#     months.corr()["active_energy"]), annot=True, cmap="BrBG")
# plt.title("Months correlation")
# plt.show()
