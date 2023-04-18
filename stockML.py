import yfinance as yf
import matplotlib.pyplot as plt
import os
import pandas as pd
from functions import timer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import precision_score

stock_name = "MSFT"
if stock_name == "":
    stock_name = input("Enter stock to graph: ")
stock = yf.Ticker(stock_name)

#Get entire history of stock
stock_hist = stock.history(period="max")

#Plot closing price vs. time
plt.plot(stock_hist["Close"])
plt.title(stock_name)
plt.ylabel("Price at Close")
plt.xlabel("Year")
plt.savefig("static/stock.png")
plt.show()

data = stock_hist[["Close"]]
data = data.rename(columns={"Close":"Actual_Close"})

#Return 1 if next day is greater than prev day
data["Target"] = stock_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

stock_prev = stock_hist.copy()
stock_prev = stock_prev.shift(1)

predictors = ["Close", "Volume", "Open", "High", "Low"]
data = data.join(stock_prev[predictors]).iloc[1:]

#Create an ml model
model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)

# Create a train and test set
train = data.iloc[:-100]
test = data.iloc[-100:]

model.fit(train[predictors], train["Target"])

# Evaluate error of predictions
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
precision_score(test["Target"], preds)

combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
combined.plot()

i = 1000
step = 750

train = data.iloc[0:i].copy()
test = data.iloc[i:(i+step)].copy()
model.fit(train[predictors], train["Target"])
preds = model.predict(test[predictors])

preds = model.predict_proba(test[predictors])[:,1]
preds = pd.Series(preds, index=test.index)
preds[preds > .6] = 1
preds[preds<=.6] = 0

@timer
def backtest(data, model, predictors, start=1000, step=250):
    predictions = []
    # Loop over the dataset in increments
    for i in range(start, data.shape[0], step):
        # Split into train and test sets
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()

        # Fit the random forest model
        model.fit(train[predictors], train["Target"])

        # Make predictions
        preds = model.predict_proba(test[predictors])[:,1]
        preds = pd.Series(preds, index=test.index)
        preds[preds > .6] = 1
        preds[preds<=.6] = 0

        # Combine predictions and test values
        combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)

        predictions.append(combined)

    return pd.concat(predictions)

predictions = backtest(data, model, predictors)

weekly_mean = data.rolling(7).mean()["Close"]
quarterly_mean = data.rolling(90).mean()["Close"]
annual_mean = data.rolling(365).mean()["Close"]

weekly_trend = data.shift(1).rolling(7).sum()["Target"]

data["weekly_mean"] = weekly_mean / data["Close"]
data["quarterly_mean"] = quarterly_mean / data["Close"]
data["annual_mean"] = annual_mean / data["Close"]

data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]

data["weekly_trend"] = weekly_trend

data["open_close_ratio"] = data["Open"] / data["Close"]
data["high_close_ratio"] = data["High"] / data["Close"]
data["low_close_ratio"] = data["Low"] / data["Close"]

full_predictors = predictors + ["weekly_mean", "quarterly_mean", "annual_mean", "annual_weekly_mean", "annual_quarterly_mean", "open_close_ratio", "high_close_ratio", "low_close_ratio"]
predictions = backtest(data.iloc[365:], model, full_predictors)
precision_score = precision_score(predictions["Target"], predictions["Predictions"]) * 100
print(f"Model is {precision_score:0.2f}% precise")
print(predictions["Predictions"].value_counts())