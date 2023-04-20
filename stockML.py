import yfinance as yf
import matplotlib.pyplot as plt
import os
import math
import pandas as pd
from functions import timer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import precision_score

stock_name = "MSFT"
outFile = open(stock_name + ".txt", 'w')
if stock_name == "":
    stock_name = input("Enter stock to graph: ")

DATA_PATH = "data/" + stock_name.lower() + "_data.json"

if os.path.exists(DATA_PATH):
    # Read from file if we've already downloaded the data.
    with open(DATA_PATH) as f:
        stock_hist = pd.read_json(DATA_PATH)
else:
    stock = yf.Ticker(stock_name)
    #Get entire history of stock
    stock_hist = stock.history(period="max")
    # Save file to json in case we need it later.  This prevents us from having to re-download it every time.
    stock_hist.to_json(DATA_PATH)

#Plot closing price vs. time
plt.plot(stock_hist["Close"])
plt.title(stock_name)
plt.ylabel("Price at Close")
plt.xlabel("Year")
plt.savefig("static/stock.png")
#plt.show(block = True)

#Create dataset based on closing price over entire stock history
data = stock_hist[["Close"]]
data = data.rename(columns={"Close":"Actual_Close"})

#Return 1 if next day is greater than prev day
data["Target"] = stock_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

#Make new list of stock history shifted by one day to determine change
stock_prev = stock_hist.copy()
stock_prev = stock_prev.shift(1)

predictors = ["Close", "Volume", "Open", "High", "Low"]
data = data.join(stock_prev[predictors]).iloc[1:]

#Create an ml model
@timer
def createModel(num_est, min_samples):
    model = RandomForestClassifier(n_estimators=num_est, min_samples_split=min_samples, random_state=1)
    outFile.write(f"\n# of estimators: {num_est}\n")
    outFile.write(f"Minimum samples: {min_samples}\n")
    return model

@timer
def backtest(data, model, predictors, threshhold, start=math.floor(num_days/2), step=math.floor(num_days/10)):
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
        preds[preds > threshhold] = 1
        preds[preds <= threshhold] = 0

        # Combine predictions and test values
        combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)

        predictions.append(combined)
        
    outFile.write(f"Start: {start}\n")
    outFile.write(f"Step: {step}\n")
    return pd.concat(predictions)

#Create more predictors to improve accuracy
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
model = createModel(100, 20)
predictions = backtest(data.iloc[365:], model, full_predictors, 0.7)
precision_score = precision_score(predictions["Target"], predictions["Predictions"]) * 100

outFile.write(f"Model is {precision_score:0.2f}% precise\n")
#outFile.write(predictions["Predictions"].value_counts())
outFile.write("\n\n\n")
outFile.close()
print("done")