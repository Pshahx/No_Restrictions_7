import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Function to calculate metrics
def calculate_metrics(data):
    data["Price1"] = data["Close"].shift(-1)
    data["PriceDiff"] = data["Price1"] - data["Close"]
    data["DailyReturn"] = data["PriceDiff"] / data["Close"]
    data["MA10"] = data["Close"].rolling(window=10).mean()
    data["MA50"] = data["Close"].rolling(window=50).mean()
    data["LongOrNot"] = (data["MA10"] > data["MA50"]).astype(int)
    data["Profit"] = (data["Price1"] - data["Close"]) * data["LongOrNot"]
    data["Wealth"] = data["Profit"].cumsum()
    return data

# Function to plot closing price and moving averages
def plot_moving_averages(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Close"], label="Close Price", color="blue")
    plt.plot(data.index, data["MA10"], label="10-Day Moving Average", color="orange")
    plt.plot(data.index, data["MA50"], label="50-Day Moving Average", color="red")
    plt.title("TD Close Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()

# Function to plot wealth
def plot_wealth(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Wealth"], label="Cumulative Wealth", color="gold")
    plt.title("Wealth Over Time")
    plt.xlabel("Date")
    plt.ylabel("Wealth")
    plt.grid()
    plt.show()

def plot_profit(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Profit"], label="Profit Over Time", color="blue")
    plt.title("Profit Over Time")
    plt.xlabel("Date")
    plt.ylabel("Profit")
    plt.grid()
    plt.show()

# Main function
def main():
    stock = yf.Ticker("MARA")
    data = stock.history(period="5y")
    data = calculate_metrics(data)
    print(data)
    plot_moving_averages(data)
    plot_profit(data)
    plot_wealth(data)
   

if __name__ == "__main__":
    main()