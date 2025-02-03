import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt

# Fetch the data
stock = yf.Ticker("MARA")
data = stock.history(period="5y")

# Calculate Log Returns
data['LogReturn'] = np.log(data['Close'] / data['Close'].shift(1))

mean_log_return = data['LogReturn'].mean()
std_log_return = data['LogReturn'].std(ddof=1)

#print(mean_log_return, std_log_return)
density = pd.DataFrame()
density['x'] = np.linspace(data['LogReturn'].min() - 0.01, data['LogReturn'].max() + 0.01, 500)
density['pdf'] = norm.pdf(density['x'], mean_log_return, std_log_return)
plt.figure(figsize=(12, 6))
plt.hist(data['LogReturn'], bins=50, density=True, alpha=0.6, color='blue', label="Log Return Histogram")
plt.plot(density['x'], density['pdf'], color='red', linewidth=2, label="Normal Distribution")

plt.show()

VaR = norm.ppf(0.05, mean_log_return, std_log_return)
