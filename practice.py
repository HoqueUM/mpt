import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import datetime
import cvxpy as cp

# Get the data
start = datetime.datetime(2020, 1, 1)
end = datetime.datetime(2024, 1, 1)

# Get the data
data = yf.download(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NFLX', 'META', 'NVDA', 'INTC', 'AMD'], start=start, end=end)

