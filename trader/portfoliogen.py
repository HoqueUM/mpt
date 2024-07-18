import pandas as pd
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# Define the ticker symbols of the assets you're interested in
tickers = ["T","F", "SIRI", "VOD", "NOK", "WBA", "GCB", "NYCB","BURBY", "ABEV", "ING", "PBR", "VWAGY", "WDS"]

# Fetch historical price data using yfinance
stock_data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']

# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(stock_data)
S = risk_models.sample_cov(stock_data)

# Optimize for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
ef.save_weights_to_file("weights.csv")  # saves to file
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

latest_prices = get_latest_prices(stock_data)

da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=50)
allocation, leftover = da.greedy_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))
