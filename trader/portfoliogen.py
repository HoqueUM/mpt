import pandas as pd
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from get_stocks import get_random_stocks
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def generate_portfolio(num_stocks=30):
    """
    Returns an optimized portfolio of randomly selected stocks. Dates are in the range from
    the first of January to yesterday.
    """
    tickers = get_random_stocks(num_stocks=num_stocks)
    stock_data = yf.download(tickers, start=datetime.today() - timedelta(days=30*6), end=datetime.today())['Adj Close']

    mu = expected_returns.mean_historical_return(stock_data)
    S = risk_models.sample_cov(stock_data)

    ef = EfficientFrontier(mu, S)
    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    ef.portfolio_performance(verbose=True)

    latest_prices = get_latest_prices(stock_data)

    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=50)
    allocation, leftover = da.greedy_portfolio()
    return allocation