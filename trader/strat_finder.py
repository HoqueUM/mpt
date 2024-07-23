from portfoliogen import generate_portfolio
import warnings
import vectorbt as vbt
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings("ignore", category=UserWarning)

# Define the date range
start_date = '2020-01-01'
end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

while True:
    try:
        # Generate the portfolio allocation
        allocation = generate_portfolio()
        symbols = list(allocation.keys())

        # Download price data
        price = vbt.YFData.download(symbols, start=start_date, end=end_date, missing_index='drop').get('Close')
        break
    except Exception as e:
        print(f'Error: {e}')

# Set the range for moving average windows
windows = np.arange(2, 101)

# Run the moving average combinations
fast_ma, slow_ma = vbt.MA.run_combs(price, window=windows, r=2, short_names=['fast', 'slow'])

# Define entry and exit signals
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

# Define portfolio parameters
pf_kwargs = dict(size=np.inf, fees=0.001, freq='1D')

# Initialize a dictionary to store the best strategies for each ticker
best_strategies = {}

for ticker in symbols:
    # Create the portfolio for each ticker
    
    pf = vbt.Portfolio.from_signals(price, entries, exits, **pf_kwargs)

    # Get the total returns for each combination
    total_returns = pf.total_return()

    # Identify the strategy with the highest return for this ticker
    max_return_idx = total_returns.idxmax()
    best_strategies[ticker] = max_return_idx
    print(f'Ticker: {ticker}, Best Strategy Index: {max_return_idx}, Total Return: {total_returns[max_return_idx]}')

print("Best strategies for all tickers:", best_strategies)
