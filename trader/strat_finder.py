import math
import decimal
import datetime
import numpy as np
import pandas as pd
import vectorbt as vbt

symbols = ['CSUAY', 'BCMXY', 'DNKEY']

# Fetch historical price data
price = vbt.YFData.download(symbols, missing_index='drop').get('Close')

# Ensure price index is timezone-aware if necessary
if price.index.tzinfo is None:
    price.index = price.index.tz_localize('UTC')

# Define the range of windows for moving averages
windows = np.arange(2, 101)

# Compute moving averages for all combinations of windows
fast_ma, slow_ma = vbt.MA.run_combs(price, window=windows, r=2, short_names=['fast', 'slow'])

# Generate entry and exit signals
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)


# Define portfolio parameters
pf_kwargs = dict(size=np.inf, fees=0.001, freq='1D')

# Backtest the strategies
pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=50, **pf_kwargs)

# Analyze results
fig = pf.total_return().vbt.heatmap(
    x_level='fast_window', y_level='slow_window', slider_level='symbol', symmetric=True,
    trace_kwargs=dict(colorbar=dict(title='Total return', tickformat='%')))
fig.show()

# Plot a specific strategy
pf[(10, 20, 'CSUAY')].plot().show()
pf[(10, 20, 'BCMXY')].plot().show()
pf[(10, 20, 'DNKEY')].plot().show()
