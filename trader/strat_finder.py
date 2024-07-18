import math
import decimal
import datetime
import numpy as np
import pandas as pd
import vectorbt as vbt

dec = decimal.Decimal

def position(now=None): 
    if now is None: 
        now = datetime.datetime.now()

    # Ensure `now` is timezone-aware if necessary
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)

    diff = now - datetime.datetime(2001, 1, 1, tzinfo=datetime.timezone.utc)
    days = dec(diff.days) + (dec(diff.seconds) / dec(86400))
    lunations = dec("0.20439731") + (days * dec("0.03386319269"))

    return lunations % dec(1)

def phase(pos): 
    index = (pos * dec(8)) + dec("0.5")
    index = math.floor(index)
    return {
        0: "New Moon", 
        1: "Waxing Crescent", 
        2: "First Quarter", 
        3: "Waxing Gibbous", 
        4: "Full Moon", 
        5: "Waning Gibbous", 
        6: "Last Quarter", 
        7: "Waning Crescent"
    }[int(index) & 7]

# Function to get moon phase for a given date
def get_moon_phase(date):
    pos = position(date)
    return phase(pos)

# Define the list of symbols
symbols = ['F', 'NYCB']

# Fetch historical price data
price = vbt.YFData.download(symbols, start='2023-07-16', end='2024-07-16', missing_index='drop').get('Close')

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

# Compute Bollinger Bands
bb = vbt.BBANDS.run(price)

# Additional condition: Buy if the price is below the lower Bollinger Band
entries = entries & (price < bb.lower)

# Additional condition: Sell if the price is above the upper Bollinger Band
exits = exits & (price > bb.upper)

# Add moon phase condition
moon_phases = price.index.to_series().map(get_moon_phase)

# Add moon phase to entries and exits conditions


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
pf[(10, 20, 'F')].plot().show()
