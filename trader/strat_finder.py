import math
import decimal
import datetime
import numpy as np
import pandas as pd
import vectorbt as vbt
from portfoliogen import generate_portfolio

while True:
    try:
        allocation = generate_portfolio()
        symbols = list(allocation.keys())
        price = vbt.YFData.download(symbols, missing_index='drop').get('Close')
        break
    except Exception as e:
        print(f"Exception: {e}. Retrying...")

if price.index.tzinfo is None:
    price.index = price.index.tz_localize('UTC')

# Example strategy: Moving Average Crossover
def run_ma_crossover(price, short_window, long_window):
    fast_ma = vbt.MA.run(price, window=short_window)
    slow_ma = vbt.MA.run(price, window=long_window)
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=100)
    return pf

