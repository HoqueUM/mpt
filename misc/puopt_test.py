# https://github.com/polakowo/vectorbt/blob/master/examples/PortfolioOptimization.ipynb
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import base_optimizer
import pytz

import vectorbt as vbt
from vectorbt.generic.nb import nanmean_nb
from vectorbt.portfolio.nb import order_nb, sort_call_seq_nb
from vectorbt.portfolio.enums import SizeType, Direction

symbols = ["F", "NYCB", "SIRI", "CPG", "SWN"]
start_date = datetime(2017, 1, 1, tzinfo=pytz.utc)
end_date = datetime(2024, 1, 1, tzinfo=pytz.utc)
num_tests = 2000

vbt.settings.array_wrapper['freq'] = 'days'
vbt.settings.returns['year_freq'] = '252 days'
vbt.settings.portfolio['seed'] = 42
vbt.settings.portfolio.stats['incl_unrealized'] = True



yfdata = vbt.YFData.download(symbols, start=start_date, end=end_date)

ohlcv = yfdata.concat()
price = ohlcv['Close']
returns = price.pct_change()
print(returns.corr())
