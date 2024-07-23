from portfoliogen import generate_portfolio
import warnings
import vectorbt as vbt
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings("ignore", category=UserWarning)

def get_best_strat(min_price=3, max_price=5, total_portfolio_value=50):
    start_date = '2020-01-01'
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    while True:
        try:
            # Generate the portfolio allocation
            allocation = generate_portfolio(min_price=min_price, max_price=max_price, total_portfolio_value=total_portfolio_value)
            symbols = list(allocation.keys())

            # Download price data
            #price = vbt.YFData.download(symbols, start=start_date, end=end_date, missing_index='drop').get('Close')
            break
        except Exception as e:
            print(f'Error: {e}')

    best_strategies = {}
    for symbol in symbols:
        # Set the range for moving average windows
        windows = np.arange(2, 101)
        current_price = vbt.YFData.download(symbol, start=start_date, end=end_date, missing_index='drop').get('Close')
        # Run the moving average combinations
        fast_ma, slow_ma = vbt.MA.run_combs(current_price, window=windows, r=2, short_names=['fast', 'slow'])

        # Define entry and exit signals
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)

        # Define portfolio parameters
        pf_kwargs = dict(size=np.inf, fees=0.001, freq='1D')

            
        pf = vbt.Portfolio.from_signals(current_price, entries, exits, **pf_kwargs)
        returns = pf.total_return()
        max_return = pf.total_return().idxmax()
        current_percent = round(pf.total_return().max() * 100, 2)    
        best_strategies[symbol] = {
            'fast_ma': max_return[0],
            'slow_ma': max_return[1],
            'return': current_percent,
            'num_shares': allocation[symbol]
        }
    return best_strategies

