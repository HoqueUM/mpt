
from portfoliogen import generate_portfolio
import backtrader as bt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

allocation = generate_portfolio()
symbols = list(allocation.keys())


