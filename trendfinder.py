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

# Download the data
NVDA = yf.download('NVDA', start=start, end=end)
GOOG = yf.download('GOOG', start=start, end=end)


spy_close = NVDA['Adj Close']
vix_close = GOOG['Adj Close']

# Calculate daily returns
spy_returns = spy_close.pct_change().dropna()
vix_returns = vix_close.pct_change().dropna()

# Combine into a DataFrame
returns = pd.DataFrame({'NVDA': spy_returns, 'GOOG': vix_returns})

# Calculate the covariance matrix
cov_matrix = returns.cov()

# Number of assets
n = 2

# Variables for optimization
w = cp.Variable(n)

# Objective function (minimize portfolio variance)
objective = cp.Minimize(cp.quad_form(w, cov_matrix.values))

# Constraints (weights sum to 1)
constraints = [cp.sum(w) == 1, w >= 0]

# Solve the problem
prob = cp.Problem(objective, constraints)
prob.solve()

# Print the optimal weights
print(f"Optimal weights: NVDA: {w.value[0]:.2f}, GOOG: {w.value[1]:.2f}")

# Graph the two
plt.figure(figsize=(10, 6))
spy_close.plot(label='NVDA')
vix_close.plot(label='GOOG')
plt.legend()
plt.savefig('nvda_goog.png')