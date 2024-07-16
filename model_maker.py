import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import datetime

# Download historical data
start = datetime.datetime(1993, 1, 22)
end = datetime.datetime.today()

spy = yf.download('SPY', start=start, end=end)

# Function to calculate technical indicators
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data):
    shortEMA = data['Close'].ewm(span=12, adjust=False).mean()
    longEMA = data['Close'].ewm(span=26, adjust=False).mean()
    macd = shortEMA - longEMA
    return macd

def calculate_roc(data, window=14):
    roc = ((data['Close'] - data['Close'].shift(window)) / data['Close'].shift(window)) * 100
    return roc

def calculate_stochastic_oscillator(data, window=14):
    high = data['High'].rolling(window=window).max()
    low = data['Low'].rolling(window=window).min()
    k = ((data['Close'] - low) / (high - low)) * 100
    d = k.rolling(window=3).mean()
    return k, d

def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band

# Calculate technical indicators
spy['RSI'] = calculate_rsi(spy)
spy['MACD'] = calculate_macd(spy)
spy['ROC'] = calculate_roc(spy)
spy['%K'], spy['%D'] = calculate_stochastic_oscillator(spy)
spy['upper_band'], spy['lower_band'] = calculate_bollinger_bands(spy)

# Define additional features
spy['50_SMA'] = spy['Close'].rolling(window=50).mean()
spy['volatility'] = spy['Close'].rolling(window=50).std()
spy['pct_change'] = spy['Close'].pct_change()
spy['log_return'] = np.log(spy['Close'] / spy['Close'].shift(1))
spy['cumulative_return'] = np.exp(spy['log_return'].cumsum())
spy['200_SMA'] = spy['Close'].rolling(window=200).mean()
spy['amount_change'] = spy['Close'].diff()

# Define target classification function
def classify_target(sma_condition, ema_condition, rsi_condition, macd_condition, bollinger_condition, roc_condition, stochastic_condition):
    if sma_condition and ema_condition and rsi_condition and macd_condition and bollinger_condition and roc_condition and stochastic_condition:
        return 'STRONG BULL'
    elif sma_condition and ema_condition and rsi_condition and macd_condition and bollinger_condition:
        return 'MILD BULL'
    elif not sma_condition and not ema_condition and not rsi_condition and not macd_condition and not bollinger_condition:
        return 'NEUTRAL'
    elif not sma_condition and not ema_condition and not rsi_condition and not macd_condition and bollinger_condition:
        return 'MILD BEAR'
    elif not sma_condition and not ema_condition and not rsi_condition and not macd_condition and bollinger_condition and roc_condition and stochastic_condition:
        return 'STRONG BEAR'
    else:
        return 'NEUTRAL'

# Apply target classification logic
cleaned_spy = spy.dropna()
cleaned_spy['EMA'] = cleaned_spy['Close'].ewm(span=20, adjust=False).mean()

cleaned_spy['sma_condition'] = cleaned_spy['50_SMA'] > cleaned_spy['200_SMA']
cleaned_spy['ema_condition'] = cleaned_spy['Close'] > cleaned_spy['EMA']
cleaned_spy['rsi_condition'] = cleaned_spy['RSI'] < 30  # Example RSI condition (oversold)
cleaned_spy['macd_condition'] = cleaned_spy['MACD'] > 0  # Example MACD condition (bullish crossover)
cleaned_spy['bollinger_condition'] = (cleaned_spy['Close'] < cleaned_spy['lower_band']) & (cleaned_spy['RSI'] < 30)
cleaned_spy['roc_condition'] = cleaned_spy['ROC'] > 0  # Example ROC condition (positive momentum)
cleaned_spy['stochastic_condition'] = (cleaned_spy['%K'] > cleaned_spy['%D']) & (cleaned_spy['%K'] < 20)  # Example Stochastic condition

cleaned_spy['target'] = cleaned_spy.apply(
    lambda row: classify_target(row['sma_condition'], row['ema_condition'], row['rsi_condition'], 
                                row['macd_condition'], row['bollinger_condition'], row['roc_condition'], 
                                row['stochastic_condition']),
    axis=1
)

# Prepare data for training
X = cleaned_spy[['50_SMA', 'volatility', 'pct_change', 'log_return', 'cumulative_return',
                 '200_SMA', 'amount_change', 'RSI', 'MACD', 'upper_band', 'lower_band',
                 'ROC', '%K', '%D']]
y = cleaned_spy['target']

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
clf = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print(classification_report(y_test, y_pred))
