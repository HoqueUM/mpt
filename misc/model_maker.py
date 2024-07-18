import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import datetime
import matplotlib.pyplot as plt

# Download historical data
start = datetime.datetime(1993, 1, 22)
end = datetime.datetime.today()

spy = yf.download('SPY', start=start, end=end)

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD
def calculate_macd(data):
    shortEMA = data['Close'].ewm(span=12, adjust=False).mean()
    longEMA = data['Close'].ewm(span=26, adjust=False).mean()
    macd = shortEMA - longEMA
    return macd

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band

# Calculate technical indicators
spy['50_SMA'] = spy['Close'].rolling(window=50).mean()
spy['volatility'] = spy['Close'].rolling(window=50).std()
spy['pct_change'] = spy['Close'].pct_change()
spy['log_return'] = np.log(spy['Close'] / spy['Close'].shift(1))
spy['cumulative_return'] = np.exp(spy['log_return'].cumsum())
spy['200_SMA'] = spy['Close'].rolling(window=200).mean()
spy['amount_change'] = spy['Close'] - spy['Open']
spy['upper_band'], spy['lower_band'] = calculate_bollinger_bands(spy)
spy['RSI'] = calculate_rsi(spy)
spy['MACD'] = calculate_macd(spy)

# Calculate 200-day SMA slope and define target variable
spy['200_SMA_slope'] = spy['200_SMA'].diff()  # Slope of 200-day SMA
neutral_threshold = 0.1  # Adjust as needed
spy['target'] = np.where(spy['200_SMA_slope'] > neutral_threshold, 'BULL',
                         np.where(spy['200_SMA_slope'] < -neutral_threshold, 'BEAR', 'NEUTRAL'))

# Drop NaN values and prepare features and target
cleaned_spy = spy.dropna()
X = cleaned_spy[['50_SMA', 'volatility', 'pct_change', 'log_return', 'cumulative_return', '200_SMA', 'amount_change', 'RSI', 'MACD', 'upper_band', 'lower_band']]
y = cleaned_spy['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest classifier
clf = RandomForestClassifier(n_estimators=500, random_state=42)
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

