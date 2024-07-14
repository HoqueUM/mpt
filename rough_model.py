import yfinance as yf
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from newsapi import NewsApiClient
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingClassifier
# Collect text data
def collect_text_data(stock, start_date, end_date):
    newsapi = NewsApiClient(api_key='b249cf68dea34a99b92ab09163ee404d')
    all_articles = newsapi.get_everything(q=stock, from_param=start_date, to=end_date, language='en', sort_by='relevancy')
    headlines = [article['title'] for article in all_articles['articles']]
    return headlines

# Calculate indicators
def calculate_indicators(data):
    data['50_SMA'] = data['Close'].rolling(window=50).mean()
    data['200_SMA'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    data['Avg_Volume'] = data['Volume'].rolling(window=20).mean()
    return data

# Calculate RSI
def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Get sentiment score
def get_sentiment_score(text):
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    result = nlp(text)
    sentiment_score = 0
    if result[0]['label'] == 'positive':
        sentiment_score = 1
    elif result[0]['label'] == 'negative':
        sentiment_score = -1
    return sentiment_score

# Aggregate sentiment scores
'''
def aggregate_sentiment_scores(stock, start_date, end_date):
    text_data = collect_text_data(stock, start_date, end_date)
    if not text_data:
        return 0.0  # Return neutral sentiment if no data is available
    sentiment_scores = [get_sentiment_score(text) for text in text_data]
    return np.mean(sentiment_scores)  # Average sentiment score
'''
# Define universe of small-cap stocks
small_cap_stocks = ['F']  # Replace with actual small-cap stocks

# Initialize portfolio and results
portfolio = {}
results = []
portfolio_values = []  # List to track portfolio value over time
dates = []  # List to track dates

# Collect and preprocess data
data_frames = []
for stock in small_cap_stocks:
    data = yf.download(stock, start="2010-01-01", end="2024-01-01")
    data = calculate_indicators(data)
    # Add sentiment score (example, replace with actual sentiment data collection)
    #data['Sentiment'] = data.index.map(lambda date: aggregate_sentiment_scores(stock, str(date.date()), str(date.date())))
    data_frames.append(data)

# Combine data frames for machine learning
combined_data = pd.concat(data_frames)
combined_data.fillna(method='ffill', inplace=True)  # Forward fill to handle initial NaNs
combined_data.fillna(method='bfill', inplace=True) # Backfill to ensure no NaNs remain

# Feature engineering
features = combined_data[['50_SMA', '200_SMA', 'RSI', 'Avg_Volume']]
target = (combined_data['Close'].shift(-1) > combined_data['Close']).astype(int)  # 1 if price increases, 0 otherwise

# Train-test split
data.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model training
model = HistGradientBoostingClassifier()
model.fit(X_train, y_train)

# Model evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2%}")

# Backtesting loop
for stock in small_cap_stocks:
    data = yf.download(stock, start="2023-01-01", end="2024-01-01")
    data = calculate_indicators(data)
    
    for i in range(len(data)):
        date = str(data.index[i].date())
        
        feature_row = data.iloc[i][['50_SMA', '200_SMA', 'RSI', 'Avg_Volume']]
        prediction = model.predict(np.array(feature_row).reshape(1, -1))
        
        if prediction == 1 and data['Close'][i] > data['50_SMA'][i] > data['200_SMA'][i] and 30 < data['RSI'][i] < 70 and data['Volume'][i] > data['Avg_Volume'][i]:
            if stock not in portfolio:
                buy_price = data['Close'][i]
                stop_loss = buy_price * 0.95
                portfolio[stock] = {'buy_price': buy_price, 'stop_loss': stop_loss}
        
        if stock in portfolio:
            if data['Close'][i] < data['50_SMA'][i] or data['RSI'][i] < 30 or data['Close'][i] < portfolio[stock]['stop_loss'] or prediction == 0:
                sell_price = data['Close'][i]
                gain = (sell_price - portfolio[stock]['buy_price']) / portfolio[stock]['buy_price']
                results.append({'stock': stock, 'buy_price': portfolio[stock]['buy_price'], 'sell_price': sell_price, 'gain': gain})
                del portfolio[stock]
        
        # Calculate portfolio value
        portfolio_value = sum(data['Close'][i] for stock in portfolio)
        portfolio_values.append(portfolio_value)
        dates.append(data.index[i])

# Calculate overall performance
total_gain = sum(result['gain'] for result in results) / len(results)
print(f"Average Gain: {total_gain:.2%}")

# Plot portfolio value over time
plt.figure(figsize=(12, 6))
plt.plot(dates, portfolio_values, label='Portfolio Value')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.title('Portfolio Value Over Time')
plt.legend()
plt.grid(True)
plt.savefig('portfolio_value_over_time.png')
plt.show()

# Further optimization and forward testing can be done after initial backtest
