import requests
import json
import numpy as np
import random

def get_stocks():
    """
    Returns an array of json objects. A ticker and a price.
    """
    url = "https://scanner.tradingview.com/america/scan"

    payload = json.dumps({
    "columns": [
        "name",
        "description",
        "logoid",
        "close"
    ]
    })

    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload).json()
    data = response["data"]
    tickers = []
    prices = []

    for i in data:
        tickers.append(i['d'][0])
        prices.append(i['d'][3])

    ticker_to_price = []

    for i in range(len(tickers)):
        obj = {
            'Ticker': tickers[i],
            'Price': prices[i]
        }
        ticker_to_price.append(obj)

    return ticker_to_price

def get_random_stocks(num_stocks=30):
    """
    Returns an array of randomly selected stock below $5. Allows a definition of the amount 
    of stocks. 
    """

    stocks = get_stocks()

    filtered_stocks = [obj for obj in stocks if obj['Price'] < 5 and obj['Price'] != 1e6]

    choices = random.sample(filtered_stocks, k=min(num_stocks, len(filtered_stocks)))

    final_choices = [obj['Ticker'] for obj in choices]

    return final_choices



