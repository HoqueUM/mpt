import requests
import json
import numpy as np
import random

class Stocks():
    def __init__(self):
        self.tickers = None

    def initialize_stocks(self):
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

        self.tickers = ticker_to_price

    def get_random_stocks(self, num_stocks=30, min_price=3, max_price=5):
        """
        Returns an array of randomly selected stock below $5. Allows a definition of the amount 
        of stocks. 
        """
        stocks = self.tickers

        filtered_stocks = [obj for obj in stocks if obj['Price'] <= max_price and  obj['Price'] >= min_price]

        choices = random.sample(filtered_stocks, k=min(num_stocks, len(filtered_stocks)))

        final_choices = [obj['Ticker'] for obj in choices]

        return final_choices



