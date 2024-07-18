import requests
import json
import csv

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

with open('stocks.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(['Ticker', 'Price'])

    for ticker, price in zip(tickers, prices):
        writer.writerow([ticker, price])