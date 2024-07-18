from bs4 import BeautifulSoup
import requests


url = 'https://www.tradingview.com/markets/stocks-usa/market-movers-all-stocks/'

response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
stocks = soup.find_all('a', class_='apply-common-tooltip tickerNameBox-GrtoTeat tickerName-GrtoTeat')

for stock in stocks:
    print(stock.text)