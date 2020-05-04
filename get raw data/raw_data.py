import requests
r = requests.get('https://api.binance.com/api/v1/klines?symbol=ETHUSDT&interval=1d').json()
print(r)