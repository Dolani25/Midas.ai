import requests
import pandas as pd
import ta
import csv
import json

api_key = 'ZT0VVKDEMFA9IJPM'



def get_tradable_symbols(api_key, asset_type):
    base_url = 'https://www.alphavantage.co/query'
    function = 'LISTING_STATUS'
    
    
    params = {
        'function': function,
        'apikey': api_key,
        'date': '2024-03-14',  # Adjust the date as needed
    }

    response = requests.get(base_url, params=params)

    try:
        data = response.text

        # Parsing CSV data
        csv_data = csv.DictReader(data.splitlines())

        # Converting CSV data to JSON
        data = json.dumps([row for row in csv_data])

        print(data)
        
        if 'data' in data and assetType in data['data']:
            symbols_data = data['data'][asset_type]
            #symbols = [symbol['symbol'] for symbol in symbols_data]
            trimmed_sym = symbols_data.split(',')[0].strip()
            return trimmed_sym
            
    except Exception as e:
        print("Error:", e)

        print("Failed to retrieve tradable symbols for", asset_type)
        print("Response status code:", response.status_code)
        #print("Response text:", response.text)
        

# Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
api_key = 'ZT0VVKDEMFA9IJPM'





def get_real_time_rsi(symbol):
    # Replace 'YOUR_API_KEY' with your Alpha Vantage API key
    api_key = 'ZT0VVKDEMFA9IJPM'
    base_url = 'https://www.alphavantage.co/query'
    function = 'TIME_SERIES_INTRADAY'
    interval = '1min'  # You can adjust the interval as needed
    base_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey=ZT0VVKDEMFA9IJPM'

    params = {
        'function': function,
        'symbol': symbol,
        'interval': interval,
        'apikey': api_key
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    # Convert data to DataFrame
    if 'Time Series (1min)' in data:
        df = pd.DataFrame(data['Time Series (1min)']).T
        df.index = pd.to_datetime(df.index)

        # Calculate RSI
        close_prices = df['4. close'].astype(float)
        rsi = ta.momentum.RSIIndicator(close=close_prices, window=14).rsi().iloc[-1]

        return rsi


def get_stocks_with_low_real_time_rsi(symbol_list):
    low_rsi_stocks = []

    for symbol in symbol_list:
        real_time_rsi = get_real_time_rsi(symbol)
        
        # Check if the real-time RSI is below 30
        if int(real_time_rsi) <= 30:
            low_rsi_stocks.append(symbol)

    return low_rsi_stocks



# Example usage

symbols_to_check = ["APPL"]

# Get tradable stocks
#stocks = get_tradable_symbols(api_key, 'stocks')


# Get tradable metals
#metals = get_tradable_symbols(api_key, 'metals')


# Get tradable forex
forex = get_tradable_symbols(api_key, 'forex')
#print("Tradable Forex:", forex)

#symbols_to_check.append(stocks)

#symbols_to_check.append(metals)

symbols_to_check.append(forex)


result = get_stocks_with_low_real_time_rsi(symbols_to_check)
print("Stocks with real-time RSI below 30:", result)