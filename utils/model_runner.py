import asyncio
from typing import List, Dict
import importlib

# Import your models
from models import model1, model2, model3, model4, model5, model6, model7

# List of model modules
MODEL_MODULES = [model1, model2, model3, model4, model5, model6, model7]

async def run_model(model_module, data: Dict) -> Dict:
    """
    Run a single model asynchronously.
    """
    try:
        # Assume each model module has an async 'predict' function
        result = await model_module.predict(data)
        return {
            "model_name": model_module.__name__,
            "prediction": result
        }
    except Exception as e:
        print(f"Error running {model_module.__name__}: {str(e)}")
        return {
            "model_name": model_module.__name__,
            "error": str(e)
        }

async def run_models(data: Dict) -> List[Dict]:
    """
    Run all models concurrently and return their results.
    """
    tasks = [run_model(model, data) for model in MODEL_MODULES]
    results = await asyncio.gather(*tasks)
    return results

def prepare_model_data(twitter_data: Dict, solana_data: Dict) -> Dict:
    """
    Prepare the data for the models by combining Twitter and Solana data.
    """
    # Extract relevant features from Twitter data
    twitter_sentiment = twitter_data['sentiment']['average_sentiment']
    tweet_volume = twitter_data['sentiment']['tweet_count']

    # Extract relevant features from Solana data
    token_price = solana_data['token_data']['current_price']
    market_cap = solana_data['token_data']['market_cap']
    volume_24h = solana_data['token_data']['volume_24h']

    # Extract historical data for technical indicators
    historical_prices = [data_point['price'] for data_point in solana_data['historical_data']]

    # Calculate some basic technical indicators (as an example)
    sma_5 = sum(historical_prices[-5:]) / 5 if len(historical_prices) >= 5 else None
    sma_10 = sum(historical_prices[-10:]) / 10 if len(historical_prices) >= 10 else None

    # Combine all data into a single dictionary
    model_data = {
        'twitter_sentiment': twitter_sentiment,
        'tweet_volume': tweet_volume,
        'token_price': token_price,
        'market_cap': market_cap,
        'volume_24h': volume_24h,
        'sma_5': sma_5,
        'sma_10': sma_10,
        'historical_prices': historical_prices
    }

    return model_data

async def process_and_run_models(twitter_data: Dict, solana_data: Dict) -> List[Dict]:
    """
    Prepare data and run all models.
    """
    model_data = prepare_model_data(twitter_data, solana_data)
    return await run_models(model_data)

# You can test the module by running it directly
if __name__ == "__main__":
    # Mock data for testing
    mock_twitter_data = {
        'sentiment': {'average_sentiment': 0.6, 'tweet_count': 1000}
    }
    mock_solana_data = {
        'token_data': {
            'current_price': 100,
            'market_cap': 1000000,
            'volume_24h': 500000
        },
        'historical_data': [{'price': 95 + i} for i in range(20)]
    }

    results = asyncio.run(process_and_run_models(mock_twitter_data, mock_solana_data))
    for result in results:
        print(f"Model: {result['model_name']}")
        if 'prediction' in result:
            print(f"Prediction: {result['prediction']}")
        else:
            print(f"Error: {result['error']}")
        print()