import aiohttp
import asyncio
from typing import Dict, List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Solana marketplace API endpoint (replace with actual endpoint)
SOLANA_API_ENDPOINT = os.getenv("SOLANA_API_ENDPOINT")
API_KEY = os.getenv("SOLANA_API_KEY")

async def fetch_token_data(session: aiohttp.ClientSession, token_address: str) -> Dict:
    """
    Fetch data for a specific token from the Solana marketplace.
    """
    url = f"{SOLANA_API_ENDPOINT}/token/{token_address}"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    async with session.get(url, headers=headers) as response:
        if response.status == 200:
            return await response.json()
        else:
            raise Exception(f"Failed to fetch data for token {token_address}. Status: {response.status}")

async def fetch_market_data(session: aiohttp.ClientSession) -> Dict:
    """
    Fetch overall market data from the Solana marketplace.
    """
    url = f"{SOLANA_API_ENDPOINT}/market"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    async with session.get(url, headers=headers) as response:
        if response.status == 200:
            return await response.json()
        else:
            raise Exception(f"Failed to fetch market data. Status: {response.status}")

async def fetch_new_tokens(session: aiohttp.ClientSession, limit: int = 10) -> List[Dict]:
    """
    Fetch data for new tokens from the Solana marketplace.
    """
    url = f"{SOLANA_API_ENDPOINT}/new_tokens"
    params = {"limit": limit}
    headers = {"Authorization": f"Bearer {API_KEY}"}

    async with session.get(url, params=params, headers=headers) as response:
        if response.status == 200:
            return await response.json()
        else:
            raise Exception(f"Failed to fetch new tokens. Status: {response.status}")

async def fetch_historical_data(session: aiohttp.ClientSession, token_address: str, days: int = 30) -> List[Dict]:
    """
    Fetch historical data for a specific token.
    """
    url = f"{SOLANA_API_ENDPOINT}/token/{token_address}/history"
    params = {"days": days}
    headers = {"Authorization": f"Bearer {API_KEY}"}

    async with session.get(url, params=params, headers=headers) as response:
        if response.status == 200:
            return await response.json()
        else:
            raise Exception(f"Failed to fetch historical data for token {token_address}. Status: {response.status}")

async def fetch_data(token_address: str = None) -> Dict:
    """
    Main function to fetch Solana data. If a token address is provided,
    it fetches data for that specific token. Otherwise, it fetches general market data.
    """
    async with aiohttp.ClientSession() as session:
        market_data = await fetch_market_data(session)
        new_tokens = await fetch_new_tokens(session)

        if token_address:
            token_data = await fetch_token_data(session, token_address)
            historical_data = await fetch_historical_data(session, token_address)
            return {
                "market_data": market_data,
                "new_tokens": new_tokens,
                "token_data": token_data,
                "historical_data": historical_data
            }
        else:
            return {
                "market_data": market_data,
                "new_tokens": new_tokens
            }

# You can test the module by running it directly
if __name__ == "__main__":
    # Replace with a real Solana token address if you have one
    token_address = "EXAMPLE_TOKEN_ADDRESS"
    result = asyncio.run(fetch_data(token_address))
    print(f"Fetched data for token: {token_address}")
    print(f"Market data: {result['market_data']}")
    print(f"New tokens: {result['new_tokens']}")
    print(f"Token data: {result['token_data']}")
    print(f"Historical data points: {len(result['historical_data'])}")