import asyncio
import aiohttp
from typing import Dict
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Solana API endpoint (replace with actual endpoint)
SOLANA_API_ENDPOINT = os.getenv("SOLANA_API_ENDPOINT")
API_KEY = os.getenv("SOLANA_API_KEY")

# Trade monitoring parameters
MAX_LOSS_PERCENTAGE = 0.05  # 5% maximum loss
CHECK_INTERVAL = 60  # Check every 60 seconds

async def get_current_price(session: aiohttp.ClientSession, token: str) -> float:
    """Fetch the current price of a token."""
    url = f"{SOLANA_API_ENDPOINT}/token/{token}/price"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    async with session.get(url, headers=headers) as response:
        if response.status == 200:
            data = await response.json()
            return data['price']
        else:
            logger.error(f"Failed to fetch price for {token}. Status: {response.status}")
            return None

async def monitor(token: str, purchase_price: float):
    """
    Monitor a trade for a specific token.
    Yields True if the token should be sold, False otherwise.
    """
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                current_price = await get_current_price(session, token)
                
                if current_price is None:
                    logger.warning(f"Unable to fetch price for {token}. Continuing monitoring.")
                    yield False
                    await asyncio.sleep(CHECK_INTERVAL)
                    continue

                price_change = (current_price - purchase_price) / purchase_price

                logger.info(f"Token: {token}, Current Price: {current_price}, " 
                            f"Purchase Price: {purchase_price}, Change: {price_change:.2%}")

                if price_change <= -MAX_LOSS_PERCENTAGE:
                    logger.warning(f"Sell alert for {token}! Loss: {price_change:.2%}")
                    yield True
                else:
                    yield False

            except Exception as e:
                logger.error(f"Error monitoring {token}: {str(e)}")
                yield False

            await asyncio.sleep(CHECK_INTERVAL)

# Example usage
async def main():
    token = "EXAMPLE_TOKEN"
    purchase_price = 100  # Example purchase price

    async for should_sell in monitor(token, purchase_price):
        if should_sell:
            print(f"Selling {token} due to significant loss!")
            break
        else:
            print(f"Continuing to hold {token}")

if __name__ == "__main__":
    asyncio.run(main())