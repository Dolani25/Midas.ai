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

# Trading API configuration
TRADING_API_ENDPOINT = os.getenv("TRADING_API_ENDPOINT")
API_KEY = os.getenv("TRADING_API_KEY")
API_SECRET = os.getenv("TRADING_API_SECRET")

class EmergencyWithdrawError(Exception):
    """Custom exception for emergency withdrawal errors."""
    pass

async def get_token_balance(session: aiohttp.ClientSession, token: str) -> float:
    """Fetch the current balance of a specific token."""
    url = f"{TRADING_API_ENDPOINT}/balance/{token}"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    async with session.get(url, headers=headers) as response:
        if response.status == 200:
            data = await response.json()
            return data['balance']
        else:
            raise EmergencyWithdrawError(f"Failed to fetch balance for {token}. Status: {response.status}")

async def execute_market_sell(session: aiohttp.ClientSession, token: str, amount: float) -> Dict:
    """Execute a market sell order."""
    url = f"{TRADING_API_ENDPOINT}/market_sell"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "token": token,
        "amount": amount
    }

    async with session.post(url, json=payload, headers=headers) as response:
        if response.status == 200:
            return await response.json()
        else:
            raise EmergencyWithdrawError(f"Failed to execute market sell for {token}. Status: {response.status}")

async def sell(token: str) -> Dict:
    """
    Emergency sell function for a specific token.
    Sells the entire balance of the token at market price.
    """
    async with aiohttp.ClientSession() as session:
        try:
            # Get current balance of the token
            balance = await get_token_balance(session, token)
            
            if balance <= 0:
                logger.warning(f"No balance available for {token}. Cannot execute emergency sell.")
                return {"status": "no_balance", "message": f"No balance available for {token}"}

            # Execute market sell for the entire balance
            sell_result = await execute_market_sell(session, token, balance)

            logger.info(f"Emergency sell executed for {token}. Amount: {balance}, Result: {sell_result}")
            
            return {
                "status": "success",
                "token": token,
                "amount_sold": balance,
                "sell_result": sell_result
            }

        except EmergencyWithdrawError as e:
            logger.error(f"Emergency withdrawal failed for {token}: {str(e)}")
            return {"status": "error", "message": str(e)}
        
        except Exception as e:
            logger.error(f"Unexpected error during emergency withdrawal for {token}: {str(e)}")
            return {"status": "error", "message": f"Unexpected error: {str(e)}"}

# Example usage
async def main():
    token = "EXAMPLE_TOKEN"
    result = await sell(token)
    print(f"Emergency Withdrawal Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())