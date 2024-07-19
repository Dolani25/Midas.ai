import aiohttp
import asyncio
from typing import Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Trading API configuration
TRADING_API_ENDPOINT = os.getenv("TRADING_API_ENDPOINT")
API_KEY = os.getenv("TRADING_API_KEY")
API_SECRET = os.getenv("TRADING_API_SECRET")

class TradeExecutionError(Exception):
    """Custom exception for trade execution errors."""
    pass

async def execute_trade(session: aiohttp.ClientSession, decision: str, token: str, amount: float) -> Dict:
    """
    Execute a trade based on the decision.
    
    :param session: aiohttp ClientSession
    :param decision: 'buy' or 'sell'
    :param token: The token to trade
    :param amount: The amount to trade
    :return: Dictionary with trade execution details
    """
    url = f"{TRADING_API_ENDPOINT}/execute_trade"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "action": decision,
        "token": token,
        "amount": amount
    }

    async with session.post(url, json=payload, headers=headers) as response:
        if response.status == 200:
            return await response.json()
        else:
            error_message = await response.text()
            raise TradeExecutionError(f"Failed to execute trade. Status: {response.status}, Message: {error_message}")

async def close_position(session: aiohttp.ClientSession, token: str) -> Dict:
    """
    Close the position for a given token.
    
    :param session: aiohttp ClientSession
    :param token: The token to close position for
    :return: Dictionary with position closing details
    """
    url = f"{TRADING_API_ENDPOINT}/close_position"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "token": token
    }

    async with session.post(url, json=payload, headers=headers) as response:
        if response.status == 200:
            return await response.json()
        else:
            error_message = await response.text()
            raise TradeExecutionError(f"Failed to close position. Status: {response.status}, Message: {error_message}")

async def get_account_balance(session: aiohttp.ClientSession) -> float:
    """
    Get the current account balance.
    
    :param session: aiohttp ClientSession
    :return: Current account balance
    """
    url = f"{TRADING_API_ENDPOINT}/account_balance"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    async with session.get(url, headers=headers) as response:
        if response.status == 200:
            data = await response.json()
            return data['balance']
        else:
            error_message = await response.text()
            raise TradeExecutionError(f"Failed to get account balance. Status: {response.status}, Message: {error_message}")

async def trigger_trade(decision: str, token: str, confidence: float) -> Dict:
    """
    Trigger a trade based on the decision and confidence.
    
    :param decision: The trading decision ('buy', 'sell', 'hold', or 'close')
    :param token: The token to trade
    :param confidence: The confidence in the decision
    :return: Dictionary with trade execution details
    """
    async with aiohttp.ClientSession() as session:
        try:
            if decision in ['buy', 'sell']:
                # Get account balance
                balance = await get_account_balance(session)
                
                # Calculate trade amount based on confidence and balance
                # This is a simple example - you might want to use a more sophisticated strategy
                trade_amount = balance * 0.1 * confidence  # Trade 10% of balance, adjusted by confidence
                
                # Execute the trade
                result = await execute_trade(session, decision, token, trade_amount)
                return {
                    "status": "success",
                    "action": decision,
                    "token": token,
                    "amount": trade_amount,
                    "details": result
                }
            elif decision == 'close':
                # Close the position
                result = await close_position(session, token)
                return {
                    "status": "success",
                    "action": "close",
                    "token": token,
                    "details": result
                }
            elif decision == 'hold':
                return {
                    "status": "success",
                    "action": "hold",
                    "token": token,
                    "details": "No action taken"
                }
            else:
                raise ValueError(f"Invalid decision: {decision}")
        except TradeExecutionError as e:
            return {
                "status": "error",
                "message": str(e)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"An unexpected error occurred: {str(e)}"
            }

# You can test the module by running it directly
if __name__ == "__main__":
    # Example usage
    decision = "buy"
    token = "SOL"
    confidence = 0.8
    
    result = asyncio.run(trigger_trade(decision, token, confidence))
    print(result)