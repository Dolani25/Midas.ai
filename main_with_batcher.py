import asyncio
from data import streamTweets, streamNewTokens
from utils import data_validator, model_runner, decision_aggregator, trade_executor
from TradeMonitor import streamTrade, streamTwitterSentiment, EmergencyWithdraw
from OptimizeGas import Batcher
from strategies import StrategyParser
from TelegramBot import TelegramBot
import logging
from solana.rpc.async_api import AsyncClient
from solana.keypair import Keypair
from solana.publickey import PublicKey

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
latest_twitter_data = None
latest_solana_data = None
open_trades = {}  # Dictionary to keep track of open trades

# Solana configuration
RPC_URL = "https://api.mainnet-beta.solana.com"  # Replace with your preferred RPC endpoint
client = AsyncClient(RPC_URL)

# Batcher configuration
MULTISIG_SIGNERS = [
    "signer1_pubkey",
    "signer2_pubkey",
    "signer3_pubkey"
]  # Replace with actual signer public keys
THRESHOLD = 2  # Number of signatures required for multisig

# Initialize TelegramBot
telegram_bot = TelegramBot("YOUR_BOT_TOKEN")

async def fetch_twitter_data():
    global latest_twitter_data
    while True:
        try:
            latest_twitter_data = await streamTweets.fetch_data()
            logger.info("Updated Twitter data")
        except Exception as e:
            logger.error(f"Error fetching Twitter data: {e}")
        await asyncio.sleep(60)  # Fetch every 60 seconds

async def fetch_solana_data():
    global latest_solana_data
    while True:
        try:
            latest_solana_data = await streamNewTokens.fetch_data()
            logger.info("Updated Solana data")
        except Exception as e:
            logger.error(f"Error fetching Solana data: {e}")
        await asyncio.sleep(30)  # Fetch every 30 seconds

async def monitor_trade(token, user_id):
    trade_info = open_trades[token]
    purchase_price = trade_info['price']
    
    trade_monitor = asyncio.create_task(streamTrade.monitor(token, purchase_price))
    sentiment_monitor = asyncio.create_task(streamTwitterSentiment.monitor(token))
    
    user_settings = telegram_bot.get_user_settings(user_id)
    take_profit = user_settings.get('take_profit', None)
    
    while token in open_trades:
        if await trade_monitor or await sentiment_monitor:
            if user_settings['use_ai_trading']:
                sell_decision = await decision_aggregator.should_sell(token)
            else:
                sell_decision = StrategyParser.evaluate_sell_condition(user_settings['strategy'], token)
            
            if sell_decision or (take_profit and trade_info['current_price'] >= take_profit):
                emergency_sell_result = await EmergencyWithdraw.sell(token)
                
                if emergency_sell_result['status'] == 'success':
                    logger.info(f"Emergency sell successful for {token}")
                    del open_trades[token]
                    await telegram_bot.send_message(user_id, f"Emergency sell executed for {token}")
                else:
                    logger.error(f"Emergency sell failed for {token}: {emergency_sell_result['message']}")
                    await telegram_bot.send_message(user_id, f"Emergency sell failed for {token}")
                
                break
        await asyncio.sleep(5)  # Check every 5 seconds
    
    trade_monitor.cancel()
    sentiment_monitor.cancel()

async def process_data(batcher):
    while True:
        if latest_twitter_data and latest_solana_data:
            try:
                validation_result = data_validator.validate_data(latest_twitter_data, latest_solana_data)
                
                if validation_result["can_proceed"]:
                    for user_id, user_settings in telegram_bot.get_all_user_settings().items():
                        if user_settings['use_ai_trading']:
                            model_results = await model_runner.process_and_run_models(latest_twitter_data, latest_solana_data)
                            aggregated_result = decision_aggregator.aggregate_decisions(model_results)
                            
                            if decision_aggregator.should_execute_trade(aggregated_result):
                                await execute_trade(user_id, user_settings, aggregated_result, batcher)
                        else:
                            if StrategyParser.evaluate_buy_condition(user_settings['strategy'], latest_twitter_data, latest_solana_data):
                                await execute_trade(user_id, user_settings, None, batcher)
                else:
                    logger.warning("Data validation failed, cannot proceed with analysis")
            except Exception as e:
                logger.error(f"Error in data processing: {e}")
        await asyncio.sleep(10)  # Process every 10 seconds

async def execute_trade(user_id, user_settings, aggregated_result, batcher):
    token_address = latest_solana_data['token_data']['address']
    amount = calculate_trade_amount(user_settings['wallet_percentage'])
    
    if user_settings.get('ignore_ai_scam_detection', False) or (aggregated_result and not aggregated_result.get('is_scam', False)):
        if user_settings['use_batching']:
            success = await batcher.add_order(PublicKey(user_settings['wallet_address']), token_address, amount)
            if success:
                logger.info(f"Order added to batch for user {user_id}, token {token_address}")
                await telegram_bot.send_message(user_id, f"Order added to batch for token {token_address}")
            else:
                logger.warning(f"Order failed compliance check for user {user_id}, token {token_address}")
                await telegram_bot.send_message(user_id, f"Order failed compliance check for token {token_address}")
        else:
            trade_result = await trade_executor.trigger_trade('buy', token_address, amount)
            if trade_result['status'] == 'success':
                logger.info(f"Trade executed for user {user_id}: {trade_result}")
                await telegram_bot.send_message(user_id, f"Trade executed for token {token_address}")
                open_trades[token_address] = trade_result
                asyncio.create_task(monitor_trade(token_address, user_id))
            else:
                logger.error(f"Trade failed for user {user_id}: {trade_result}")
                await telegram_bot.send_message(user_id, f"Trade failed for token {token_address}")
    else:
        logger.info(f"Trade not executed for user {user_id} due to scam detection")
        await telegram_bot.send_message(user_id, f"Trade not executed for token {token_address} due to scam detection")

def calculate_trade_amount(wallet_percentage):
    # Implement your logic to determine trade amount based on wallet percentage
    # This is a placeholder implementation
    return int(wallet_percentage * 1000000)  # Example: 1 SOL = 1,000,000,000 lamports

async def main():
    # Initialize the Batcher
    batcher = Batcher(RPC_URL, MULTISIG_SIGNERS, THRESHOLD)
    
    # Create tasks
    twitter_task = asyncio.create_task(fetch_twitter_data())
    solana_task = asyncio.create_task(fetch_solana_data())
    processing_task = asyncio.create_task(process_data(batcher))
    batcher_task = asyncio.create_task(batcher.run())
    telegram_task = asyncio.create_task(telegram_bot.run())

    # Run all tasks concurrently
    await asyncio.gather(
        twitter_task,
        solana_task,
        processing_task,
        batcher_task,
        telegram_task
    )

if __name__ == "__main__":
    asyncio.run(main())