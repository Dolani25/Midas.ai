import asyncio
from data import streamTweets, streamNewTokens
from utils import data_validator, model_runner, decision_aggregator, trade_executor
from TradeMonitor import streamTrade, streamTwitterSentiment, EmergencyWithdraw
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
latest_twitter_data = None
latest_solana_data = None
open_trades = {}  # Dictionary to keep track of open trades

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

async def monitor_trade(token):
    trade_info = open_trades[token]
    purchase_price = trade_info['price']  # Assuming this is available in the trade result
    
    trade_monitor = asyncio.create_task(streamTrade.monitor(token, purchase_price))
    sentiment_monitor = asyncio.create_task(streamTwitterSentiment.monitor(token))
    
    while token in open_trades:
        if await trade_monitor or await sentiment_monitor:
            emergency_sell_result = await EmergencyWithdraw.sell(token)
            
            if emergency_sell_result['status'] == 'success':
                logger.info(f"Emergency sell successful for {token}")
                del open_trades[token]
            else:
                logger.error(f"Emergency sell failed for {token}: {emergency_sell_result['message']}")
            
            break
        await asyncio.sleep(5)  # Check every 5 seconds
    
    trade_monitor.cancel()
    sentiment_monitor.cancel()

async def process_data():
    while True:
        if latest_twitter_data and latest_solana_data:
            try:
                validation_result = data_validator.validate_data(latest_twitter_data, latest_solana_data)
                
                if validation_result["can_proceed"]:
                    model_results = await model_runner.process_and_run_models(latest_twitter_data, latest_solana_data)
                    aggregated_result = decision_aggregator.aggregate_decisions(model_results)
                    
                    if decision_aggregator.should_execute_trade(aggregated_result):
                        trade_result = await trade_executor.trigger_trade(
                            aggregated_result['consensus_decision'],
                            latest_solana_data['token_data']['symbol'],
                            aggregated_result['confidence']
                        )
                        logger.info(f"Trade executed: {trade_result}")
                        
                        if trade_result['status'] == 'success':
                            token = trade_result['token']
                            if trade_result['action'] == 'buy':
                                open_trades[token] = trade_result
                                asyncio.create_task(monitor_trade(token))
                            elif trade_result['action'] == 'sell' and token in open_trades:
                                del open_trades[token]
                    else:
                        logger.info("Trade not executed due to low confidence")
                else:
                    logger.warning("Data validation failed, cannot proceed with analysis")
            except Exception as e:
                logger.error(f"Error in data processing: {e}")
        await asyncio.sleep(10)  # Process every 10 seconds



async def main():
    twitter_task = asyncio.create_task(fetch_twitter_data())
    solana_task = asyncio.create_task(fetch_solana_data())
    processing_task = asyncio.create_task(process_data())

    await asyncio.gather(twitter_task, solana_task, processing_task)

if __name__ == "__main__":
    asyncio.run(main())