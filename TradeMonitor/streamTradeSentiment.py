import asyncio
import aiohttp
from typing import Dict, List
import os
from dotenv import load_dotenv
import logging
import time
from textblob import TextBlob

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Twitter API configuration
TWITTER_API_ENDPOINT = os.getenv("TWITTER_API_ENDPOINT")
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")

# Sentiment analysis parameters
SENTIMENT_WINDOW = 100  # Number of recent tweets to consider
NEGATIVE_THRESHOLD = -0.2  # Average sentiment below this is considered negative
PANIC_THRESHOLD = 0.5  # Proportion of negative tweets to trigger panic alert

async def fetch_recent_tweets(session: aiohttp.ClientSession, token: str, count: int = 100) -> List[Dict]:
    """Fetch recent tweets about a specific token."""
    url = f"{TWITTER_API_ENDPOINT}/search/tweets.json"
    headers = {"Authorization": f"Bearer {TWITTER_API_KEY}"}
    params = {
        "q": token,
        "count": count,
        "result_type": "recent",
        "tweet_mode": "extended"
    }

    async with session.get(url, headers=headers, params=params) as response:
        if response.status == 200:
            data = await response.json()
            return data['statuses']
        else:
            logger.error(f"Failed to fetch tweets for {token}. Status: {response.status}")
            return []

def analyze_sentiment(tweet: str) -> float:
    """Analyze the sentiment of a tweet."""
    blob = TextBlob(tweet)
    return blob.sentiment.polarity

async def calculate_average_sentiment(tweets: List[Dict]) -> float:
    """Calculate the average sentiment of a list of tweets."""
    if not tweets:
        return 0

    sentiments = [analyze_sentiment(tweet['full_text']) for tweet in tweets]
    return sum(sentiments) / len(sentiments)

async def monitor(token: str):
    """
    Monitor Twitter sentiment for a specific token.
    Yields True if a panic sell is detected, False otherwise.
    """
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                tweets = await fetch_recent_tweets(session, token, SENTIMENT_WINDOW)
                if not tweets:
                    logger.warning(f"No tweets found for {token}. Continuing monitoring.")
                    yield False
                    await asyncio.sleep(60)  # Wait for 60 seconds before next check
                    continue

                avg_sentiment = await calculate_average_sentiment(tweets)
                negative_tweets = sum(1 for tweet in tweets if analyze_sentiment(tweet['full_text']) < NEGATIVE_THRESHOLD)
                negative_proportion = negative_tweets / len(tweets)

                logger.info(f"Token: {token}, Average Sentiment: {avg_sentiment:.2f}, "
                            f"Negative Tweet Proportion: {negative_proportion:.2f}")

                if avg_sentiment < NEGATIVE_THRESHOLD and negative_proportion > PANIC_THRESHOLD:
                    logger.warning(f"Panic sell alert for {token}! "
                                   f"Average Sentiment: {avg_sentiment:.2f}, "
                                   f"Negative Tweet Proportion: {negative_proportion:.2f}")
                    yield True
                else:
                    yield False

            except Exception as e:
                logger.error(f"Error monitoring Twitter sentiment for {token}: {str(e)}")
                yield False

            await asyncio.sleep(60)  # Check every 60 seconds

# Example usage
async def main():
    token = "EXAMPLE_TOKEN"
    async for should_sell in monitor(token):
        if should_sell:
            print(f"Panic sell detected for {token}! Consider emergency withdrawal.")
            break
        else:
            print(f"No panic detected for {token}. Continuing to monitor.")

if __name__ == "__main__":
    asyncio.run(main())