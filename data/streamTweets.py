from twikit import Client
import asyncio
from typing import List, Dict

async def fetch_tweets(keyword: str, count: int = 100) -> List[Dict]:
  """
  Fetch tweets asynchronously based on a keyword using Twikit.
  """
  async with Client() as client:
      tweets = await client.search.tweets(q=keyword, count=count)
      return [
          {
              "id": tweet.id,
              "text": tweet.full_text,
              "user": tweet.user.screen_name,
              "created_at": tweet.created_at.strftime("%Y-%m-%d %H:%M:%S"),  # Format for consistency
              "retweet_count": tweet.retweet_count,
              "favorite_count": tweet.favorite_count
          }
          for tweet in tweets
      ]

async def analyze_sentiment(tweets: List[Dict]) -> Dict:
    """
    Perform sentiment analysis on the fetched tweets.
    This is a placeholder function - you'd typically use a more sophisticated
    sentiment analysis tool here.
    """
    # This is a very simplistic sentiment analysis for demonstration
    # In a real scenario, you'd use a proper NLP library or API
    positive_words = set(["good", "great", "excellent", "amazing", "wonderful", "fantastic"])
    negative_words = set(["bad", "awful", "terrible", "horrible", "disappointing", "poor"])

    total_sentiment = 0
    for tweet in tweets:
        words = tweet["text"].lower().split()
        tweet_sentiment = sum(1 for word in words if word in positive_words) - \
                          sum(1 for word in words if word in negative_words)
        total_sentiment += tweet_sentiment

    avg_sentiment = total_sentiment / len(tweets) if tweets else 0
    return {
        "average_sentiment": avg_sentiment,
        "tweet_count": len(tweets)
    }

async def fetch_data(token_name: str) -> Dict:
    """
    Main function to fetch Twitter data for a given token.
    """
    tweets = await fetch_tweets(token_name)
    sentiment = await analyze_sentiment(tweets)
    
    return {
        "token": token_name,
        "tweets": tweets,
        "sentiment": sentiment
    }

# You can test the module by running it directly
if __name__ == "__main__":
    token = "Solana"  # or any other token you're interested in
    result = asyncio.run(fetch_data(token))
    print(f"Fetched {len(result['tweets'])} tweets about {token}")
    print(f"Average sentiment: {result['sentiment']['average_sentiment']}")
