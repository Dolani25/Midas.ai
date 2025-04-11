from typing import Dict, List, Union
import re

def validate_twitter_data(twitter_data: Dict) -> Dict:
    """
    Validate Twitter data.
    
    :param twitter_data: Dictionary containing Twitter data
    :return: Dictionary with validation results
    """
    validation_result = {
        "is_valid": True,
        "errors": [],
        "warnings": []
    }

    # Check if required fields are present
    required_fields = ["token", "tweets", "sentiment"]
    for field in required_fields:
        if field not in twitter_data:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Missing required field: {field}")

    # Check tweet count
    if "tweets" in twitter_data:
        tweet_count = len(twitter_data["tweets"])
        if tweet_count < 10:
            validation_result["warnings"].append(f"Low tweet count: {tweet_count}")
        if tweet_count == 0:
            validation_result["is_valid"] = False
            validation_result["errors"].append("No tweets found")

    # Check sentiment
    if "sentiment" in twitter_data:
        sentiment = twitter_data["sentiment"]
        if not -1 <= sentiment["average_sentiment"] <= 1:
            validation_result["errors"].append("Invalid sentiment value")
            validation_result["is_valid"] = False

    return validation_result

def validate_solana_data(solana_data: Dict) -> Dict:
    """
    Validate Solana marketplace data.
    
    :param solana_data: Dictionary containing Solana data
    :return: Dictionary with validation results
    """
    validation_result = {
        "is_valid": True,
        "errors": [],
        "warnings": []
    }

    # Check if required fields are present
    required_fields = ["token_data", "market_data", "historical_data"]
    for field in required_fields:
        if field not in solana_data:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Missing required field: {field}")

    # Validate token data
    if "token_data" in solana_data:
        token_data = solana_data["token_data"]
        if "current_price" not in token_data or token_data["current_price"] <= 0:
            validation_result["errors"].append("Invalid or missing current price")
            validation_result["is_valid"] = False
        if "market_cap" not in token_data or token_data["market_cap"] < 0:
            validation_result["errors"].append("Invalid or missing market cap")
            validation_result["is_valid"] = False

    # Validate historical data
    if "historical_data" in solana_data:
        historical_data = solana_data["historical_data"]
        if len(historical_data) < 7:
            validation_result["warnings"].append("Insufficient historical data")

    return validation_result

def check_for_scam_indicators(twitter_data: Dict, solana_data: Dict) -> Dict:
    """
    Check for potential scam or rugpull indicators.
    
    :param twitter_data: Dictionary containing Twitter data
    :param solana_data: Dictionary containing Solana data
    :return: Dictionary with scam check results
    """
    scam_check_result = {
        "is_potential_scam": False,
        "indicators": []
    }

    # Check for suspicious Twitter activity
    if "tweets" in twitter_data:
        tweets = twitter_data["tweets"]
        if len(tweets) > 0:
            # Check for a high proportion of new accounts
            new_account_ratio = sum(1 for tweet in tweets if tweet["user_age_days"] < 30) / len(tweets)
            if new_account_ratio > 0.5:
                scam_check_result["indicators"].append("High proportion of tweets from new accounts")

            # Check for repetitive content
            content_set = set(tweet["text"] for tweet in tweets)
            if len(content_set) / len(tweets) < 0.3:
                scam_check_result["indicators"].append("High proportion of repetitive tweet content")

    # Check Solana data for suspicious activity
    if "token_data" in solana_data and "historical_data" in solana_data:
        token_data = solana_data["token_data"]
        historical_data = solana_data["historical_data"]

        # Check for sudden price spikes
        if len(historical_data) > 1:
            price_change = (token_data["current_price"] - historical_data[0]["price"]) / historical_data[0]["price"]
            if price_change > 5:  # 500% increase
                scam_check_result["indicators"].append("Suspicious rapid price increase")

        # Check for very low liquidity
        if "liquidity" in token_data and token_data["liquidity"] < 1000:  # Example threshold
            scam_check_result["indicators"].append("Very low liquidity")

        # Check for anonymous team
        if "team_info" in token_data and not token_data["team_info"]:
            scam_check_result["indicators"].append("Anonymous or missing team information")

    # Set the overall scam potential based on the number of indicators
    scam_check_result["is_potential_scam"] = len(scam_check_result["indicators"]) > 2

    return scam_check_result

def validate_data(twitter_data: Dict, solana_data: Dict) -> Dict:
    """
    Validate both Twitter and Solana data, and check for scam indicators.
    
    :param twitter_data: Dictionary containing Twitter data
    :param solana_data: Dictionary containing Solana data
    :return: Dictionary with overall validation results
    """
    twitter_validation = validate_twitter_data(twitter_data)
    solana_validation = validate_solana_data(solana_data)
    scam_check = check_for_scam_indicators(twitter_data, solana_data)

    overall_result = {
        "is_valid": twitter_validation["is_valid"] and solana_validation["is_valid"],
        "twitter_validation": twitter_validation,
        "solana_validation": solana_validation,
        "scam_check": scam_check,
        "can_proceed": False
    }

    # Determine if we can proceed with the analysis
    if overall_result["is_valid"] and not scam_check["is_potential_scam"]:
        overall_result["can_proceed"] = True
    else:
        overall_result["can_proceed"] = False

    return overall_result

# You can test the module by running it directly
if __name__ == "__main__":
    # Example usage with mock data
    mock_twitter_data = {
        "token": "EXAMPLE",
        "tweets": [
            {"text": "Great token!", "user_age_days": 10},
            {"text": "To the moon!", "user_age_days": 100},
            {"text": "Great token!", "user_age_days": 5},
        ],
        "sentiment": {"average_sentiment": 0.8}
    }

    mock_solana_data = {
        "token_data": {
            "current_price": 1.5,
            "market_cap": 1000000,
            "liquidity": 50000,
            "team_info": "John Doe, Jane Smith"
        },
        "market_data": {"total_volume": 10000000},
        "historical_data": [
            {"price": 1.0},
            {"price": 1.1},
            {"price": 1.2},
            {"price": 1.3},
            {"price": 1.4},
        ]
    }

    validation_result = validate_data(mock_twitter_data, mock_solana_data)
    print(validation_result)