from typing import List, Dict
from collections import Counter

# Define possible decision types
DECISION_TYPES = ["buy", "sell", "hold", "close"]

def aggregate_decisions(model_results: List[Dict]) -> Dict:
    """
    Aggregate decisions from multiple models and reach a consensus.
    
    :param model_results: List of dictionaries containing model predictions
    :return: Dictionary with the consensus decision and additional metadata
    """
    # Extract decisions from model results
    decisions = [result['prediction']['decision'] for result in model_results if 'prediction' in result]
    
    # Count the occurrences of each decision
    decision_counts = Counter(decisions)
    
    # Find the most common decision (the mode)
    if decision_counts:
        consensus_decision = decision_counts.most_common(1)[0][0]
    else:
        consensus_decision = "hold"  # Default to 'hold' if no valid decisions
    
    # Calculate confidence as the proportion of models that agree with the consensus
    total_valid_decisions = len(decisions)
    confidence = decision_counts[consensus_decision] / total_valid_decisions if total_valid_decisions > 0 else 0
    
    # Calculate average certainty across all models
    certainties = [result['prediction']['certainty'] for result in model_results 
                   if 'prediction' in result and 'certainty' in result['prediction']]
    avg_certainty = sum(certainties) / len(certainties) if certainties else 0
    
    # Prepare detailed results for each decision type
    detailed_results = {decision: {
        'count': decision_counts.get(decision, 0),
        'percentage': (decision_counts.get(decision, 0) / total_valid_decisions * 100) if total_valid_decisions > 0 else 0
    } for decision in DECISION_TYPES}
    
    # Prepare the final aggregated result
    aggregated_result = {
        'consensus_decision': consensus_decision,
        'confidence': confidence,
        'average_certainty': avg_certainty,
        'total_models': len(model_results),
        'valid_decisions': total_valid_decisions,
        'detailed_results': detailed_results
    }
    
    return aggregated_result

def should_execute_trade(aggregated_result: Dict, confidence_threshold: float = 0.6, certainty_threshold: float = 0.7) -> bool:
    """
    Determine whether a trade should be executed based on the aggregated result.
    
    :param aggregated_result: The result from aggregate_decisions
    :param confidence_threshold: Minimum confidence required to execute a trade
    :param certainty_threshold: Minimum average certainty required to execute a trade
    :return: Boolean indicating whether to execute the trade
    """
    return (aggregated_result['confidence'] >= confidence_threshold and 
            aggregated_result['average_certainty'] >= certainty_threshold and 
            aggregated_result['consensus_decision'] in ['buy', 'sell'])

# You can test the module by running it directly
if __name__ == "__main__":
    # Mock model results for testing
    mock_results = [
        {'model_name': 'model1', 'prediction': {'decision': 'buy', 'certainty': 0.8}},
        {'model_name': 'model2', 'prediction': {'decision': 'buy', 'certainty': 0.7}},
        {'model_name': 'model3', 'prediction': {'decision': 'sell', 'certainty': 0.6}},
        {'model_name': 'model4', 'prediction': {'decision': 'hold', 'certainty': 0.5}},
        {'model_name': 'model5', 'prediction': {'decision': 'buy', 'certainty': 0.9}},
        {'model_name': 'model6', 'error': 'Model failed to produce a prediction'},
        {'model_name': 'model7', 'prediction': {'decision': 'buy', 'certainty': 0.75}},
    ]
    
    aggregated_result = aggregate_decisions(mock_results)
    print("Aggregated Result:")
    for key, value in aggregated_result.items():
        print(f"{key}: {value}")
    
    should_trade = should_execute_trade(aggregated_result)
    print(f"\nShould execute trade: {should_trade}")