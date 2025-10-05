"""
Shared Test Dataset for Intent Recognition
"""

TEST_DATASET = [
    # Order intents - Clear ordering phrases
    ("I want to order a large pepperoni pizza", "order"),
    ("Can I get two medium pizzas with extra cheese", "order"),
    ("I'd like to purchase a pizza", "order"),
    ("Place an order for delivery", "order"),
    ("I want a small pizza", "order"),
    ("Order a large margherita", "order"),
    ("I'd like to buy pizza", "order"),
    ("Can I place an order", "order"),
    ("Start a new order please", "order"),
    ("Make an order for pickup", "order"),

    # Complaint intents - Problems and issues
    ("My pizza was cold when it arrived", "complaint"),
    ("This is terrible, I want a refund", "complaint"),
    ("The order is wrong and I'm very disappointed", "complaint"),
    ("My pizza is burnt", "complaint"),
    ("I have a complaint about my order", "complaint"),
    ("Wrong toppings on my pizza", "complaint"),
    ("Late delivery and cold food", "complaint"),
    ("I want to speak to the manager", "complaint"),
    ("Not satisfied with the service", "complaint"),
    ("Missing items from my order", "complaint"),

    # Hours/Location intents - Business info
    ("What time do you close", "hours_location"),
    ("When are you open", "hours_location"),
    ("What's your address", "hours_location"),
    ("Where are you located", "hours_location"),
    ("What are your business hours", "hours_location"),
    ("Are you open today", "hours_location"),
    ("Opening hours please", "hours_location"),
    ("Where is the nearest store", "hours_location"),
    ("What time do you open tomorrow", "hours_location"),
    ("Store location", "hours_location"),

    # Menu inquiry intents - Menu questions
    ("What toppings do you have", "menu_inquiry"),
    ("What's on your menu", "menu_inquiry"),
    ("How much does a large pizza cost", "menu_inquiry"),
    ("Do you have vegetarian options", "menu_inquiry"),
    ("What sizes do you offer", "menu_inquiry"),
    ("Show me the menu", "menu_inquiry"),
    ("Any gluten free options", "menu_inquiry"),
    ("What are your specialty pizzas", "menu_inquiry"),
    ("Prices for medium pizza", "menu_inquiry"),
    ("Do you have vegan cheese", "menu_inquiry"),

    # Delivery tracking intents - Order status
    ("Where is my order", "delivery"),
    ("Can you track my delivery", "delivery"),
    ("What's the status of my pizza", "delivery"),
    ("How much is the delivery fee", "delivery"),
    ("When will my order arrive", "delivery"),
    ("Track my order please", "delivery"),
    ("Delivery time estimate", "delivery"),
    ("Order status check", "delivery"),
    ("How long for delivery", "delivery"),
    ("Do you deliver to my area", "delivery"),

    # General/greeting intents - Conversational
    ("Hello", "general"),
    ("Hi there", "general"),
    ("Thanks for your help", "general"),
    ("Good morning", "general"),
    ("Goodbye", "general"),
    ("Thank you", "general"),
    ("Hey", "general"),
    ("Okay", "general"),
    ("Sure", "general"),
    ("Yes please", "general"),
]


def get_test_dataset():
    """Return the complete test dataset"""
    return TEST_DATASET


def get_test_dataset_by_intent(intent_name):
    """
    Get test queries for a specific intent

    Args:
        intent_name: Name of intent to filter by

    Returns:
        List of (query, intent) tuples for that intent
    """
    return [(query, intent) for query, intent in TEST_DATASET if intent == intent_name]


def get_dataset_statistics():
    """Get statistics about the test dataset"""
    intent_counts = {}

    for query, intent in TEST_DATASET:
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    return {
        'total_queries': len(TEST_DATASET),
        'unique_intents': len(intent_counts),
        'intent_distribution': intent_counts
    }
