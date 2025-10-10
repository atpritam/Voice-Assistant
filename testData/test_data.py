"""
Shared Test Dataset for Intent Recognition
"""

# === NORMAL TEST DATA ===
NORMAL_TEST_DATASET = [
    # Order intents
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
    ("Yeah I'm ready to order", "order"),
    ("I want a thin crust pepperoni with olives", "order"),
    ("Order two medium pizzas, one veggie one meat", "order"),
    ("lemme get a pizza", "order"),

    # Complaint intents
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
    ("I'm never ordering from you again", "complaint"),
    ("My pizza arrived an hour late", "complaint"),
    ("Extremely disappointed with the quality", "complaint"),

    # Hours/Location intents
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
    ("Where are you and when are you open", "hours_location"),
    ("where you at", "hours_location"),

    # Menu inquiry intents
    ("What toppings do you have", "menu_inquiry"),
    ("What's on your menu", "menu_inquiry"),
    ("How much does a large pizza cost", "menu_inquiry"),
    ("Do you have vegetarian options", "menu_inquiry"),
    ("What sizes do you offer", "menu_inquiry"),
    ("Show me the menu", "menu_inquiry"),
    ("What's in your supreme pizza", "menu_inquiry"),
    ("Any gluten free options", "menu_inquiry"),
    ("Is there a low carb option", "menu_inquiry"),
    ("What specials are running", "menu_inquiry"),
    ("What are your specialty pizzas", "menu_inquiry"),
    ("Prices for medium pizza", "menu_inquiry"),
    ("Do you have vegan cheese", "menu_inquiry"),
    ("What's the biggest pizza you make", "menu_inquiry"),

    # Delivery intents
    ("Where is my order", "delivery"),
    ("Can you track my delivery", "delivery"),
    ("What's the status of my pizza", "delivery"),
    ("How much is the delivery fee", "delivery"),
    ("When will my order arrive", "delivery"),
    ("Track my order please", "delivery"),
    ("Delivery time estimate", "delivery"),
    ("How much longer will it take", "delivery"),
    ("Can I change my delivery address", "delivery"),
    ("Order status check", "delivery"),
    ("How long for delivery", "delivery"),
    ("Do you deliver to my area", "delivery"),
    ("Do you charge for delivery", "delivery"),
    ("The app says delivered but I don't have it", "delivery"),

    # General intents
    ("Hello", "general"),
    ("Hi there", "general"),
    ("Thanks for your help", "general"),
    ("Good morning", "general"),
    ("Goodbye", "general"),
    ("Sounds good", "general"),
    ("Sure", "general"),
    ("Yes please", "general"),
]


# === EDGE CASES DATA ===
EDGE_CASES_DATASET = [
    # Ambiguous
    ("What can I get", "menu_inquiry"),
    ("Tell me about your pizza", "menu_inquiry"),

    # Long queries
    ("Hi there I was wondering if you could help me because I ordered a large pepperoni pizza about two hours ago and it still hasn't arrived yet", "delivery"),
    ("So I'm looking at your menu and I'm trying to figure out what the best deal is and also I want to know if you have any vegetarian options available", "menu_inquiry"),

    # Multiple intents - should recognize top intent
    ("I want to order but first tell me your hours", "hours_location"),
    ("Can I get a refund and also where is my order", "complaint"),

    # Negative queries
    ("I don't want pepperoni what else do you have", "menu_inquiry"),
    ("I don't late delivery again please", "complaint"),

    # Sarcasm
    ("Oh great another wrong order", "complaint"),
    ("Just what I needed, cold pizza again", "complaint"),
    ("Wonderful, an hour late as usual", "complaint"),

    # Context dependent
    ("What about the large one", "menu_inquiry"),
    ("I will take that one", "order"),
    ("Same as before", "order"),
]


def get_test_dataset(include_edge_cases=False):
    """Return dataset, optionally with edge cases"""
    if include_edge_cases:
        return NORMAL_TEST_DATASET + EDGE_CASES_DATASET
    return NORMAL_TEST_DATASET



def get_test_dataset_by_intent(intent_name, include_edge_cases=False):
    dataset = get_test_dataset(include_edge_cases)
    return [(q, i) for q, i in dataset if i == intent_name]


def get_dataset_statistics(include_edge_cases=False):
    dataset = get_test_dataset(include_edge_cases)
    counts = {}
    for _, intent in dataset:
        counts[intent] = counts.get(intent, 0) + 1
    return {
        'total_queries': len(dataset),
        'unique_intents': len(counts),
        'intent_distribution': counts
    }