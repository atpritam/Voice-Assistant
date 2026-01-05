"""
Test Dataset for Intent Recognition
400 queries for comprehensive evaluation

Dataset diversity score: 0.970
"""

# === NORMAL TEST DATA ===
NORMAL_TEST_DATASET = [
    # Order intents (65 queries)
    ("I want to order a large pepperoni pizza", "order"),
    ("Can I get two medium pizzas with extra cheese", "order"),
    ("I'd like to purchase a pizza", "order"),
    ("Place an order for delivery", "order"),
    ("I want a small pizza", "order"),
    ("Can you take my order?", "order"),
    ("Order a large margherita", "order"),
    ("Can I customize a pizza with half and half", "menu_inquiry"),
    ("Can I place an order", "order"),
    ("Start a new order please", "order"),
    ("Make an order for pickup", "order"),
    ("I want a thin crust pepperoni with olives", "order"),
    ("Order two medium pizzas, one veggie one meat", "order"),
    ("lemme get a pizza", "order"),
    ("Can I get a medium veggie pizza without onions?", "order"),
    ("I'll have a medium pizza half pepperoni half cheese.", "order"),
    ("Can I order a white pizza with spinach?", "order"),
    ("I want a BBQ chicken pizza large size.", "order"),
    ("I need three large pizzas for a party", "order"),
    ("Can I get a stuffed crust supreme", "order"),
    ("I want to order a family size Hawaiian", "order"),
    ("Get me a meat lovers with extra bacon", "order"),
    ("I need three medium pizzas", "order"),
    ("Let me get a veggie pizza", "order"),
    ("I'll order the meat lovers", "order"),
    ("Can I get a thin crust pizza?", "order"),
    ("I want thick crust please", "order"),
    ("Order a family size pizza", "order"),
    ("I'd like a gluten free pizza", "order"),
    ("Can I order a vegan pizza?", "order"),
    ("I'll get the same as last time", "order"),
    ("Order me a stuffed crust", "order"),
    ("I need a large with extra cheese", "order"),
    ("I want to order some food", "order"),
    ("Let me place an order please", "order"),
    ("I'm looking to order", "order"),
    ("I'd like to order something", "order"),
    ("I want a pizza for delivery", "order"),
    ("I'll take a veggie deluxe no mushrooms", "order"),
    ("Can I order a pizza with chicken and peppers", "order"),
    ("I want a personal pan pizza with sausage", "order"),
    ("Order a thick crust margherita for me", "order"),
    ("I need a gluten free pizza with vegetables", "order"),
    ("Can you make me two small cheese pizzas", "order"),
    ("I want to get four medium pizzas", "order"),
    ("Let me order a pizza for pickup tonight", "order"),
    ("I'd like a large pizza with mushrooms and olives", "order"),
    ("Can I get a pizza with no cheese", "order"),
    ("I want a thin crust with extra sauce", "order"),
    ("Order me a pizza half veggie half meat", "order"),
    ("I'll get the large special", "order"),
    ("Can I buy a pizza with pineapple and ham", "order"),
    ("I need to order pizza for delivery to my office", "order"),
    ("Get me your biggest pizza with everything", "order"),
    ("I want a medium with bacon and jalapenos", "order"),
    ("Can I order a pizza with light cheese", "order"),
    ("I'd like to get a fresh pizza made", "order"),
    ("Let me order the combo deal", "order"),
    ("I want a large with pepperoni and sausage", "order"),
    ("Can you make me a custom pizza", "order"),
    ("I need to place a large order", "order"),
    ("I want to buy two pizzas and some sides", "order"),
    ("Order a pizza with extra toppings", "order"),
    ("I'd like a medium pizza well done", "order"),
    ("Can I get a square cut pizza", "order"),
    ("I'd like to order for pickup tomorrow", "order"),

    # Complaint intents (60 queries)
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
    ("The toppings are wrong", "complaint"),
    ("Let me talk to your supervisor", "complaint"),
    ("The box was damaged when it arrived", "complaint"),
    ("My pizza has the wrong crust", "complaint"),
    ("This pizza tastes awful", "complaint"),
    ("The cheese is burnt and disgusting", "complaint"),
    ("I want a refund, i dont like your service", "complaint"),
    ("You forgot my drinks", "complaint"),
    ("This is not what I ordered at all", "complaint"),
    ("The pizza is undercooked", "complaint"),
    ("I found a hair in my pizza", "complaint"),
    ("This is the worst service ever", "complaint"),
    ("My order never showed up", "complaint"),
    ("I want to escalate this issue", "complaint"),
    ("I need to talk to someone about my order", "complaint"),
    ("My order is completely wrong", "complaint"),
    ("The pizza arrived cold", "complaint"),
    ("This tastes horrible", "complaint"),
    ("The service is unacceptable", "complaint"),
    ("I need compensation", "complaint"),
    ("This is ridiculous", "complaint"),
    ("The pizza is overcooked", "complaint"),
    ("I want a refund and to speak to a manager", "complaint"),
    ("My pizza is cold and I want my money back", "complaint"),
    ("Driver still hasn't shown up", "complaint"),
    ("I'm very disappointed with my pizza", "complaint"),
    ("This is absolutely unacceptable", "complaint"),
    ("I'm so angry about this order", "complaint"),
    ("This is disgusting and I want a refund", "complaint"),
    ("Worst pizza I've ever had", "complaint"),
    ("The ingredients taste stale", "complaint"),
    ("This is ridiculous, I want a manager", "complaint"),
    ("Absolutely terrible experience", "complaint"),
    ("The driver was very rude", "complaint"),
    ("I want my money back immediately", "complaint"),
    ("You charged me twice", "complaint"),
    ("The crust is hard as a rock", "complaint"),
    ("I'm missing half my order", "complaint"),
    ("This is unacceptable quality", "complaint"),
    ("The sauce tastes bad", "complaint"),
    ("My pizza was cut incorrectly", "complaint"),
    ("The delivery address was wrong", "complaint"),
    ("I ordered large but got medium", "complaint"),
    ("The toppings are missing", "complaint"),
    ("This is ice cold", "complaint"),
    ("I need compensation for this", "complaint"),

    # Hours/Location intents (49 queries)
    ("What time do you close", "hours_location"),
    ("When are you open", "hours_location"),
    ("What's your address", "hours_location"),
    ("Where are you located", "hours_location"),
    ("What are your business hours", "hours_location"),
    ("Are you open today", "hours_location"),
    ("Where is the nearest store", "hours_location"),
    ("What time do you open tomorrow", "hours_location"),
    ("Store location", "hours_location"),
    ("Are you still open?", "hours_location"),
    ("How do I get directions to your store", "hours_location"),
    ("What street are you on?", "hours_location"),
    ("Give me directions", "hours_location"),
    ("What are your delivery hours?", "hours_location"),
    ("When do you stop taking orders?", "hours_location"),
    ("Are you open for lunch?", "hours_location"),
    ("Do you open early?", "hours_location"),
    ("What neighborhood are you in?", "hours_location"),
    ("Is there a location near me?", "hours_location"),
    ("Where is your shop?", "hours_location"),
    ("What's the closest pizza place?", "hours_location"),
    ("How many locations do you have?", "hours_location"),
    ("Can I come in now?", "hours_location"),
    ("Are you accepting orders right now?", "hours_location"),
    ("What are your pickup hours?", "hours_location"),
    ("When does delivery end?", "hours_location"),
    ("Are you open on weekends?", "hours_location"),
    ("What time do you start serving?", "hours_location"),
    ("Where can I find you?", "hours_location"),
    ("Where are you and when are you open", "hours_location"),
    ("where you at", "hours_location"),
    ("What time do you open today", "hours_location"),
    ("What time does the kitchen close?", "hours_location"),
    ("What is your street number?", "hours_location"),
    ("Are you open on Sundays", "hours_location"),
    ("When do you close on weekends", "hours_location"),
    ("Do you have late night hours", "hours_location"),
    ("Where exactly is your restaurant", "hours_location"),
    ("How late are you open tonight", "hours_location"),
    ("What are your hours for pickup", "hours_location"),
    ("Are you open now", "hours_location"),
    ("When do you open in the morning", "hours_location"),
    ("Are you open on holidays", "hours_location"),
    ("How do I get to your location", "hours_location"),
    ("Are you open on Christmas", "hours_location"),
    ("What time should I come by", "hours_location"),
    ("Where you guys located at?", "hours_location"),
    ("Are you closed on Mondays", "hours_location"),
    ("What time do you stop taking orders", "hours_location"),

    # Menu inquiry intents (58 queries)
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
    ("Any deals today?", "menu_inquiry"),
    ("What drinks do you offer?", "menu_inquiry"),
    ("Do you have stuffed crust", "menu_inquiry"),
    ("What crust types are available", "menu_inquiry"),
    ("What's your best seller?", "menu_inquiry"),
    ("What comes on a Hawaiian pizza", "menu_inquiry"),
    ("Do you have meat lovers", "menu_inquiry"),
    ("What vegetables can I add", "menu_inquiry"),
    ("Do you offer thin crust", "menu_inquiry"),
    ("What's your cheapest pizza", "menu_inquiry"),
    ("Do you have any combo deals", "menu_inquiry"),
    ("What sides do you have", "menu_inquiry"),
    ("Can I see a list of toppings?", "menu_inquiry"),
    ("What are your signature pizzas?", "menu_inquiry"),
    ("Do you have a BBQ pizza?", "menu_inquiry"),
    ("What's your most popular pizza?", "menu_inquiry"),
    ("Do you serve pasta?", "menu_inquiry"),
    ("Do you have appetizers?", "menu_inquiry"),
    ("What desserts are available?", "menu_inquiry"),
    ("What's the price range?", "menu_inquiry"),
    ("Can I customize my pizza?", "menu_inquiry"),
    ("Do you have dairy-free cheese?", "menu_inquiry"),
    ("What are the size dimensions", "menu_inquiry"),
    ("Do you have promotions?", "menu_inquiry"),
    ("What cheese varieties do you use?", "menu_inquiry"),
    ("What's your Hawaiian pizza made of?", "menu_inquiry"),
    ("Tell me about your supreme pizza", "menu_inquiry"),
    ("What's in a margherita?", "menu_inquiry"),
    ("Describe your meat lovers pizza", "menu_inquiry"),
    ("What goes on a veggie pizza?", "menu_inquiry"),
    ("Do you have thin crust?", "menu_inquiry"),
    ("What drinks do you have?", "menu_inquiry"),
    ("Do you sell salads", "menu_inquiry"),
    ("What's in the meat lovers pizza", "menu_inquiry"),
    ("How much is extra cheese", "menu_inquiry"),
    ("Do you have dairy free options", "menu_inquiry"),
    ("What sauces do you offer", "menu_inquiry"),
    ("Can I see nutritional information", "menu_inquiry"),
    ("What's the price difference between sizes", "menu_inquiry"),
    ("Do you have wings", "menu_inquiry"),
    ("Do you make calzones", "menu_inquiry"),
    ("What's included in the veggie pizza", "menu_inquiry"),
    ("How many toppings can I choose", "menu_inquiry"),

    # Delivery intents (43 queries)
    ("Where is my order", "delivery"),
    ("Can you track my delivery", "delivery"),
    ("What's the status of my pizza", "delivery"),
    ("How much is the delivery fee", "delivery"),
    ("When will my order arrive", "delivery"),
    ("Track my order please", "delivery"),
    ("Delivery time estimate", "delivery"),
    ("How much longer will it take", "delivery"),
    ("Can I update my delivery address", "delivery"),
    ("Order status check", "delivery"),
    ("How long for delivery", "delivery"),
    ("Do you deliver to my area", "delivery"),
    ("Do you charge for delivery", "delivery"),
    ("Do you deliver to this address?", "delivery"),
    ("still waiting on my pizza", "delivery"),
    ("Can I track the driver", "delivery"),
    ("How far do you deliver", "delivery"),
    ("Is there a delivery charge", "delivery"),
    ("When will the driver arrive", "delivery"),
    ("Can you give me an ETA", "delivery"),
    ("How long until my order arrives?", "delivery"),
    ("What's the estimated delivery time?", "delivery"),
    ("Can you check on my delivery?", "delivery"),
    ("Is my order on the way?", "delivery"),
    ("I ordered an hour ago where is my food?", "delivery"),
    ("Where is my order and can I track it?", "delivery"),
    ("Do you do late night delivery?", "delivery"),
    ("How much longer for delivery?", "delivery"),
    ("What's my order status?", "delivery"),
    ("Is delivery free over thirty dollars?", "delivery"),
    ("What's your delivery range?", "delivery"),
    ("What's the delivery time estimate?", "delivery"),
    ("I know you're busy but where is my order?", "delivery"),
    ("Is contactless delivery available", "delivery"),
    ("Do you have real-time tracking?", "delivery"),
    ("Can I see where the driver is?", "delivery"),
    ("Do you offer free delivery", "delivery"),
    ("What's your delivery radius", "delivery"),
    ("How long does delivery usually take", "delivery"),
    ("What's the delivery wait time", "delivery"),
    ("Do you deliver during lunch", "delivery"),
    ("Can I get contactless delivery", "delivery"),
    ("How much longer until my pizza gets here", "delivery"),
    ("What's the minimum order value", "delivery"),

    # General intents (20 queries)
    ("Hello", "general"),
    ("Hi there", "general"),
    ("Thanks for your help", "general"),
    ("can I reserve a table?", "general"),
    ("Goodbye", "general"),
    ("Sounds good", "general"),
    ("Sure", "general"),
    ("Yes please", "general"),
    ("Thank you very much", "general"),
    ("Good evening", "general"),
    ("Hey", "general"),
    ("Okay thanks", "general"),
    ("No problem", "general"),
    ("Alright", "general"),
    ("Got it", "general"),
    ("I understand", "general"),
    ("Amazing service as always", "general"),
    ("Perfect", "general"),
    ("That works", "general"),
    ("what a good day it is today", "general"),
]


# === EDGE CASES DATA ===
EDGE_CASES_DATASET = [
    # Ambiguous (10 queries)
    ("What can I get", "menu_inquiry"),
    ("Tell me about your pizza", "menu_inquiry"),
    ("What do you recommend", "menu_inquiry"),
    ("I'm hungry", "order"),
    ("Help me decide", "menu_inquiry"),
    ("What are your recommended good options?", "menu_inquiry"),
    ("I'm looking for something", "menu_inquiry"),
    ("Can you help me", "general"),
    ("Hey there, I'm looking to get some food", "order"),
    ("How much would it cost to get my cold pizza replaced?", "complaint"),

    # Long queries (8 queries) - should recognize dominant intent
    ("Hi there I was wondering if you could help me because I ordered a large pepperoni pizza about two hours ago and it still hasn't arrived yet", "complaint"),
    ("So I'm looking at your menu and I'm trying to figure out what the best deal is and also I want to know if you have any vegetarian options available", "menu_inquiry"),
    ("I placed an order earlier today around noon and I specifically asked for no onions but when the pizza arrived it had onions all over it and also the crust was burnt", "complaint"),
    ("Good afternoon I was just calling to check if you're still open because I'd like to place an order but I'm not sure what time you close on Saturdays", "hours_location"),
    ("I'm having a party tonight and I need to order several large pizzas but I want to know what your best deals are and if you can deliver to my area", "menu_inquiry"),
    ("My friend ordered from you last week and said it was amazing so I want to try it but I need to know what you have that's vegetarian and how much it costs", "menu_inquiry"),
    ("I tried calling earlier but no one answered and now I'm worried my order didn't go through can you check if there's an delivery under my name", "delivery"),
    ("I'm looking at your menu online but I can't find any information about whether you have gluten free options or what the price difference would be", "menu_inquiry"),

    # Multiple intents - should recognize dominant intent (17 queries)
    ("I want to order but first tell me your hours", "hours_location"),
    ("Can I get a refund and also where is my order", "complaint"),
    ("The driver went to the wrong address", "complaint"),
    ("I saw on the menu you have gluten free but can I order that for delivery", "order"),
    ("How much does delivery cost and when will it arrive", "delivery"),
    ("What toppings do you have and can I order now", "menu_inquiry"),
    ("My order is late and I want to know where it is", "delivery"),
    ("I want to complain but also need to know your hours", "complaint"),
    ("I need a refund but first where's my pizza", "delivery"),
    ("My order is wrong but I want to order again", "order"),
    ("The app says delivered but I don't have it", "complaint"),
    ("I need to place an order but what sizes do you have?", "menu_inquiry"),
    ("I want to order but my last delivery was late", "complaint"),
    ("Can I order a large pepperoni and how long will it take?", "order"),
    ("I want to order but is delivery free?", "delivery"),
    ("Can I order now or are you closed?", "hours_location"),
    ("I don't want delivery I want pickup within next hour", "order"),

    # Negative queries (5 queries)
    ("I don't want pepperoni what else do you have", "menu_inquiry"),
    ("I can't eat gluten what can I order", "menu_inquiry"),
    ("I'm not a fan of tomato sauce what are my options", "menu_inquiry"),
    ("I don't like thick crust do you have thin", "menu_inquiry"),
    ("No mushrooms please what else is there", "menu_inquiry"),

    # Sarcasm (10 queries)
    ("Oh great another wrong order", "complaint"),
    ("Just what I needed, cold pizza again", "complaint"),
    ("Wonderful, an hour late as usual", "complaint"),
    ("Perfect timing, I only waited forever", "complaint"),
    ("Love getting the wrong toppings every time", "complaint"),
    ("Exactly what I asked for, NOT", "complaint"),
    ("So happy with this burnt pizza", "complaint"),
    ("Great service, been waiting two hours", "complaint"),
    ("Love how my pizza never showed up", "complaint"),
    ("The delivery took forever", "complaint"),

    # Context dependent (11 queries)
    ("What about the large one", "menu_inquiry"),
    ("I will take that one", "order"),
    ("Get me my regular", "order"),
    ("Same order as before", "order"),
    ("The usual please", "order"),
    ("Make it a large", "order"),
    ("Add extra cheese to that", "order"),
    ("Change that to pickup", "order"),
    ("Make it two", "order"),
    ("Same as last time", "order"),
    ("I want my usual", "order"),

    # Very Short Queries (13 queries)
    ("hours", "hours_location"),
    ("price", "menu_inquiry"),
    ("order", "order"),
    ("menu", "menu_inquiry"),
    ("delivery", "delivery"),
    ("location", "hours_location"),
    ("open", "hours_location"),
    ("refund", "complaint"),
    ("track", "delivery"),
    ("specials", "menu_inquiry"),
    ("pepperoni large", "order"),
    ("hawaiian pizza please", "order"),
    ("Pizza?", "order"),

    # Unusual Phrasing (7 queries)
    ("Inquiring about your operational schedule", "hours_location"),
    ("Seeking information regarding menu items", "menu_inquiry"),
    ("Expressing dissatisfaction with service", "complaint"),
    ("Requesting status update on my purchase", "delivery"),
    ("What is your geographical position", "hours_location"),
    ("I require nutritional options", "menu_inquiry"),
    ("Demanding monetary reimbursement", "complaint"),

    # Typos and misspellings (7 queries)
    ("I want to oder a pizza", "order"),
    ("Whats on the menue", "menu_inquiry"),
    ("My oder is late", "complaint"),
    ("Whre are you located", "hours_location"),
    ("Do you have vegitarian options", "menu_inquiry"),
    ("What time do you clse", "hours_location"),
    ("Can I trak my order", "delivery"),

    # Casual/Slang (11 queries)
    ("yo whats good", "general"),
    ("gimme a pizza", "order"),
    ("where u guys at", "hours_location"),
    ("how much for a pie", "menu_inquiry"),
    ("my food aint here yet", "complaint"),
    ("this pizza sucks", "complaint"),
    ("u guys open", "hours_location"),
    ("where's my food at", "delivery"),
    ("send it over to my place", "delivery"),
    ("whats the damage for a large", "menu_inquiry"),
    ("hold the onions", "order"),

    # Questions about questions (6 queries)
    ("Can I ask about the menu", "menu_inquiry"),
    ("I have a question about delivery", "delivery"),
    ("Can you tell me your prices", "menu_inquiry"),
    ("I want to know about your hours", "hours_location"),
    ("Can I inquire about an order I placed for delivery?", "delivery"),
    ("I'd like to ask about your specials", "menu_inquiry"),
]


def get_test_dataset(include_edge_cases=False):
    """Return dataset, optionally with edge cases"""
    if include_edge_cases:
        return NORMAL_TEST_DATASET + EDGE_CASES_DATASET
    return NORMAL_TEST_DATASET


def check_duplicates():
    """
    Check if dataset contains duplicate queries.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from utils.text_processor import TextProcessor

    text_processor = TextProcessor()
    dataset = NORMAL_TEST_DATASET + EDGE_CASES_DATASET
    seen = {}
    duplicates = []

    for query, intent in dataset:
        normalized = text_processor.normalize(query)

        if normalized in seen:
            duplicates.append((query, intent, seen[normalized]))
        else:
            seen[normalized] = (query, intent)

    return len(duplicates) > 0, duplicates


def calculate_diversity_score():
    """Calculate dataset diversity score (0-1, higher is better)"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        queries = [q for q, _ in NORMAL_TEST_DATASET + EDGE_CASES_DATASET]
        vectors = TfidfVectorizer(ngram_range=(1, 2), max_features=500).fit_transform(queries)
        similarities = cosine_similarity(vectors)

        return round(1 - np.mean(similarities[np.triu_indices_from(similarities, k=1)]), 4)
    except ImportError:
        return None


def get_dataset_statistics():
    """Get comprehensive statistics about the dataset"""
    normal_dataset = NORMAL_TEST_DATASET
    edge_cases = EDGE_CASES_DATASET
    full_dataset = normal_dataset + edge_cases

    # Count intents in each dataset
    normal_counts = {}
    edge_counts = {}
    total_counts = {}

    for _, intent in normal_dataset:
        normal_counts[intent] = normal_counts.get(intent, 0) + 1
        total_counts[intent] = total_counts.get(intent, 0) + 1

    for _, intent in edge_cases:
        edge_counts[intent] = edge_counts.get(intent, 0) + 1
        total_counts[intent] = total_counts.get(intent, 0) + 1

    return {
        'total_queries': len(full_dataset),
        'normal_queries': len(normal_dataset),
        'edge_case_queries': len(edge_cases),
        'unique_intents': len(total_counts),
        'normal_intent_distribution': normal_counts,
        'edge_case_intent_distribution': edge_counts,
        'total_intent_distribution': total_counts
    }


if __name__ == '__main__':
    import json
    stats = get_dataset_statistics()
    print()
    print(json.dumps(stats, indent=4))

    has_dupes, dupes = check_duplicates()
    print("\nDuplicates found:", has_dupes)

    if has_dupes:
        print(f"\nFound {len(dupes)} duplicate(s):")
        for query, intent, original in dupes:
            print(f"  - '{query}' ({intent})")
            print(f"    Duplicate of: '{original[0]}' ({original[1]})")

    diversity_score = calculate_diversity_score()
    print(f"\nDataset diversity score: {diversity_score}")