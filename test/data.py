"""
Test Dataset for Intent Recognition - Hotel Booking Domain
400 queries for comprehensive evaluation

Dataset: Grand Horizon Hotel
Dataset diversity score: 0.9693
"""

# === NORMAL TEST DATA ===
NORMAL_TEST_DATASET = [
    # Booking intents (70 queries)
    ("I want to book a deluxe room for two nights", "booking"),
    ("Can I reserve a suite with a harbor view", "booking"),
    ("I'd like to make a reservation", "booking"),
    ("Book a room for this weekend", "booking"),
    ("I want a standard room", "booking"),
    ("Can you reserve a room for me?", "booking"),
    ("Book a family room for three nights", "booking"),
    ("Can I make a reservation for next week", "booking"),
    ("I'd like to reserve a king bed room", "booking"),
    ("Start a new booking please", "booking"),
    ("Make a reservation for check-in tomorrow", "booking"),
    ("I want an executive suite with a balcony", "booking"),
    ("Reserve two rooms, one deluxe one standard", "booking"),
    ("lemme book a room", "booking"),
    ("Can I get a room with two queen beds?", "booking"),
    ("I'll take a suite for my anniversary", "booking"),
    ("Can I book a pet-friendly room?", "booking"),
    ("I want a room with a jacuzzi", "booking"),
    ("I need three rooms for a conference", "booking"),
    ("Can I reserve an accessible room", "booking"),
    ("I want to book a room near the convention center", "booking"),
    ("Get me a harbor view suite", "booking"),
    ("I need two standard rooms", "booking"),
    ("Let me book a deluxe room", "booking"),
    ("I'll reserve the executive suite", "booking"),
    ("Can I get a room with a king bed?", "booking"),
    ("I want a balcony room please", "booking"),
    ("Book a room for five nights", "booking"),
    ("I'd like a room with breakfast included", "booking"),
    ("Can I reserve a room with parking?", "booking"),
    ("I want my usual room", "booking"),
    ("I'll get the same suite as last time", "booking"),
    ("Reserve me a room with ocean view", "booking"),
    ("I need a large suite with a kitchen", "booking"),
    ("Can I book four rooms?", "booking"),
    ("I want to reserve accommodations", "booking"),
    ("Let me make a reservation please", "booking"),
    ("I'm looking to book a room", "booking"),
    ("I'd like to reserve something for the weekend", "booking"),
    ("I want a room for next month", "booking"),
    ("I'll take a family room with two beds", "booking"),
    ("Can I book a room with city view", "booking"),
    ("I want a standard room for business travel", "booking"),
    ("Reserve a suite for my honeymoon", "booking"),
    ("I need a room with work desk and WiFi", "booking"),
    ("Can you book me two deluxe rooms", "booking"),
    ("I want to reserve four rooms for a group", "booking"),
    ("Let me book a room for check-in Friday", "booking"),
    ("I'd like a suite with separate living room", "booking"),
    ("Can I get a room with a sofa bed", "booking"),
    ("I want a room with a harbor view", "booking"),
    ("Book me a room near the waterfront", "booking"),
    ("I'll get the premium suite", "booking"),
    ("Can I reserve a room with full kitchen", "booking"),
    ("I need to book a room for a week", "booking"),
    ("Get me your best available room", "booking"),
    ("I want a deluxe with king bed and balcony", "booking"),
    ("Can I book a room with late check-out", "booking"),
    ("I'd like to reserve a quiet room", "booking"),
    ("Let me book a room on high floor", "booking"),
    ("I want a suite for business meetings", "booking"),
    ("Can you reserve a corner room", "booking"),
    ("I need to book a long stay", "booking"),
    ("I want to reserve two rooms and parking", "booking"),
    ("Book a room with breakfast and WiFi", "booking"),
    ("I'd like a deluxe room with extra amenities", "booking"),
    ("Can I get a room with harbor and city view", "booking"),
    ("I'd like to book for check-in next Friday", "booking"),

    # Complaint intents (62 queries)
    ("My room is dirty and I want it cleaned", "complaint"),
    ("This is terrible, I want a refund", "complaint"),
    ("The room is wrong and I'm very disappointed", "complaint"),
    ("My room smells bad", "complaint"),
    ("I have a complaint about my stay", "complaint"),
    ("Wrong room type given to me", "complaint"),
    ("The room is too noisy", "complaint"),
    ("I want to speak to the manager", "complaint"),
    ("Not satisfied with the service", "complaint"),
    ("Missing amenities in my room", "complaint"),
    ("I'm never staying here again", "complaint"),
    ("The check-in took forever", "complaint"),
    ("Extremely disappointed with the quality", "complaint"),
    ("The bed is uncomfortable", "complaint"),
    ("Let me talk to your supervisor", "complaint"),
    ("The room key doesn't work", "complaint"),
    ("My room has the wrong bed type", "complaint"),
    ("This room is awful", "complaint"),
    ("I booked an hour ago where is my confirmation", "booking"),
    ("The bathroom is disgusting", "complaint"),
    ("I want a refund, I don't like this hotel", "complaint"),
    ("The WiFi doesn't work", "complaint"),
    ("This is not what I booked at all", "complaint"),
    ("The air conditioning is broken", "complaint"),
    ("I found bugs in my room", "complaint"),
    ("This is the worst hotel ever", "complaint"),
    ("My reservation was never confirmed", "complaint"),
    ("I want to escalate this issue", "complaint"),
    ("This is a terrible experience", "complaint"),
    ("My room is completely wrong", "complaint"),
    ("The room is freezing cold", "complaint"),
    ("This is horrible", "complaint"),
    ("The service is unacceptable", "complaint"),
    ("I need compensation", "complaint"),
    ("This is ridiculous", "complaint"),
    ("The room is not clean", "complaint"),
    ("I want a refund and to speak to a manager", "complaint"),
    ("My room is dirty and I want my money back", "complaint"),
    ("Actually, my room is filthy", "complaint"),
    ("I'm very disappointed with my room", "complaint"),
    ("This is absolutely unacceptable", "complaint"),
    ("I'm so angry about this booking", "complaint"),
    ("This is disgusting and I want a refund", "complaint"),
    ("Worst hotel I've ever stayed at", "complaint"),
    ("The towels are dirty", "complaint"),
    ("This is ridiculous, I want a manager", "complaint"),
    ("Absolutely terrible experience", "complaint"),
    ("The front desk was very rude", "complaint"),
    ("I want my money back immediately", "complaint"),
    ("You charged me twice", "complaint"),
    ("The bed is hard as a rock", "complaint"),
    ("I'm missing amenities from my room", "complaint"),
    ("This is unacceptable quality", "complaint"),
    ("The shower doesn't work", "complaint"),
    ("My room overlooks a construction site", "complaint"),
    ("The elevator was broken", "complaint"),
    ("I booked a suite but got a standard", "complaint"),
    ("The view is terrible", "complaint"),
    ("This room is ice cold", "complaint"),
    ("I need compensation for this", "complaint"),

    # Hours/Location intents (48 queries)
    ("What time is check-in", "hours_location"),
    ("When is check-out time", "hours_location"),
    ("What's your address", "hours_location"),
    ("Where are you located", "hours_location"),
    ("What are your front desk hours", "hours_location"),
    ("Is the front desk open 24/7", "hours_location"),
    ("Where is the hotel", "hours_location"),
    ("What time can I check in tomorrow", "hours_location"),
    ("Hotel location", "hours_location"),
    ("Are you still accepting check-ins?", "hours_location"),
    ("How do I get directions to your hotel", "hours_location"),
    ("What street are you on?", "hours_location"),
    ("Give me directions", "hours_location"),
    ("What time does check-out end?", "hours_location"),
    ("When can I check in early?", "hours_location"),
    ("Is early check-in available?", "hours_location"),
    ("What neighborhood are you in?", "hours_location"),
    ("Is there a hotel near me?", "hours_location"),
    ("Where is your property?", "hours_location"),
    ("What's the closest hotel?", "hours_location"),
    ("How many locations do you have?", "hours_location"),
    ("Can I check in now?", "hours_location"),
    ("Are you accepting reservations right now?", "hours_location"),
    ("What time is the latest check-in?", "hours_location"),
    ("When does the concierge close?", "hours_location"),
    ("Is the front desk always available?", "hours_location"),
    ("What time should I arrive?", "hours_location"),
    ("Where can I find you?", "hours_location"),
    ("Where are you and when can I check in", "hours_location"),
    ("where you at", "hours_location"),
    ("What time is check-in today", "hours_location"),
    ("What time does the front desk open?", "hours_location"),
    ("What is your street address?", "hours_location"),
    ("Is check-in available on Sundays", "hours_location"),
    ("What time is check-out on weekends", "hours_location"),
    ("Do you have 24 hour service", "amenities_inquiry"),
    ("Where exactly is your hotel", "hours_location"),
    ("How late can I check in tonight", "hours_location"),
    ("What are your hours for the front desk", "hours_location"),
    ("Can I check in right now", "hours_location"),
    ("When does check-in start", "hours_location"),
    ("Are you open on holidays", "hours_location"),
    ("How do I get to your location", "hours_location"),
    ("Is check-in available on Christmas", "hours_location"),
    ("What time should I arrive for check-in", "hours_location"),
    ("Where you guys located at?", "hours_location"),
    ("What time must I check out by", "hours_location"),

    # Room types inquiry intents (57 queries)
    ("What rooms do you have", "room_types"),
    ("What types of rooms are available", "room_types"),
    ("How much does a deluxe room cost", "room_types"),
    ("Do you have suites", "room_types"),
    ("What room sizes do you offer", "room_types"),
    ("Show me room options", "room_types"),
    ("Tell me about your rooms", "room_types"),
    ("Do you have family rooms", "room_types"),
    ("What's the difference between standard and deluxe", "room_types"),
    ("Room prices", "room_types"),
    ("What kind of rooms are available", "room_types"),
    ("Do you have rooms with king beds", "room_types"),
    ("Tell me about the executive suite", "room_types"),
    ("What bed types are available", "room_types"),
    ("How much is a suite", "room_types"),
    ("What rooms have balconies", "room_types"),
    ("Do you have ocean view rooms", "room_types"),
    ("What's your cheapest room", "room_types"),
    ("Do you have luxury suites", "room_types"),
    ("What's included in each room", "room_types"),
    ("Can I see room descriptions", "room_types"),
    ("What amenities come with rooms", "amenities_inquiry"),
    ("Do you have connecting rooms", "room_types"),
    ("What's the largest room", "room_types"),
    ("Tell me about room features", "room_types"),
    ("Do you have harbor view rooms", "room_types"),
    ("What's in the standard room", "room_types"),
    ("Describe the deluxe room", "room_types"),
    ("What's the price range for rooms", "room_types"),
    ("Can I upgrade my room", "cancellation_modification"),
    ("What rooms have kitchens", "room_types"),
    ("Do you have accessible rooms", "amenities_inquiry"),
    ("What's your best room", "room_types"),
    ("How much for a weekend stay", "room_types"),
    ("What rooms are pet-friendly", "room_types"),
    ("Tell me the room rates", "room_types"),
    ("What's the cost per night", "room_types"),
    ("Do you have rooms with two beds", "booking"),
    ("What suites do you offer", "room_types"),
    ("Show me the premium rooms", "room_types"),
    ("What are weekday rates", "room_types"),
    ("Do you have business rooms", "room_types"),
    ("What rooms have the best views", "room_types"),
    ("Tell me about the family room", "room_types"),
    ("What's included in the suite", "room_types"),
    ("Do you have junior suites", "room_types"),
    ("What rooms come with breakfast", "amenities_inquiry"),
    ("How much is a king bed room", "room_types"),
    ("What's the difference in price", "room_types"),
    ("Do you have corner rooms", "room_types"),
    ("What rooms have sofa beds", "room_types"),
    ("Tell me about room upgrades", "room_types"),
    ("What's the rate for two nights", "room_types"),
    ("Do you have penthouse suites", "room_types"),
    ("What rooms are on high floors", "room_types"),
    ("Show me all available room types", "room_types"),
    ("What's your most popular room", "room_types"),

    # Amenities inquiry intents (48 queries)
    ("Do you have a pool", "amenities_inquiry"),
    ("Is there WiFi", "amenities_inquiry"),
    ("Do you have a gym", "amenities_inquiry"),
    ("What amenities do you offer", "amenities_inquiry"),
    ("Do you have parking", "amenities_inquiry"),
    ("Is breakfast included", "amenities_inquiry"),
    ("Do you have a restaurant", "amenities_inquiry"),
    ("What facilities are available", "amenities_inquiry"),
    ("Is there room service", "amenities_inquiry"),
    ("Do you have a spa", "amenities_inquiry"),
    ("What's included in my stay", "amenities_inquiry"),
    ("Are pets allowed", "amenities_inquiry"),
    ("Do you have an airport shuttle", "amenities_inquiry"),
    ("Is there a business center", "amenities_inquiry"),
    ("What services do you provide", "amenities_inquiry"),
    ("Do you have laundry service", "amenities_inquiry"),
    ("Is there a concierge", "amenities_inquiry"),
    ("Do you offer valet parking", "amenities_inquiry"),
    ("What amenities are included", "amenities_inquiry"),
    ("Tell me about your facilities", "amenities_inquiry"),
    ("What can I use at the hotel", "amenities_inquiry"),
    ("Do you have a fitness center", "amenities_inquiry"),
    ("Is parking free", "amenities_inquiry"),
    ("What services are available", "amenities_inquiry"),
    ("Do you have a bar", "amenities_inquiry"),
    ("Is WiFi free", "amenities_inquiry"),
    ("What activities are available", "amenities_inquiry"),
    ("Do you have wheelchair access", "amenities_inquiry"),
    ("Is there an elevator", "amenities_inquiry"),
    ("Do you allow service animals", "amenities_inquiry"),
    ("What's near the hotel", "amenities_inquiry"),
    ("Are there attractions nearby", "amenities_inquiry"),
    ("Is there a beach nearby", "amenities_inquiry"),
    ("What's within walking distance", "amenities_inquiry"),
    ("Do you have EV charging", "amenities_inquiry"),
    ("Is there a safe in the room", "amenities_inquiry"),
    ("Is breakfast complimentary", "amenities_inquiry"),
    ("What dining options do you have", "amenities_inquiry"),
    ("Do you have a rooftop pool", "amenities_inquiry"),
    ("What's your pet policy", "amenities_inquiry"),
    ("Do you have luggage storage", "amenities_inquiry"),
    ("Is there self-parking", "amenities_inquiry"),
    ("What's the parking fee", "amenities_inquiry"),
    ("Do you have a gift shop", "amenities_inquiry"),
    ("Is there coffee in the room", "amenities_inquiry"),
    ("Do you have a lounge", "amenities_inquiry"),

    # Cancellation/Modification intents (15 queries)
    ("I need to cancel my reservation", "cancellation_modification"),
    ("Can I change my booking dates", "cancellation_modification"),
    ("What's your cancellation policy", "cancellation_modification"),
    ("I want to modify my reservation", "cancellation_modification"),
    ("Can I get a refund", "cancellation_modification"),
    ("How do I cancel my booking", "cancellation_modification"),
    ("Can I reschedule my stay", "cancellation_modification"),
    ("I need to change my check-in date", "cancellation_modification"),
    ("What if I need to cancel", "cancellation_modification"),
    ("Is there a cancellation fee", "cancellation_modification"),
    ("Can I postpone my booking", "cancellation_modification"),
    ("I want to extend my stay", "cancellation_modification"),
    ("Can I shorten my reservation", "cancellation_modification"),
    ("How late can I cancel", "cancellation_modification"),
    ("Can I move my reservation to different dates", "cancellation_modification"),

    # General intents (30 queries)
    ("Hello", "general"),
    ("Hi there", "general"),
    ("Good morning", "general"),
    ("Good evening", "general"),
    ("Thank you", "general"),
    ("Thanks so much", "general"),
    ("Goodbye", "general"),
    ("Bye", "general"),
    ("Yes", "general"),
    ("No", "general"),
    ("Okay", "general"),
    ("Sure", "general"),
    ("Sounds good", "general"),
    ("I appreciate your help", "general"),
    ("Greetings", "general"),
    ("I see", "general"),
    ("Thanks for your help", "general"),
    ("That makes sense", "general"),
    ("Fair enough", "general"),
    ("That works for me", "general"),
    ("No problem", "general"),
    ("All good", "general"),
    ("Okay thanks", "general"),
    ("Sure thing", "general"),
    ("Can you help me?", "general"),
    ("What can you do?", "general"),
    ("What information do you need?", "general"),
    ("How can I contact you?", "general"),
    ("What's your phone number?", "general"),
    ("Can I speak to someone?", "general"),
]

# === EDGE CASE TEST DATA ===
EDGE_CASE_TEST_DATASET = [
    # Multi-intent queries
    ("I want to book a deluxe room and know about parking", "booking"),
    ("What's your cancellation policy and can I reserve a suite", "booking"),
    ("Do you have a pool and what are your room prices", "amenities_inquiry"),
    ("Where are you located and what time is check-in", "hours_location"),
    ("I want a refund and to speak to the manager now", "complaint"),

    # Sarcastic/negative complaints
    ("Great, another dirty room", "complaint"),
    ("Perfect, the WiFi doesn't work again", "complaint"),
    ("Wonderful service, my room key stopped working", "complaint"),
    ("Amazing, I love paying for broken air conditioning", "complaint"),
    ("Exactly what I wanted, a room that smells like smoke", "complaint"),

    # Very short queries
    ("book", "booking"),
    ("cancel", "cancellation_modification"),
    ("rooms?", "room_types"),
    ("pool", "amenities_inquiry"),
    ("hours", "hours_location"),
    ("complaint", "complaint"),
    ("hey", "general"),
    ("help", "general"),

    # Ambiguous queries
    ("tell me more", "general"),
    ("what do you have", "room_types"),
    ("how much", "booking"),
    ("can I", "general"),

    # Typos and informal language
    ("i wanna bokk a rom", "booking"),
    ("whats ur adress", "hours_location"),
    ("do u hav wifi", "amenities_inquiry"),
    ("wher r u", "hours_location"),
    ("my room suks", "complaint"),
    ("gime a refnd", "cancellation_modification"),

    # Slang and casual speech
    ("yo I need a room", "booking"),
    ("gimme your best suite", "booking"),
    ("u got parking?", "amenities_inquiry"),
    ("this place is trash", "complaint"),
    ("room's nasty", "complaint"),

    # Long queries
    ("I'm planning a business trip next month and need to book three deluxe rooms with king beds and I'm wondering if you have a business center and meeting rooms available", "booking"),
    ("My family is coming for vacation and we need two family rooms with harbor views and I want to know if breakfast is included and what your pool hours are", "booking"),
    ("I made a reservation last week but I need to change the dates because my conference got rescheduled and I also want to upgrade to a suite if possible", "cancellation_modification"),

    # Context-dependent queries
    ("that one sounds good", "general"),
    ("yes I'll take it", "general"),
    ("how much is that", "room_types"),
    ("do you have availability", "booking"),

    # Mixed casing and punctuation
    ("WHAT ARE YOUR HOURS", "hours_location"),
    ("i WANT to BOOK a ROOM!!!", "booking"),
    ("Do You Have WIFI???", "amenities_inquiry"),
    ("CANCEL MY BOOKING NOW", "cancellation_modification"),

    # Implicit intents
    ("I'm arriving tomorrow at 2pm", "booking"),
    ("I have two small dogs", "amenities_inquiry"),
    ("My flight lands at midnight", "hours_location"),

    # Price-focused queries
    ("what's the cheapest option", "room_types"),
    ("how much for the weekend", "booking"),
    ("do you have deals", "room_types"),
    ("what's your best rate", "room_types"),
    ("is there a discount", "booking"),

    # Comparison queries
    ("what's better, deluxe or suite", "room_types"),
    ("standard vs deluxe room", "room_types"),
    ("which room has the best view", "room_types"),

    # Time-sensitive queries
    ("I need a room tonight", "booking"),
    ("do you have anything available now", "booking"),
    ("last minute booking", "booking"),
    ("emergency reservation", "booking"),

    # Complaint variations
    ("not happy with my stay", "complaint"),
    ("this isn't what I expected", "complaint"),
    ("disappointed", "complaint"),
    ("unsatisfied", "complaint"),

    # Recommendation requests
    ("what do you recommend", "general"),
    ("which room would you suggest", "room_types"),
    ("suggest something for couples", "room_types"),
    ("what's good for families", "room_types"),

    # Follow-up style queries
    ("and what about WiFi", "amenities_inquiry"),
    ("also do you have a pool", "amenities_inquiry"),
    ("one more thing, parking?", "amenities_inquiry"),

    # Emotional expressions
    ("I'm so excited to stay here", "general"),
    ("can't wait for my visit", "general"),
    ("really looking forward to it", "general"),

    # Edge case modifications
    ("I booked yesterday but want to change", "cancellation_modification"),
    ("need to add another night", "cancellation_modification"),
    ("can I switch room types", "cancellation_modification"),
    ("update my reservation please", "cancellation_modification"),
]

# Combine datasets
TEST_DATASET = NORMAL_TEST_DATASET + EDGE_CASE_TEST_DATASET


def get_test_dataset(include_edge_cases=False):
    """Return dataset, optionally with edge cases"""
    if include_edge_cases:
        return NORMAL_TEST_DATASET + EDGE_CASE_TEST_DATASET
    return NORMAL_TEST_DATASET


def check_duplicates():
    """
    Check if dataset contains duplicate queries.
    """
    import re

    dataset = NORMAL_TEST_DATASET + EDGE_CASE_TEST_DATASET
    seen = {}
    duplicates = []

    for query, intent in dataset:
        normalized = query.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = ' '.join(normalized.split())

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

        queries = [q for q, _ in NORMAL_TEST_DATASET + EDGE_CASE_TEST_DATASET]
        vectors = TfidfVectorizer(ngram_range=(1, 2), max_features=500).fit_transform(queries)
        similarities = cosine_similarity(vectors)

        return round(1 - np.mean(similarities[np.triu_indices_from(similarities, k=1)]), 4)
    except ImportError:
        return None


def get_dataset_statistics():
    """Get comprehensive statistics about the dataset"""
    normal_dataset = NORMAL_TEST_DATASET
    edge_cases = EDGE_CASE_TEST_DATASET
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
    stats = get_dataset_statistics()
    print(stats)

    has_dupes, dupes = check_duplicates()
    print("Duplicates found:", has_dupes)

    if has_dupes:
        print(f"\nFound {len(dupes)} duplicate(s):")
        for query, intent, original in dupes:
            print(f"  - '{query}' ({intent})")
            print(f"    Duplicate of: '{original[0]}' ({original[1]})")

    diversity_score = calculate_diversity_score()
    print(f"\nDataset diversity score: {diversity_score}")