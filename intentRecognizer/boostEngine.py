"""
Domain-Specific Contextual Boost Rules for Algorithmic Intent Recognition
Current Domain: Hotel Booking and Guest Services
"""

import logging, os, json
from typing import Dict, Set

# ============================================================================
# BOOST VALUES - Positive adjustments to intent scores

BOOKING_ACTION_BOOST = 0.25
NEGATIVE_SENTIMENT_BOOST = 0.4
PRICE_ROOM_BOOST = 0.25
TIME_LOCATION_BOOST = 0.25
ESCALATION_BOOST = 0.35
AMENITIES_INQUIRY_BOOST = 0.25
SARCASM_COMPLAINT_BOOST = 0.35
ROOM_MENTION_BOOKING_BOOST = 0.20
RECOMMENDATION_INQUIRY_BOOST = 0.25
ROOM_DESCRIPTION_BOOST = 0.25
QUANTITY_BOOKING_BOOST = 0.35

# ============================================================================
# PENALTY VALUES - Negative adjustments to intent scores

BOOKING_INQUIRY_PENALTY = 0.15
COMPLAINT_BOOKING_PENALTY = 0.30
AMENITIES_ROOM_TYPES_PENALTY = 0.20
ROOM_TYPES_BOOKING_PENALTY = 0.15
CANCELLATION_COMPLAINT_PENALTY = 0.25



class BoostRuleEngine:
    """
    Applies domain-specific contextual boost rules to intent scores
    """

    def __init__(self, intent_critical_keywords: Dict, synonyms: Dict, enable_logging: bool = False):
        """
        Initialize boost rule engine

        Args:
            intent_critical_keywords: Domain-specific critical keywords per intent
            synonyms: Synonym lookup dictionary
            enable_logging: Enable detailed logging of boost applications
        """
        self.intent_critical_keywords = intent_critical_keywords
        self.synonyms = synonyms
        self.enable_logging = enable_logging
        if enable_logging:
            self.logger = logging.getLogger(__name__)
        self.res_info = self._load_res_info()
        self.room_entities = self._extract_room_entities(self.res_info) if self.res_info else set()

        self.negative_words = {
            'wrong', 'bad', 'terrible', 'horrible', 'disappointed', 'complain',
            'unhappy', 'angry', 'upset', 'disgusted', 'awful', 'missing', 'dirty',
            'noisy', 'issue', 'problem', 'never', 'poor', 'nasty', 'disgusting',
            'unacceptable', 'uncomfortable', 'broken', 'smells', 'furious',
            'mistake', 'error', 'incorrect', 'filthy', 'rude', 'overlooks',
            'unprofessional', 'unclean', 'messy', 'worst', 'hate'
        }

    def _load_res_info(self, res_info_file: str = None) -> Dict:
        """Load hotel information from JSON file"""
        if res_info_file is None:
            utils_dir = os.path.join(os.path.dirname(__file__), '..', 'utils')
            res_info_file = os.path.join(utils_dir, 'res_info.json')

        try:
            with open(res_info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            if self.enable_logging:
                self.logger.warning(f"Could not load res_info: {e}")
            return {}

    def _extract_room_entities(self, res_info: Dict) -> Set[str]:
        """Extract all room types and bed types from res_info"""
        items = set()

        if not res_info:
            return items

        # Extract room types
        room_types = res_info.get('room_types', {})
        for room_name in room_types.keys():
            room_lower = room_name.lower()
            items.add(room_lower)
            items.update(room_lower.split())

        # Extract bed types
        for room_data in room_types.values():
            if 'bed_type' in room_data:
                bed_type = room_data['bed_type'].lower()
                items.update(bed_type.split())

        return items

    def apply_all_boosts(self, query_words: Set[str], intent_scores: Dict, query_text: str = "") -> Dict:
        """
        Apply all boost rules to intent scores

        Args:
            query_words: Set of words from user query
            intent_scores: Dict of intent names to score data
            query_text: Original query text (optional, for pattern matching)

        Returns:
            Updated intent_scores with boosts applied
        """
        self._apply_booking_action_boost(query_words, intent_scores)
        self._apply_negative_sentiment_boost(query_words, intent_scores)
        self._apply_price_room_boost(query_words, intent_scores)
        self._apply_time_location_boost(query_words, intent_scores)
        self._apply_escalation_boost(query_words, intent_scores)
        self._apply_amenities_inquiry_boost(query_words, intent_scores)
        self._apply_sarcasm_boost(query_words, intent_scores)
        self._apply_room_mention_booking_boost(query_words, intent_scores)
        self._apply_inquiry_pattern_boost(query_words, intent_scores)
        self._apply_room_description_boost(query_words, intent_scores)
        self._apply_quantity_booking_boost(query_words, intent_scores)
        return intent_scores

    def _boost_intent(self, intent_name: str, intent_scores: Dict, boost: float, label: str = ""):
        """Helper to apply boost to intent"""
        if intent_name in intent_scores:
            original = intent_scores[intent_name]['similarity']
            intent_scores[intent_name]['similarity'] = min(1.0, original + boost)
            if self.enable_logging and label:
                self.logger.debug(f"{label} [{intent_name}]: {original:.3f} -> {intent_scores[intent_name]['similarity']:.3f}")

    def _penalty_intent(self, intent_name: str, intent_scores: Dict, penalty: float, label: str = ""):
        """Helper to apply penalty to intent"""
        if intent_name in intent_scores:
            original = intent_scores[intent_name]['similarity']
            intent_scores[intent_name]['similarity'] = max(0.0, original - penalty)
            if self.enable_logging and label:
                self.logger.debug(f"{label} [{intent_name}]: {original:.3f} -> {intent_scores[intent_name]['similarity']:.3f}")


    # ========================================================================
    # HOTEL BOOKING DOMAIN-SPECIFIC BOOST RULES (11 FOCUSED RULES)

    def _apply_booking_action_boost(self, query_words: Set[str], intent_scores: Dict):
        """
        RULE 1: Booking action verb handling
        Boosts booking intent when booking action verbs and 'book/reserve' keywords present
        """
        has_negative = bool(query_words & self.negative_words)
        has_booking_action = bool(query_words & self.intent_critical_keywords.get('booking', set()))
        booking_keywords = {'book', 'reserve', 'reservation', 'booking'}
        has_booking_keyword = bool(query_words & booking_keywords)

        if has_booking_action and has_booking_keyword and not has_negative:
            self._boost_intent('booking', intent_scores, BOOKING_ACTION_BOOST, "Booking action boost")

    def _apply_negative_sentiment_boost(self, query_words: Set[str], intent_scores: Dict):
        """
        RULE 2: Negative sentiment for complaints
        Universal rule - boosts complaints based on negative words
        """
        negative_count = len(query_words & self.negative_words)

        if negative_count > 0:
            boost_amount = min(NEGATIVE_SENTIMENT_BOOST * min(negative_count, 3) / 2, 0.35)
            self._boost_intent('complaint', intent_scores, boost_amount,
                               f"Negative sentiment boost (x{negative_count})")

            # Past tense complaint pattern: "I booked but got wrong room"
            past_complaint_words = {'got', 'received', 'was', 'had', 'booked'}
            has_past = bool(query_words & past_complaint_words)
            if has_past:
                self._penalty_intent('booking', intent_scores, COMPLAINT_BOOKING_PENALTY,
                                     "Booking penalty (past complaint)")

            # Cancellation with negative = complaint about policy
            cancellation_keywords = {'cancel', 'cancellation', 'refund'}
            has_cancellation = bool(query_words & cancellation_keywords)
            if has_cancellation:
                self._penalty_intent('cancellation_modification', intent_scores, CANCELLATION_COMPLAINT_PENALTY,
                                     "Cancellation penalty (complaint about policy)")

    def _apply_price_room_boost(self, query_words: Set[str], intent_scores: Dict):
        """
        RULE 3: Price + room types indicates room_types inquiry
        """
        price_words = {'price', 'prices', 'cost', 'costs', 'much', 'expensive', 'charge', 'pay', 'rate', 'rates', 'fee'}
        has_price = bool(query_words & price_words)
        has_room_entity = bool(query_words & self.room_entities)
        has_negative = bool(query_words & self.negative_words)

        if has_price and has_room_entity and not has_negative:
            self._boost_intent('room_types', intent_scores, PRICE_ROOM_BOOST, "Price+room boost")
        elif has_price and has_negative:
            self._penalty_intent('room_types', intent_scores, BOOKING_INQUIRY_PENALTY,
                               "Room types penalty (complaint context)")

    def _apply_time_location_boost(self, query_words: Set[str], intent_scores: Dict):
        """
        RULE 4: Time/location questions boost hours_location
        Universal rule across domains
        """
        time_location_context = {
            'when', 'what time', 'how long', 'until when', 'from when',
            'where', 'which', 'what address', 'how far', 'get to'
        }
        hours_keywords = {
            'open', 'close', 'hours', 'location', 'address', 'hotel',
            'at', 'find', 'there', 'check-in', 'check-out', 'checkin', 'checkout',
            'desk', 'service', 'front'
        }

        has_question = bool(query_words & time_location_context)
        has_hours = bool(query_words & hours_keywords)

        if has_question and has_hours:
            self._boost_intent('hours_location', intent_scores, TIME_LOCATION_BOOST,
                               "Time/location boost")
            self._penalty_intent('booking', intent_scores, BOOKING_INQUIRY_PENALTY,
                               "Booking penalty for time query")

    def _apply_escalation_boost(self, query_words: Set[str], intent_scores: Dict):
        """
        RULE 5: Escalation keywords strongly indicate complaint
        Universal across customer service domains
        """
        escalation_keywords = {
            'refund', 'manager', 'supervisor', 'speak to', 'talk to',
            'compensation', 'money back', 'unacceptable', 'ridiculous'
        }
        query_norm = ' '.join(query_words)

        if any(keyword in query_norm for keyword in escalation_keywords):
            self._boost_intent('complaint', intent_scores, ESCALATION_BOOST, "Escalation boost")

    def _apply_amenities_inquiry_boost(self, query_words: Set[str], intent_scores: Dict):
        """
        RULE 6: Amenity/service-specific queries
        Differentiates amenities from room types
        """
        amenity_keywords = {
            'pool', 'gym', 'spa', 'wifi', 'parking', 'breakfast', 'restaurant', 'bar',
            'policy', 'policies', 'fee', 'shuttle', 'luggage', 'wheelchair', 'accessible',
            'pet', 'pets', 'dog', 'dogs', 'animal', 'valet', 'storage', 'dining'
        }
        question_words = {'what', 'do', 'does', 'is', 'are', 'have', 'offer'}

        has_amenity = bool(query_words & amenity_keywords)
        has_question = bool(query_words & question_words)

        if has_amenity and has_question:
            self._boost_intent('amenities_inquiry', intent_scores, AMENITIES_INQUIRY_BOOST,
                             "Amenities inquiry boost")
            self._penalty_intent('room_types', intent_scores, AMENITIES_ROOM_TYPES_PENALTY,
                               "Room types penalty (amenity query)")

    def _apply_sarcasm_boost(self, query_words: Set[str], intent_scores: Dict):
        """
        RULE 7: Sarcasm/negative complaint indicators
        Universal pattern for detecting sarcastic complaints
        """
        sarcasm_markers = {'amazing', 'wonderful', 'perfect', 'great', 'love', 'exactly', 'brilliant'}
        negative_context = {'wrong', 'not', 'forever', 'still', 'yet', 'didnt', 'dirty', 'noisy', 'broken', 'smells'}
        expectation_words = {'waited', 'waiting', 'only', 'still', 'yet', 'finally'}

        has_sarcasm = bool(query_words & sarcasm_markers)
        has_negative = bool(query_words & negative_context)
        has_expectation = bool(query_words & expectation_words)

        if has_sarcasm and (has_negative or has_expectation):
            self._boost_intent('complaint', intent_scores, SARCASM_COMPLAINT_BOOST,
                               "Sarcasm complaint boost")

    def _apply_room_mention_booking_boost(self, query_words: Set[str], intent_scores: Dict):
        """
        RULE 8: Room mention + action verb suggests booking
        Similar to pizza domain's menu item ordering boost
        """
        has_room_entity = bool(query_words & self.room_entities)
        question_words = {'what', 'which', 'how', 'do', 'does', 'tell', 'show', 'describe'}
        has_question = bool(query_words & question_words)
        has_booking_action = bool(query_words & self.synonyms.get('booking', set()))
        has_size = bool(query_words & {'one', 'two', 'three', 'four', '1', '2', '3', '4'})

        # Room mention without question words = booking intent
        if has_room_entity and not has_question:
            self._boost_intent('booking', intent_scores, ROOM_MENTION_BOOKING_BOOST,
                             "Room mention booking boost")
            self._penalty_intent('room_types', intent_scores, ROOM_TYPES_BOOKING_PENALTY,
                               "Room types penalty (booking context)")

        # Short query with room/size + booking verb = booking
        elif (has_room_entity or has_size) and has_booking_action and len(query_words) <= 6:
            self._boost_intent('booking', intent_scores, ROOM_MENTION_BOOKING_BOOST,
                             "Room mention booking boost")

    def _apply_inquiry_pattern_boost(self, query_words: Set[str], intent_scores: Dict):
        """
        RULE 9: Question about X inquiry pattern
        Handles recommendation and information queries
        """
        inquiry_words = {'question', 'asking', 'inquire', 'wondering', 'ask', 'tell', 'know', 'recommend', 'suggest'}
        about_words = {'about', 'regarding', 'can', 'concerning'}
        room_context = {'options', 'choices', 'good', 'best', 'room', 'suite', 'couples', 'families'}

        has_inquiry = bool(query_words & inquiry_words)
        has_about = bool(query_words & about_words)
        has_room = bool(query_words & room_context)

        # Recommendation queries = room_types
        if has_inquiry and has_room:
            self._boost_intent('room_types', intent_scores, RECOMMENDATION_INQUIRY_BOOST,
                             "Recommendation inquiry boost")
            self._penalty_intent('booking', intent_scores, BOOKING_INQUIRY_PENALTY,
                             "Booking penalty for recommendation")

    def _apply_room_description_boost(self, query_words: Set[str], intent_scores: Dict):
        """
        RULE 10: Room description/feature queries = room_types
        "What's included", "describe the room", "tell me about features"
        """
        description_words = {'describe', 'included', 'comes', 'features', 'amenities', 'has', 'offer', 'difference', 'vs', 'versus', 'compare', 'better'}
        room_words = {'room', 'suite', 'deluxe', 'standard', 'executive', 'family'} | self.room_entities

        has_description = bool(query_words & description_words)
        has_room = bool(query_words & room_words)

        if has_description and has_room:
            self._boost_intent('room_types', intent_scores, ROOM_DESCRIPTION_BOOST,
                             "Room description boost")
            self._penalty_intent('booking', intent_scores, BOOKING_INQUIRY_PENALTY,
                             "Booking penalty (description query)")
            self._penalty_intent('amenities_inquiry', intent_scores, AMENITIES_ROOM_TYPES_PENALTY,
                               "Amenities penalty (room description)")

    def _apply_quantity_booking_boost(self, query_words: Set[str], intent_scores: Dict):
        """
        RULE 11: Quantity + rooms/group = booking intent
        "I need three rooms for a conference"
        """
        quantity_words = {'one', 'two', 'three', 'four', 'five', 'six', '1', '2', '3', '4', '5', 'couple', 'few', 'several'}
        room_plural = {'rooms', 'suites'}
        group_context = {'conference', 'group', 'party', 'team', 'family', 'guests'}
        need_words = {'need', 'want', 'require', 'book', 'reserve'}

        has_quantity = bool(query_words & quantity_words)
        has_rooms = bool(query_words & room_plural)
        has_group = bool(query_words & group_context)
        has_need = bool(query_words & need_words)
        has_negative = bool(query_words & self.negative_words)

        if ((has_quantity and has_rooms) or (has_quantity and has_group and has_need)) and not has_negative:
            self._boost_intent('booking', intent_scores, QUANTITY_BOOKING_BOOST,
                             "Quantity booking boost")
            self._penalty_intent('complaint', intent_scores, COMPLAINT_BOOKING_PENALTY,
                               "Complaint penalty (quantity booking)")
