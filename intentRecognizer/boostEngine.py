"""
Domain-Specific Contextual Boost Rules for Algorithmic Intent Recognition
Current Domain: Pizza Restaurant Customer Service
"""

import logging
from typing import Dict, Set

# BOOST VALUES
ORDER_ACTION_BOOST = 0.20
ORDER_DELIVERY_PENALTY = 0.15
NEGATIVE_SENTIMENT_BOOST = 0.20
PRICE_SIZE_BOOST = 0.25
TIME_LOCATION_BOOST = 0.20
ESCALATION_BOOST = 0.30
REPEAT_ORDER_BOOST = 0.30


class BoostRuleEngine:
    """
    Applies domain-specific contextual boost rules to intent scores
    """

    def __init__(self, intent_critical_keywords: Dict, enable_logging: bool = False):
        """
        Initialize boost rule engine

        Args:
            intent_critical_keywords: Domain-specific critical keywords per intent
            enable_logging: Enable detailed logging of boost applications
        """
        self.intent_critical_keywords = intent_critical_keywords
        self.enable_logging = enable_logging
        if enable_logging:
            self.logger = logging.getLogger(__name__)

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
        self._apply_order_action_boost(query_words, intent_scores)
        self._apply_negative_sentiment_boost(query_words, intent_scores)
        self._apply_price_size_boost(query_words, intent_scores)
        self._apply_time_location_boost(query_words, intent_scores)
        self._apply_escalation_boost(query_words, intent_scores)
        return intent_scores

    def _boost_intent(self, intent_name: str, intent_scores: Dict, boost: float, label: str = ""):
        """Helper to apply boost to intent"""
        if intent_name in intent_scores:
            original = intent_scores[intent_name]['similarity']
            intent_scores[intent_name]['similarity'] = min(1.0, original + boost)
            if self.enable_logging and label:
                self.logger.debug(f"{label}: {original:.3f} -> {intent_scores[intent_name]['similarity']:.3f}")

    def _penalty_intent(self, intent_name: str, intent_scores: Dict, penalty: float, label: str = ""):
        """Helper to apply penalty to intent"""
        if intent_name in intent_scores:
            original = intent_scores[intent_name]['similarity']
            intent_scores[intent_name]['similarity'] = max(0.0, original - penalty)
            if self.enable_logging and label:
                self.logger.debug(f"{label}: {original:.3f} -> {intent_scores[intent_name]['similarity']:.3f}")


    # DOMAIN-SPECIFIC BOOST RULES

    def _apply_order_action_boost(self, query_words: Set[str], intent_scores: Dict):
        """
        RULE 1: Order action verb handling
        """
        has_order_action = bool(query_words & self.intent_critical_keywords.get('order', set()))
        has_order_keyword = 'order' in query_words

        if has_order_action and has_order_keyword:
            self._boost_intent('order', intent_scores, ORDER_ACTION_BOOST, "Order boost")

            has_tracking = bool(query_words & {'track', 'status', 'where', 'eta'})
            if not has_tracking:
                self._penalty_intent('delivery', intent_scores, ORDER_DELIVERY_PENALTY, "Delivery penalty")

    def _apply_negative_sentiment_boost(self, query_words: Set[str], intent_scores: Dict):
        """
        RULE 2: Negative sentiment for complaints
        This rule is generally applicable to most customer service domains.
        """
        negative_words = {
            'wrong', 'bad', 'terrible', 'horrible', 'disappointed', 'complain',
            'unhappy', 'angry', 'upset', 'disgusted', 'awful', 'missing', 'cold',
            'late', 'issue', 'problem', 'never', 'poor', 'nasty', 'disgusting',
            'unacceptable', 'burnt', 'undercooked', 'overcooked', 'stale'
        }
        negative_count = len(query_words & negative_words)

        if negative_count > 0:
            boost_amount = min(NEGATIVE_SENTIMENT_BOOST * min(negative_count, 3) / 2, 0.35)
            self._boost_intent('complaint', intent_scores, boost_amount,
                               f"Negative sentiment boost (x{negative_count})")

    def _apply_price_size_boost(self, query_words: Set[str], intent_scores: Dict):
        """
        RULE 3: Price + size indicates menu inquiry
        """
        has_price = bool(query_words & {'price', 'prices', 'cost', 'much'})
        has_size = bool(query_words & {'small', 'medium', 'large', 'family'})

        if has_price and has_size:
            self._boost_intent('menu_inquiry', intent_scores, PRICE_SIZE_BOOST, "Price+size boost")

    def _apply_time_location_boost(self, query_words: Set[str], intent_scores: Dict):
        """
        RULE 4: Time/location questions boost hours_location
        This rule is generally applicable across domains.
        """
        time_location_context = {
            'when', 'what time', 'how long', 'until when', 'from when',
            'where', 'which', 'what address', 'how far'
        }
        hours_keywords = {'open', 'close', 'hours', 'location', 'address', 'store'}

        has_question = bool(query_words & time_location_context)
        has_hours = bool(query_words & hours_keywords)

        if has_question and has_hours:
            self._boost_intent('hours_location', intent_scores, TIME_LOCATION_BOOST,
                               "Time/location boost")

    def _apply_escalation_boost(self, query_words: Set[str], intent_scores: Dict):
        """
        RULE 5: Escalation keywords strongly indicate complaint
        This rule is highly universal across customer service domains.
        """
        escalation_keywords = {
            'refund', 'manager', 'supervisor', 'speak to', 'talk to',
            'compensation', 'money back', 'unacceptable', 'ridiculous'
        }
        query_norm = ' '.join(query_words)

        if any(keyword in query_norm for keyword in escalation_keywords):
            self._boost_intent('complaint', intent_scores, ESCALATION_BOOST, "Escalation boost")


    def _apply_repeat_order_boost(self, query_words: Set[str], intent_scores: Dict):
        """
        RULE 6: Repeat order patterns strongly indicate ordering intent
        Returning customers often use shorthand references to previous orders
        """
        repeat_patterns = {
            'same', 'usual', 'regular', 'again', 'last', 'previous',
            'before', 'always', 'normally', 'typically', 'typical'
        }

        order_context = {
            'order', 'time', 'get', 'want', 'have', 'like'
        }

        has_repeat = bool(query_words & repeat_patterns)
        has_context = bool(query_words & order_context)
        word_count = len(query_words)

        if has_repeat:
            if word_count <= 3:
                self._boost_intent('order', intent_scores, REPEAT_ORDER_BOOST,
                                   "Repeat order boost (short query)")
            elif has_context:
                self._boost_intent('order', intent_scores, REPEAT_ORDER_BOOST,
                                   "Repeat order boost (with context)")