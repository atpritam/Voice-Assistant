"""
Domain-Specific Contextual Boost Rules for Algorithmic Intent Recognition
Current Domain: Pizza Restaurant Customer Service
"""

import logging, os, json
from typing import Dict, Set

# BOOST VALUES
ORDER_ACTION_BOOST = 0.20
DELIVERY_STATUS_BOOST = 0.25
ORDER_DELIVERY_PENALTY = 0.15
DELIVERY_COMPLAINT_PENALTY=0.25
NEGATIVE_SENTIMENT_BOOST = 0.4
PRICE_SIZE_BOOST = 0.25
TIME_LOCATION_BOOST = 0.20
ESCALATION_BOOST = 0.35
SARCASM_COMPLAINT_BOOST = 0.35


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
        self.res_info = self._load_res_info()
        self.menu_items = self._extract_menu_items(self.res_info) if self.res_info else set()

    def _load_res_info(self, res_info_file: str = None) -> Dict:
        """Load restaurant information from JSON file"""
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

    def _extract_menu_items(self, res_info: Dict) -> Set[str]:
        """Extract all menu items (toppings, drinks, pizzas) from res_info"""
        items = set()

        if not res_info:
            return items

        menu = res_info.get('menu_highlights', {})
        popular_pizzas = menu.get('popular_pizzas', {})
        items.update(name.lower() for name in popular_pizzas.keys())
        toppings = menu.get('toppings', {})
        for category in toppings.values():
            if isinstance(category, list):
                for item in category:
                    item_lower = item.lower()
                    items.add(item_lower)
                    words = item_lower.split()
                    for word in words:
                        if word not in {'extra', 'vegan'}:
                            items.add(word)
        drinks = menu.get('drinks', [])
        items.update(drink.lower() for drink in drinks)
        crusts = menu.get('crust_types', [])
        for crust in crusts:
            crust_lower = crust.lower()
            items.add(crust_lower.replace(' ', ''))
            items.update(crust_lower.split())

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
        self._apply_order_action_boost(query_words, intent_scores)
        self._apply_negative_sentiment_boost(query_words, intent_scores)
        self._apply_price_size_boost(query_words, intent_scores)
        self._apply_time_location_boost(query_words, intent_scores)
        self._apply_escalation_boost(query_words, intent_scores)
        self._apply_delivery_status_boost(query_words, intent_scores)
        self._apply_sarcasm_boost(query_words, intent_scores)
        self._apply_menu_item_ordering_boost(query_words, intent_scores)
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

            delivery_keywords = {
                'delivery', 'deliver', 'arrived', 'arrive'
            }
            has_delivery_keywords = bool(query_words & delivery_keywords)

            if has_delivery_keywords:
                self._penalty_intent('delivery', intent_scores, DELIVERY_COMPLAINT_PENALTY,
                                     f"Negative+delivery penalty (complaint about delivery)")

    def _apply_price_size_boost(self, query_words: Set[str], intent_scores: Dict):
        """
        RULE 3: Price + size/items indicates menu inquiry
        """
        price_words = {'price', 'prices', 'cost', 'costs', 'much', 'expensive', 'charge', 'pay'}
        has_price = bool(query_words & price_words)

        has_size = bool(query_words & {'small', 'medium', 'large', 'family'})
        has_items = bool(query_words & self.menu_items)

        if has_price and (has_size or has_items):
            boost_label = "Price+size boost" if has_size else "Price+items boost"
            self._boost_intent('menu_inquiry', intent_scores, PRICE_SIZE_BOOST, boost_label)

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


    def _apply_delivery_status_boost(self, query_words: Set[str], intent_scores: Dict):
        """
        RULE 6: Delivery status indicators
        Queries asking about order status should boost delivery
        """
        status_indicators = {'late', 'track', 'status', 'eta', 'arrive', 'long', 'estimated', 'arrival'}
        order_context = {'order', 'pizza', 'food', 'delivery'}

        has_status = bool(query_words & status_indicators)
        has_order = bool(query_words & order_context)

        if has_status and has_order:
            self._boost_intent('delivery', intent_scores, DELIVERY_STATUS_BOOST, "Delivery status boost")
            self._penalty_intent('order', intent_scores, ORDER_DELIVERY_PENALTY, "Order penalty for status query")
        if has_status:
            self._boost_intent('delivery', intent_scores, DELIVERY_STATUS_BOOST, "Delivery status boost")

    def _apply_sarcasm_boost(self, query_words: Set[str], intent_scores: Dict):
        """
        RULE 7: Sarcasm/negative complaint indicators
        Detect sarcastic complaints
        """
        sarcasm_markers = {'amazing', 'wonderful', 'perfect', 'great', 'love', 'exactly'}
        negative_context = {'wrong', 'late', 'cold', 'burnt', 'not'}

        has_sarcasm = bool(query_words & sarcasm_markers)
        has_negative = bool(query_words & negative_context)

        if has_sarcasm:
            if has_negative or 'as always' in ' '.join(query_words):
                self._boost_intent('complaint', intent_scores, SARCASM_COMPLAINT_BOOST, "Sarcasm complaint boost")

    def _apply_menu_item_ordering_boost(self, query_words: Set[str], intent_scores: Dict):
        """
        RULE 8: Menu item mention suggests ordering
        Users mentioning specific pizzas/toppings often want to order
        """
        has_menu_item = bool(query_words & self.menu_items)
        question_words = {'what', 'which', 'how', 'do', 'does', 'can', 'is', 'are', 'tell', 'show'}
        has_question = bool(query_words & question_words)

        if has_menu_item and not has_question:
            self._boost_intent('order', intent_scores, ORDER_ACTION_BOOST, "Menu item ordering boost")