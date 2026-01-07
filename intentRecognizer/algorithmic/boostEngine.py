"""
Domain-Specific Contextual Boost Rules for Algorithmic Intent Recognition
Current Domain: Pizza Restaurant Customer Service
"""

import os
import sys
import json
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Set

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.logger import ConditionalLogger

# ============================================================================
# BOOST VALUES - Positive adjustments to intent scores

ORDER_ACTION_BOOST = 0.15
DELIVERY_STATUS_BOOST = 0.20
DELIVERY_INQUIRY_BOOST = 0.20
MENU_ITEM_ORDERING_BOOST = 0.20
PRICE_INQUIRY_BOOST = 0.20
PRICE_SIZE_BOOST = 0.25
TIME_LOCATION_BOOST = 0.25
INQUIRY_QUESTION_BOOST = 0.25
MENU_ITEM_CUSTOMIZATION_BOOST = 0.25
RECOMMENDATION_INQUIRY_BOOST = 0.25
HOURS_INQUIRY_BOOST = 0.25
MENU_ITEM_QUALITY_BOOST = 0.30
SARCASM_COMPLAINT_BOOST = 0.30
ESCALATION_BOOST = 0.30
NEGATIVE_SENTIMENT_BOOST = 0.35

# ============================================================================
# PENALTY VALUES - Negative adjustments to intent scores

ORDER_DELIVERY_PENALTY = 0.15
ORDER_ESCALATION_PENALTY = 0.15
MENU_INQUIRY_PENALTY = 0.15
ORDER_INQUIRY_PENALTY = 0.15
HOURS_MENU_PENALTY = 0.15
GENERAL_INQUIRY_PENALTY = 0.15
DELIVERY_RECOMMENDATION_PENALTY = 0.15
DELIVERY_COMPLAINT_PENALTY = 0.25


@dataclass(frozen=True)
class IntentAdjustment:
    """Represents a single boost or penalty applied to an intent score."""
    intent: str
    amount: float
    label: str = ""
    is_penalty: bool = False


@dataclass(frozen=True)
class BoostContext:
    """Per-query data available to boost rules."""
    query_words: Set[str]
    query_text: str


@dataclass(frozen=True)
class BoostRule:
    """Declarative boost rule."""
    name: str
    evaluator: Callable[[BoostContext], Iterable[IntentAdjustment]]


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
        self.logger = ConditionalLogger(__name__, enable_logging)
        self.res_info = self._load_res_info()
        self.menu_items = self._extract_menu_items(self.res_info) if self.res_info else set()

        self.negative_words = {
            'wrong', 'bad', 'terrible', 'horrible', 'disappointed', 'complain',
            'unhappy', 'angry', 'upset', 'disgusted', 'awful', 'missing', 'cold',
            'late', 'issue', 'problem', 'never', 'poor', 'nasty', 'disgusting',
            'unacceptable', 'burnt', 'undercooked', 'overcooked', 'stale', 'furious',
            'mistake', 'wrong', 'error', 'incorrect', 'cancel', 'cancellation', 'botched',
            'replace', 'replaced', 'replacement', 'redo', 'remake', 'fix', 'broken'
        }

        self.rules: List[BoostRule] = self._build_rules()

    def _build_rules(self) -> List[BoostRule]:
        """Initialize list of boost rules."""
        return [
            BoostRule("order_action", self._rule_order_action),
            BoostRule("negative_sentiment", self._rule_negative_sentiment),
            BoostRule("price_size", self._rule_price_size),
            BoostRule("time_location", self._rule_time_location),
            BoostRule("escalation", self._rule_escalation),
            BoostRule("delivery_status", self._rule_delivery_status),
            BoostRule("sarcasm", self._rule_sarcasm),
            BoostRule("menu_item_ordering", self._rule_menu_item_ordering),
            BoostRule("inquiry_pattern", self._rule_inquiry_pattern),
            BoostRule("menu_item_quality", self._rule_menu_item_quality),
        ]

    def _load_res_info(self, res_info_file: str = None) -> Dict:
        """Load restaurant information from JSON file"""
        if res_info_file is None:
            utils_dir = os.path.join(os.path.dirname(__file__), '../..', 'utils')
            res_info_file = os.path.join(utils_dir, 'res_info.json')

        try:
            with open(res_info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Could not load res_info: {e}")
            return {}

    def _extract_menu_items(self, res_info: Dict) -> Set[str]:
        """Extract all menu items (toppings, drinks, pizzas) from res_info"""
        items = set()

        if not res_info:
            return items

        menu = res_info.get('menu_highlights', {})
        excluded_words = {'extra'}

        split = ([name.lower() for name in menu.get('popular_pizzas', {}).keys()]
                 + [item.lower() for category in menu.get('toppings', {}).values()
                                 if isinstance(category, list) for item in category])
        for s in split:
            items.add(s)
            items.update(word for word in s.split() if word not in excluded_words)

        items.update(drink.lower() for drink in menu.get('drinks', []))

        for crust in menu.get('crust_types', []):
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
        context = BoostContext(query_words=query_words, query_text=query_text or "")

        for rule in self.rules:
            for adjustment in rule.evaluator(context):
                self._apply_adjustment(intent_scores, adjustment)

        return intent_scores

    def _apply_adjustment(self, intent_scores: Dict, adjustment: IntentAdjustment) -> None:
        """Apply a single adjustment to the intent scores."""
        if adjustment.amount <= 0 or  adjustment.intent not in intent_scores:
            return

        original = intent_scores[adjustment.intent]['similarity']

        if adjustment.is_penalty:
            intent_scores[adjustment.intent]['similarity'] = max(0.0, original - adjustment.amount)
        else:
            intent_scores[adjustment.intent]['similarity'] = min(1.0, original + adjustment.amount)
        if adjustment.label:
            self.logger.debug(
                f"{adjustment.label} [{adjustment.intent}]: {original:.3f} -> {intent_scores[adjustment.intent]['similarity']:.3f}")

    # ========================================================================
    # DOMAIN-SPECIFIC BOOST RULES

    def _rule_order_action(self, context: BoostContext) -> Iterable[IntentAdjustment]:
        """
        RULE 1: Order action verb handling
        Boosts order intent when both order action verbs and 'order' keyword present
        """
        query_words = context.query_words
        has_negative = bool(query_words & self.negative_words)
        order_action = set(self.intent_critical_keywords.get('order', set())) - {'order'}
        has_order_action = bool(query_words & order_action)
        has_order_keyword = 'order' in query_words

        if has_order_action and has_order_keyword and not has_negative:
            yield IntentAdjustment('order', ORDER_ACTION_BOOST, "Order action boost")

    def _rule_negative_sentiment(self, context: BoostContext) -> Iterable[IntentAdjustment]:
        """
        RULE 2: Negative sentiment for complaints
        This rule is generally applicable to most customer service domains.
        """
        query_words = context.query_words
        negative_count = len(query_words & self.negative_words)

        if negative_count > 0:
            boost_amount = min(NEGATIVE_SENTIMENT_BOOST * min(negative_count, 3) / 2, 0.35)
            yield IntentAdjustment('complaint',boost_amount,f"Negative sentiment boost (x{negative_count})")

            delivery_keywords = {
                'delivery', 'deliver', 'arrived', 'arrive'
            }
            has_delivery_keywords = bool(query_words & delivery_keywords)

            if has_delivery_keywords:
                yield IntentAdjustment('delivery',DELIVERY_COMPLAINT_PENALTY,
                    "Negative+delivery penalty (complaint about delivery)",
                    is_penalty=True)

    def _rule_price_size(self, context: BoostContext) -> Iterable[IntentAdjustment]:
        """
        RULE 3: Price + size/items indicates menu inquiry
        """
        query_words = context.query_words
        price_words = {'price', 'prices', 'cost', 'costs', 'much', 'expensive', 'charge', 'pay'}
        has_price = bool(query_words & price_words)

        has_size = bool(query_words & {'small', 'medium', 'large', 'family'})
        has_items = bool(query_words & ({'drinks', 'toppings'} | set(self.menu_items or [])))
        has_negative = bool(query_words & self.negative_words)

        if has_price and (has_size or has_items):
            if not has_negative:
                boost_label = "Price+size boost" if has_size else "Price+items boost"
                yield IntentAdjustment('menu_inquiry', PRICE_SIZE_BOOST, boost_label)
            else:
                yield IntentAdjustment('menu_inquiry',MENU_INQUIRY_PENALTY,
                    "Menu inquiry penalty (complaint context)",
                    is_penalty=True)
        elif has_price and has_negative:
            yield IntentAdjustment('menu_inquiry',MENU_INQUIRY_PENALTY,
                "Menu inquiry penalty (negative+price)",
                is_penalty=True)

    def _rule_time_location(self, context: BoostContext) -> Iterable[IntentAdjustment]:
        """
        RULE 4: Time/location questions boost hours_location
        This rule is generally applicable across domains.
        """
        query = context.query_text
        query_words = context.query_words
        time_location_context = {
            'when', 'what time', 'until when', 'from when',
            'where', 'which', 'what', 'how far', 'late night', 'get to'
        }
        hl_keywords = {'open', 'close', 'hours', 'location', 'address', 'street', 'store', 'at', 'find', 'there'}
        other_keywords = {'food', 'pizza', 'delivery', 'order', 'driver'}

        has_question = any(phrase in query for phrase in time_location_context)
        has_hours_location = bool(query_words & hl_keywords)
        has_other = bool(query_words & other_keywords)

        if has_question and has_hours_location and not has_other:
            yield IntentAdjustment('hours_location', TIME_LOCATION_BOOST, "Time/location boost")
            yield IntentAdjustment('order',ORDER_DELIVERY_PENALTY,
                "Order penalty for time query",
                is_penalty=True)

    def _rule_escalation(self, context: BoostContext) -> Iterable[IntentAdjustment]:
        """
        RULE 5: Escalation keywords strongly indicate complaint
        This rule is highly universal across customer service domains.
        """
        direct_escalation = {'refund', 'compensation', 'money back', 'chargeback'}
        escalation_phrases = {'speak to', 'talk to', 'get me', 'connect me', 'transfer me'}
        authority_figures = {'manager', 'supervisor', 'boss', 'higher up'}

        query = context.query_text
        has_direct_escalation = any(phrase in query for phrase in direct_escalation)

        # escalation - phrase and authority figure combination
        has_escalation_phrase = any(phrase in query for phrase in escalation_phrases)
        has_authority_figure = any(authority in query for authority in authority_figures)
        is_hard_escalation = has_escalation_phrase and has_authority_figure

        if has_direct_escalation or is_hard_escalation:
            yield IntentAdjustment('complaint', ESCALATION_BOOST, "Escalation boost")
            yield IntentAdjustment('order',ORDER_ESCALATION_PENALTY,
                "Order penalty for escalation query",
                is_penalty=True)

    def _rule_delivery_status(self, context: BoostContext) -> Iterable[IntentAdjustment]:
        """
        RULE 6: Delivery status indicators
        Queries asking about order status should boost delivery
        """
        query_words = context.query_words
        status_indicators = {'track', 'status', 'eta', 'arrive', 'long', 'estimated',
                             'arrival', 'longer', 'expect', 'waiting'}
        order_context = {'order', 'pizza', 'food', 'delivery'}

        has_status = bool(query_words & status_indicators)
        has_order = bool(query_words & order_context)
        has_where = bool(query_words & {'where'})

        if has_status or (has_order and has_where):
            yield IntentAdjustment('delivery', DELIVERY_STATUS_BOOST, "Delivery status boost")

            if has_order:
                yield IntentAdjustment('order',ORDER_DELIVERY_PENALTY,
                    "Order penalty for status query",
                    is_penalty=True)

    def _rule_sarcasm(self, context: BoostContext) -> Iterable[IntentAdjustment]:
        """
        RULE 7: Enhanced sarcasm/negative complaint indicators
        """
        query_words = context.query_words
        sarcasm_markers = {'amazing', 'wonderful', 'perfect', 'great', 'love', 'exactly', 'brilliant', 'incredibly'}
        negative_context = {'wrong', 'late', 'cold', 'burnt', 'not', 'forever', 'still', 'yet', 'didnt', 'rude'}

        # Temporal disappointment words
        expectation_words = {'waited', 'waiting', 'only', 'still', 'yet', 'finally', 'forever'}

        has_sarcasm = bool(query_words & sarcasm_markers)
        has_negative = bool(query_words & negative_context)
        has_expectation = bool(query_words & expectation_words)

        # Sarcasm with timing/expectation is almost always complaint
        if has_sarcasm and (has_negative or has_expectation):
            yield IntentAdjustment('complaint', SARCASM_COMPLAINT_BOOST, "Sarcasm complaint boost")

    def _rule_menu_item_ordering(self, context: BoostContext) -> Iterable[IntentAdjustment]:
        """
        RULE 8: Menu item/size mention suggests ordering
        Users mentioning specific pizzas/toppings often want to order
        """
        query_words = context.query_words
        has_menu_item = bool(query_words & self.menu_items)
        question_words = {'what', 'which', 'how', 'do', 'does', 'can', 'is', 'are', 'tell', 'show', 'whats'}
        has_question = bool(query_words & question_words)
        has_order_action = bool(query_words & self.synonyms.get('order', set()))
        has_sizer = bool(query_words & {'small', 'medium', 'large', 'one', 'two', 'three', 'four', 'five'})

        if has_menu_item and not has_question:
            yield IntentAdjustment('order', MENU_ITEM_ORDERING_BOOST, "Menu item ordering boost")
            yield IntentAdjustment('menu_inquiry',MENU_INQUIRY_PENALTY,
                "Menu inquiry penalty (ordering context)",
                is_penalty=True)

        elif (has_menu_item or has_sizer) and has_order_action and len(query_words) <= 5:
            yield IntentAdjustment('order', MENU_ITEM_ORDERING_BOOST, "Menu item ordering boost")
            yield IntentAdjustment('menu_inquiry',MENU_INQUIRY_PENALTY,
                "Menu inquiry penalty (ordering context)",
                is_penalty=True)

    def _rule_inquiry_pattern(self, context: BoostContext) -> Iterable[IntentAdjustment]:
        """
        RULE 9: Question about X Inquiry Pattern / Recommendation Pattern
        """
        query_words = context.query_words
        inquiry_words = {'question', 'asking', 'inquire', 'wondering', 'ask', 'tell', 'know', 'do'}
        about_words = {'about', 'regarding', 'can', 'concerning', 'have'}
        recommendation_words = self.synonyms.get('recommendation', set())
        menu_context_words = {'options', 'choices', 'selections', 'good', 'best', 'pizza', 'meal', 'deal'}

        has_inquiry = bool(query_words & inquiry_words)
        has_about = bool(query_words & about_words)
        has_recommendation = bool(query_words & recommendation_words)
        has_menu_context = bool(query_words & menu_context_words)
        is_recommendation_query = has_recommendation and has_menu_context
        has_negative = bool(query_words & self.negative_words)

        if has_inquiry and has_about:
            if query_words & self.synonyms.get('delivery', set()):
                yield IntentAdjustment('delivery', DELIVERY_INQUIRY_BOOST, "Delivery inquiry boost")
                yield IntentAdjustment('order',ORDER_INQUIRY_PENALTY,
                    "Order penalty for delivery inquiry",
                    is_penalty=True)
                yield IntentAdjustment('general',GENERAL_INQUIRY_PENALTY,
                    "General penalty for delivery inquiry",
                    is_penalty=True)

            if (query_words & self.synonyms.get('hours', set())) or (query_words & self.synonyms.get('location', set())):
                yield IntentAdjustment('hours_location', HOURS_INQUIRY_BOOST, "Hours/Location inquiry boost")
                yield IntentAdjustment('order',ORDER_INQUIRY_PENALTY,
                    "Order penalty for Hours/Location inquiry",
                    is_penalty=True)
                yield IntentAdjustment('general',GENERAL_INQUIRY_PENALTY,
                    "General penalty for Hours/Location inquiry",
                    is_penalty=True)

            price_related_words = {'menu'} | self.synonyms.get('price', set())
            if query_words & price_related_words and not has_negative:
                yield IntentAdjustment('menu_inquiry', PRICE_INQUIRY_BOOST, "Menu/Price inquiry boost")
                yield IntentAdjustment('order',ORDER_INQUIRY_PENALTY,
                    "Order penalty for price inquiry",
                    is_penalty=True)
                yield IntentAdjustment('hours_location',HOURS_MENU_PENALTY,
                    "Hours penalty for menu/price inquiry",
                    is_penalty=True)
                yield IntentAdjustment('general',GENERAL_INQUIRY_PENALTY,
                    "General penalty for Price inquiry",
                    is_penalty=True)

        if is_recommendation_query:
            yield IntentAdjustment('menu_inquiry', RECOMMENDATION_INQUIRY_BOOST, "Recommendation inquiry boost")
            yield IntentAdjustment('order',ORDER_INQUIRY_PENALTY,
                "Order penalty for recommendation inquiry",
                is_penalty=True)
            yield IntentAdjustment('delivery',DELIVERY_RECOMMENDATION_PENALTY,
                "Delivery penalty for recommendation inquiry",
                is_penalty=True)

    def _rule_menu_item_quality(self, context: BoostContext) -> Iterable[IntentAdjustment]:
        """
        RULE 10: Specific menu items + quality/customization = could be order or complaint
        """
        query_words = context.query_words
        has_menu_item = bool(query_words & self.menu_items)
        quality_issues = self.synonyms.get('quality', set())
        has_quality_issue = bool(query_words & quality_issues)

        customization_words = {'no', 'without', 'extra', 'add', 'remove', 'different', 'more', 'less'}
        has_customization = bool(query_words & customization_words)

        if has_menu_item:
            if has_quality_issue:
                yield IntentAdjustment('complaint', MENU_ITEM_QUALITY_BOOST, "Menu item quality issue boost")
                yield IntentAdjustment('menu_inquiry',MENU_INQUIRY_PENALTY,
                    "Menu inquiry penalty (quality complaint)",
                    is_penalty=True)
            elif has_customization:
                yield IntentAdjustment('order', MENU_ITEM_CUSTOMIZATION_BOOST, "Menu item customization boost")
                yield IntentAdjustment('menu_inquiry',ORDER_INQUIRY_PENALTY,
                    "Menu inquiry penalty (customization ordering)",
                    is_penalty=True)