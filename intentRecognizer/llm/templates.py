"""
LLM Prompt Templates and selective business info extraction
Contains all prompt templates for LLM-based intent recognition and response generation
"""

import json
from typing import List, Dict, Optional

# Intent category definitions
INTENT_DESCRIPTIONS = """- general: Greetings, thanks, confirmations, chitchat, general business questions
- order: Placing NEW orders, pickup, menu item customization, returning customers wanting order like last time
- delivery: Order status checks, tracking, "where is my order", delivery related questions, "when will it arrive?"
- menu_inquiry: Asking about prices, menu options, recommendations, deals, "what do you have", deciding items
- hours_location: Store hours, location, address
- complaint: Problems with order/food issues, wrong items/size, sarcastic complaints, service/delivery issues"""

# Order flow instructions
ORDER_FLOW = """Order flow:
        1. First let User picks a SPECIFIC pizza
        2. (ask size if missing)
        3. Offer sides/drinks
        4. Ask pickup or delivery
        5. Get name
        6. If delivery, ask user's address
        7. Confirm and close"""


def build_user_prompt(query: str, conversation_history: Optional[List[Dict]] = None) -> str:
    """Build user prompt with conversation context

    Args:
        query: Current user query
        conversation_history: Previous conversation context (last 8 messages)

    Returns:
        Formatted user prompt string
    """
    if not conversation_history:
        return f'CURRENT CUSTOMER QUERY: "{query}"'

    prompt_parts = ["CONVERSATION HISTORY:"]
    recent_history = conversation_history[-8:] if len(conversation_history) > 8 else conversation_history

    for entry in recent_history:
        role = "Customer" if entry['type'] == 'user' else "Assistant"
        intent_str = f" [intent: {entry['intent']}]" if entry.get('intent') else ""
        prompt_parts.append(f"{role}: {entry['message']}{intent_str}")

    prompt_parts.extend([
        "",
        "- Do NOT repeat information already mentioned",
        "- Track the order state and what has been collected",
        "- Respond naturally based on what is still needed",
        "- Do not ask multiple questions in one response",
        "",
        f'CURRENT QUERY: "{query}"'
    ])

    return "\n".join(prompt_parts)


def get_test_mode_prompt(valid_intents: List[str]) -> str:
    """Generate system prompt for test mode (classification only, no response)

    Args:
        valid_intents: List of valid intent names

    Returns:
        Test mode system prompt
    """
    prompt = f"""Intent classification for Pizza Restaurant. Classify into: {', '.join(valid_intents)}
{INTENT_DESCRIPTIONS}

Identify Primary intent in multi-intent query.
Respond ONLY with valid JSON, no markdown formatting:
{{"intent": "intent_name", "confidence": 0.85}}"""
    return prompt


def get_response_generation_prompt(
    business_name: str,
    recognized_intent: str,
    original_conf: float,
    selective_business_info: str
) -> str:
    """Generate system prompt for response generation mode (intent already known)

    Args:
        business_name: Name of the business
        recognized_intent: Intent recognized by previous layer
        original_conf: Confidence score from previous layer
        selective_business_info: Filtered business information relevant to the intent

    Returns:
        Response generation mode system prompt
    """
    prompt = f"""You are a helpful voice customer support assistant for {business_name}.
Customer Query Intent Classified as: {recognized_intent} ({original_conf:.2f})

Use INFO to reply in under 40 words.
INFO: {selective_business_info}

{ORDER_FLOW}

Return ONLY valid JSON (no markdown):
{{"response": "Short helpful reply"}}"""
    return prompt


def get_classification_prompt(
    business_name: str,
    business_type: str,
    valid_intents: List[str],
    business_info: str,
    recognized_intent: Optional[str] = None,
    original_conf: Optional[float] = None
) -> str:
    """Generate system prompt for classification mode (intent unknown or low confidence)

    Args:
        business_name: Name of the business
        business_type: Type of business (e.g., "Pizza Restaurant")
        valid_intents: List of valid intent names
        business_info: Full business information JSON
        recognized_intent: Intent suggested by previous layer (optional)
        original_conf: Confidence from previous layer (optional)

    Returns:
        Classification mode system prompt
    """
    valid_intents_list = ["general", "order", "delivery", "menu_inquiry", "hours_location", "complaint"]

    previous_layer_hint = ""
    if recognized_intent in valid_intents_list and original_conf is not None:
        previous_layer_hint = f"Previous layer Suggested (you can correct it if seems incorrect): {recognized_intent} ({original_conf:.2f})"

    prompt = f"""You are a helpful voice customer support assistant for {business_name}, a {business_type}.

        Classify INTENT into: {', '.join(valid_intents)}
        {INTENT_DESCRIPTIONS}

        {previous_layer_hint}

        Input could be a continuation of the previous query's intent. Previous Layer does not have Conversation History.

        Use INFO to generate RESPONSE in under 40 words.
        INFO: {business_info}

        {ORDER_FLOW}

        Return ONLY valid JSON (no markdown):
        {{"intent": "intent_name", "confidence": 0.83, "response": "Short helpful reply"}}

        CONFIDENCE:
        >=0.8 high | >=0.6 medium | <0.6 low"""

    return prompt

def get_selective_business_info(res_info: Dict, intent: str) -> str:
    """Return only business info relevant to the intent to reduce token usage

    Args:
        res_info: Full restaurant/business information dictionary
        intent: Recognized intent name

    Returns:
        JSON string with filtered business information
    """
    if intent == "order" or intent == "delivery":
        info = {
            "menu_highlights": res_info.get("menu_highlights"),
            "delivery": res_info.get("delivery"),
        }
    elif intent == "complaint":
        info = {
            "location": res_info.get("location"),
            "delivery": res_info.get("delivery")
        }
    elif intent == "hours_location":
        info = {
            "hours": res_info.get("hours"),
            "location": res_info.get("location")
        }
    elif intent == "menu_inquiry":
        info = {
            "menu_highlights": res_info.get("menu_highlights")
        }
    elif intent == "general":
        info = {
            "business_type": res_info.get("business_type"),
        }
    else:
        info = {}

    return json.dumps(info, indent=2)