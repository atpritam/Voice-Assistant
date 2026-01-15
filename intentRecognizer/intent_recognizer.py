"""
Main Intent Recognition System - Configurable Pipeline Architecture
Pipeline: Algorithmic → Semantic → LLM (default)
Each layer is tried when the previous layer fails or has below threshold confidence
"""

import os
import sys
import json
import random
import re

import torch
from typing import Dict, Optional, List
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.logger import ConditionalLogger
from utils.statistics import StatisticsHelper

HIGH_CONFIDENCE_THRESHOLD = 0.8               # "high" confidence level classification
MEDIUM_CONFIDENCE_THRESHOLD = 0.6             # "medium" confidence level classification
DEFAULT_MIN_CONFIDENCE = 0.5                  # Minimum confidence to accept intent (else "unknown")
DEFAULT_ALGORITHMIC_THRESHOLD = 0.65          # Min confidence for algorithmic layer to skip next layers
DEFAULT_SEMANTIC_THRESHOLD = 0.5              # Min confidence for semantic layer to skip LLM layer
DEFAULT_LLM_MODEL = "llama3.2:3b-instruct-q4_K_M"
DEFAULT_SEMANTIC_MODEL = "all-mpnet-base-v2"


@dataclass
class RecognitionResult:
    """Unified result from intent recognition"""
    intent: str
    confidence: float
    confidence_level: str
    matched_pattern: str
    processing_method: str
    layer_used: str
    response: str = ""
    score_breakdown: Dict = None


class IntentRecognizerUtils:
    """Shared utilities for all recognizer layers"""
    @staticmethod
    def determine_confidence_level(confidence: float) -> str:
        """Determine confidence level based on thresholds"""
        if confidence >= HIGH_CONFIDENCE_THRESHOLD:
            return 'high'
        elif confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
            return 'medium'
        return 'low'

    @staticmethod
    def load_patterns_from_file(patterns_file: str, enable_logging: bool = False) -> Dict:
        """Load intent patterns from JSON file"""
        logger = ConditionalLogger(__name__, enable_logging)
        try:
            with open(patterns_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('intents', data) if 'intents' in data else data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading patterns: {e}")
            return {}

    @staticmethod
    def get_default_patterns_file() -> str:
        """Get default path to patterns file"""
        utils_dir = os.path.join(os.path.dirname(__file__), '..', 'utils')
        return os.path.join(utils_dir, 'intent_patterns.json')

    @staticmethod
    def get_response_templates_file() -> str:
        """Get default path to response templates file"""
        utils_dir = os.path.join(os.path.dirname(__file__), '..', 'utils')
        return os.path.join(utils_dir, 'response_templates.json')

    @staticmethod
    def load_response_templates(templates_file: str, enable_logging: bool = False) -> Dict:
        """Load response templates (per-intent) from JSON file"""
        logger = ConditionalLogger(__name__, enable_logging)
        def _normalize_phrases(value):
            if not isinstance(value, list):
                return value
            return [p.strip().lower() for p in value if isinstance(p, str) and p.strip()]

        def _normalize_intent_cfg(cfg: dict) -> dict:
            if not isinstance(cfg, dict):
                return cfg

            new_cfg = dict(cfg)
            new_cfg['inclusion_phrases'] = _normalize_phrases(cfg.get('inclusion_phrases'))
            new_cfg['exclusion_phrases'] = _normalize_phrases(cfg.get('exclusion_phrases'))
            return new_cfg

        def _normalize_response_group(cfg):
            if not isinstance(cfg, dict):
                return cfg

            normalized = _normalize_intent_cfg(cfg)
            if 'intents' in cfg and isinstance(cfg['intents'], dict):
                normalized['intents'] = {
                    name: _normalize_intent_cfg(intent_cfg)
                    for name, intent_cfg in cfg['intents'].items()
                }
            return normalized

        try:
            with open(templates_file, 'r', encoding='utf-8') as f:
                raw = json.load(f)

            return {
                key: _normalize_response_group(cfg)
                for key, cfg in raw.items()
            }

        except FileNotFoundError:
            logger.info(f"Response templates file not found: {templates_file}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing response templates: {e}")
            return {}

from .algorithmic import AlgorithmicRecognizer
from .semantic import SemanticRecognizer
from .llm import LLMRecognizer


class IntentRecognizer:
    """
    Configurable Intent Recognition Pipeline

    Each layer can be independently enabled/disabled:
    - Layer 1: Algorithmic (Keyword pattern Matching + Levenshtein)
    - Layer 2: Semantic (Sentence Transformers)
    - Layer 3: LLM (Ollama Cloud/Local)
    """

    def __init__(
            self,
            patterns_file: str = None,
            enable_logging: bool = False,
            min_confidence: float = DEFAULT_MIN_CONFIDENCE,
            enable_algorithmic: bool = True,
            use_boost_engine: bool = True,
            enable_semantic: bool = True,
            enable_llm: bool = True,
            device: str = "auto",
            algorithmic_threshold: float = DEFAULT_ALGORITHMIC_THRESHOLD,
            semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
            semantic_model: str = DEFAULT_SEMANTIC_MODEL,
            llm_model: str = DEFAULT_LLM_MODEL,
            test_mode: bool = False,
            ollama_base_url: str = "http://localhost:11434",
    ):
        if not (enable_algorithmic or enable_semantic or enable_llm):
            raise ValueError("Invalid configuration: At least one layer must be enabled.")

        self.patterns_file = patterns_file or IntentRecognizerUtils.get_default_patterns_file()
        self.min_confidence = min_confidence
        self.enable_algorithmic = enable_algorithmic
        self.use_boost_engine = use_boost_engine
        self.enable_semantic = enable_semantic
        self.enable_llm = enable_llm
        self.device = device
        self.test_mode = test_mode
        self.algorithmic_threshold = algorithmic_threshold if (enable_semantic or enable_llm) else 0
        self.semantic_threshold = semantic_threshold if enable_llm else 0
        self.enable_logging = enable_logging
        self.ollama_base_url = ollama_base_url
        self.logger = ConditionalLogger(__name__, enable_logging)

        self.stats = StatisticsHelper.init_base_stats(
            algorithmic_used=0,
            semantic_used=0,
            llm_used=0,
            layer_configuration={
                'algorithmic': enable_algorithmic,
                'semantic': enable_semantic,
                'llm': enable_llm
            }
        )

        self.patterns = IntentRecognizerUtils.load_patterns_from_file(self.patterns_file, self.enable_logging)
        self.response_templates = IntentRecognizerUtils.load_response_templates(
            IntentRecognizerUtils.get_response_templates_file(), self.enable_logging
        )

        self.algorithmic_recognizer = None
        self.semantic_recognizer = None
        self.llm_recognizer = None

        self._initialize_layers(semantic_model, llm_model)

    def _initialize_layers(self, semantic_model: str, llm_model: str):
        """Initialize all enabled layers"""
        if self.enable_algorithmic:
            self.algorithmic_recognizer = AlgorithmicRecognizer(
                patterns_file=self.patterns_file,
                enable_logging=self.enable_logging,
                min_confidence=self.min_confidence,
                use_boost_engine=self.use_boost_engine,
                algorithmic_threshold=self.algorithmic_threshold,
            )
            self.logger.info(" Algorithmic layer initialized")

        if self.enable_semantic:
            device = self.device if self.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
            try:
                self.semantic_recognizer = SemanticRecognizer(
                    patterns_file=self.patterns_file,
                    model_name=semantic_model,
                    device=device,
                    enable_logging=self.enable_logging,
                    min_confidence=self.min_confidence,
                    use_cache=True
                )
                self.logger.info(f" Semantic layer initialized (model: {semantic_model})")
            except ImportError as e:
                self.logger.error(f"Semantic layer failed - missing dependencies: {e}\n  Install with: pip install sentence-transformers scikit-learn")
                raise RuntimeError("Cannot initialize semantic layer - missing dependencies. Install with: pip install sentence-transformers scikit-learn")
            except Exception as e:
                self.logger.error(f" Semantic layer initialization failed: {e}")
                raise

        if self.enable_llm:
            try:
                self.llm_recognizer = LLMRecognizer(
                    model=llm_model,
                    enable_logging=self.enable_logging,
                    min_confidence=self.min_confidence,
                    ollama_base_url=self.ollama_base_url,
                    test_mode=self.test_mode
                )
                model_type = "Cloud" if llm_model.endswith('-cloud') else "Local"
                self.logger.info(f" LLM layer initialized (Ollama-{model_type}, model: {llm_model})")
            except Exception as e:
                self.logger.error(f" LLM layer initialization failed: {e}")
                raise

    def recognize_intent(self, query: str, conversation_history: Optional[List[Dict]] = None) -> RecognitionResult:
        """
        Main recognition method - routes through enabled layers

        Pipeline Flow:
        1. Try first enabled layer
        2. If confidence below threshold, try next enabled layer
        3. Return best result with generated response
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        self.stats['total_queries'] += 1
        self.logger.info("-" * 60)
        self.logger.info(f"Processing: '{query}'")
        self.logger.info("-" * 60)

        recognized_intent = None
        recognized_confidence = None

        if self.enable_algorithmic:
            result, algo_result = self._try_layer('algorithmic', self.algorithmic_recognizer, query,
                                    self.algorithmic_threshold, conversation_history)
            if result:
                return result
            if algo_result and algo_result.intent != "unknown":
                recognized_intent = algo_result.intent
                recognized_confidence = algo_result.confidence

        if self.enable_semantic:
            result, semantic_result = self._try_layer('semantic', self.semantic_recognizer, query,
                                    self.semantic_threshold, conversation_history)
            if result:
                return result
            if semantic_result and semantic_result.intent != "unknown":
                recognized_intent = semantic_result.intent
                recognized_confidence = semantic_result.confidence

        if self.enable_llm:
            result = self._try_llm_layer(query, conversation_history, recognized_intent, recognized_confidence)
            if result:
                return result

        return self._create_unknown_result("No layers produced a result")

    def _log_layer_acceptance(self, layer_name: str, result) -> None:
        """Log and track layer acceptance"""
        self.stats[f'{layer_name}_used'] += 1
        self.logger.info(
                f"✓ ACCEPTED {layer_name.upper()} Layer: {result.intent} "
                f"({result.confidence:.3f}, {result.confidence_level})"
            )

    def _try_layer(self, layer_name: str, recognizer, query: str, threshold: float,
                   conversation_history: Optional[List[Dict]] = None) -> tuple[Optional[RecognitionResult], any]:
        """Generic layer trying logic for algorithmic and semantic layers"""
        result = recognizer.recognize(query)

        if result.intent == "unknown" or result.confidence < threshold:
            next_layer = "Semantic" if layer_name == "algorithmic" and self.enable_semantic else "LLM"
            if (layer_name == "algorithmic" and self.enable_semantic) or (layer_name == "semantic" and self.enable_llm):
                self.logger.info(f"  - Proceeding to {next_layer} layer")
            return None, result

        self._log_layer_acceptance(layer_name, result)
        final_result = self._create_result(query, result, layer=layer_name,
                                  conversation_history=conversation_history,
                                  original_confidence=result.confidence)
        return final_result, result

    def _try_llm_layer(self, query: str, conversation_history: Optional[List[Dict]] = None,
                      recognized_intent: Optional[str] = None, recognized_confidence: Optional[float] = None) -> RecognitionResult:
        """Try LLM fallback layer"""
        llm_result = self.llm_recognizer.recognize(
            query, self.patterns, conversation_history, 
            recognized_intent, recognized_confidence,
            classifier=True  # LLM acts as classifier
        )
        self._log_layer_acceptance('llm', llm_result)
        return self._create_result(query, llm_result, layer='llm', conversation_history=conversation_history)

    def _should_use_llm_for_response(self, query: str, intent: str) -> tuple[bool, Optional[str]]:
        """Decide whether to use LLM or Template (when response_templates.json is available) for response generation."""
        cfg = getattr(self, 'response_templates', {}) or {}
        query_lower = query.lower()
        normalized = re.sub(r'[^\w\s]', '', query_lower).strip()

        # Check top-level greeting/acknowledgement groups for direct response
        for group in ('greetings', 'acknowledgements'):
            grp = cfg.get(group)
            if grp and isinstance(grp, dict) and grp.get('use_template'):
                inc = grp.get('inclusion_phrases', []) or []
                if normalized in inc:
                    templates = grp.get('templates', []) or []
                    return False, (random.choice(templates) if templates else None)

        intent_cfg = (cfg.get('intents', {}) or {}).get(intent) or cfg.get(intent)
        if not intent_cfg:
            return True, None
        if not intent_cfg.get('use_template', False):
            return True, None
        multi_intent_keywords = {'but', 'also', 'and', 'plus', 'however', 'though'}
        if any(kw in query_lower for kw in multi_intent_keywords):
            return True, None
        exclusion_phrases = intent_cfg.get('exclusion_phrases', []) or []
        if any(phrase in query_lower for phrase in exclusion_phrases):
            return True, None
        templates = intent_cfg.get('templates', []) if isinstance(intent_cfg, dict) else []
        return False, (random.choice(templates) if templates else None)


    def _create_result(self, query: str, result, layer: str,
                       conversation_history: Optional[List[Dict]] = None,
                       original_confidence: float = None) -> RecognitionResult:
        """Create unified RecognitionResult from any layer with LLM for response generation"""
        original_conf = original_confidence if original_confidence is not None else result.confidence
        response = getattr(result, 'response', '')

        if self.test_mode:
            response = ""
        elif not response and self.enable_llm and layer != 'llm':
            use_llm, template_response = self._should_use_llm_for_response(query, result.intent)

            if not use_llm:
                if template_response:
                    response = template_response
                    self.logger.info("  - Using template response")
                else:
                    response = self._generate_simple_response(result.intent)
                    self.logger.info("  - Using fallback template from patterns file")
            else:
                self.logger.info("  - Using LLM for response generation")

                try:
                    llm_result = self.llm_recognizer.recognize(
                        query, self.patterns, conversation_history, 
                        result.intent, original_conf,
                        classifier=False   # LLM only generates response
                    )

                    response = llm_result.response

                    if not response or not response.strip():
                        self.logger.warning("LLM returned empty response, using fallback")
                        response = self._generate_simple_response(result.intent)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"LLM response JSON parse error: {e}, using fallback")
                    response = self._generate_simple_response(result.intent)
                except Exception as e:
                    self.logger.warning(f"LLM response generation failed: {e}, using fallback")
                    response = self._generate_simple_response(result.intent)
        elif not response and not self.test_mode:
            response = self._generate_simple_response(result.intent)

        self.stats['intent_distribution'][result.intent] += 1
        self.stats['avg_confidence'].append(result.confidence)

        return RecognitionResult(
            intent=result.intent,
            confidence=result.confidence,
            confidence_level=result.confidence_level,
            matched_pattern=getattr(result, 'matched_pattern', 'ML Classification'),
            processing_method=getattr(result, 'processing_method', layer),
            layer_used=layer,
            response=response,
            score_breakdown=getattr(result, 'score_breakdown', {})
        )

    def _generate_simple_response(self, intent: str) -> str:
        """Generate simple response for non-LLM layers from pattern file"""
        if intent in self.patterns:
            return self.patterns[intent].get('default_response',
                                             "I'm here to help! Could you please clarify what you'd like to know?")
        return "I'm here to help! Could you please clarify what you'd like to know?"

    def _create_unknown_result(self, reason: str) -> RecognitionResult:
        """Create unknown result with given reason"""
        return RecognitionResult(
            intent='unknown',
            confidence=0.0,
            confidence_level='low',
            matched_pattern='',
            processing_method='error',
            layer_used='none',
            response="I'm here to help! Could you please clarify what you'd like to know?",
            score_breakdown={'error': reason}
        )

    def evaluate(self, test_data: list) -> Dict:
        """Evaluate recognizer accuracy on test data"""
        results = []
        correct = 0

        for query, expected_intent in test_data:
            result = self.recognize_intent(query)
            is_correct = result.intent == expected_intent
            if is_correct:
                correct += 1

            results.append({
                'query': query,
                'expected': expected_intent,
                'predicted': result.intent,
                'confidence': result.confidence,
                'layer_used': result.layer_used,
                'correct': is_correct,
                'score_breakdown': result.score_breakdown
            })

        total = len(test_data)
        accuracy = correct / total if total > 0 else 0.0

        high_conf = [r for r in results if r['confidence'] >= HIGH_CONFIDENCE_THRESHOLD]
        medium_conf = [r for r in results if MEDIUM_CONFIDENCE_THRESHOLD <= r['confidence'] < HIGH_CONFIDENCE_THRESHOLD]
        low_conf = [r for r in results if r['confidence'] < MEDIUM_CONFIDENCE_THRESHOLD]

        layer_results = {
            'llm': [r for r in results if r['layer_used'] == 'llm'],
            'semantic': [r for r in results if r['layer_used'] == 'semantic'],
            'algo': [r for r in results if r['layer_used'] == 'algorithmic']
        }

        return {
            'accuracy': accuracy,
            'total_queries': total,
            'correct': correct,
            'incorrect': total - correct,
            'high_confidence_count': len(high_conf),
            'medium_confidence_count': len(medium_conf),
            'low_confidence_count': len(low_conf),
            'high_confidence_accuracy': sum(r['correct'] for r in high_conf) / len(high_conf) if high_conf else 0,
            'llm_used_count': len(layer_results['llm']),
            'semantic_used_count': len(layer_results['semantic']),
            'algo_used_count': len(layer_results['algo']),
            'llm_accuracy': sum(r['correct'] for r in layer_results['llm']) / len(layer_results['llm']) if layer_results['llm'] else 0,
            'semantic_accuracy': sum(r['correct'] for r in layer_results['semantic']) / len(layer_results['semantic']) if layer_results['semantic'] else 0,
            'algo_accuracy': sum(r['correct'] for r in layer_results['algo']) / len(layer_results['algo']) if layer_results['algo'] else 0,
            'detailed_results': results
        }

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics from the pipeline"""
        avg_conf = round(StatisticsHelper.calculate_average(self.stats['avg_confidence']), 3)
        total = self.stats['total_queries']

        stats_dict = {
            'pipeline_configuration': {
                'algorithmic_enabled': self.enable_algorithmic,
                'semantic_enabled': self.enable_semantic,
                'llm_enabled': self.enable_llm,
                'algorithmic_threshold': self.algorithmic_threshold if self.enable_algorithmic else None,
                'semantic_threshold': self.semantic_threshold if self.enable_semantic else None,
            },
            'total_queries_processed': total,
            'layer_usage': {
                'algorithmic': {
                    'count': self.stats['algorithmic_used'],
                    'percentage': round((self.stats['algorithmic_used'] / total * 100), 3) if total > 0 else 0.0
                },
                'semantic': {
                    'count': self.stats['semantic_used'],
                    'percentage': round((self.stats['semantic_used'] / total * 100), 3) if total > 0 else 0.0
                },
                'llm': {
                    'count': self.stats['llm_used'],
                    'percentage': round((self.stats['llm_used'] / total * 100), 3) if total > 0 else 0.0
                }
            },
            'intent_distribution': dict(self.stats['intent_distribution']),
            'average_confidence': avg_conf
        }

        if self.enable_algorithmic and self.algorithmic_recognizer:
            stats_dict['algorithmic_layer'] = self.algorithmic_recognizer.get_statistics()

        if self.enable_semantic and self.semantic_recognizer:
            try:
                stats_dict['semantic_layer'] = self.semantic_recognizer.get_statistics()
            except Exception as e:
                self.logger.warning(f"Could not get semantic statistics: {e}")

        if self.enable_llm and self.llm_recognizer:
            try:
                stats_dict['llm_layer'] = self.llm_recognizer.get_statistics()
            except Exception as e:
                self.logger.warning(f"Could not get LLM statistics: {e}")

        return stats_dict

    def reset_statistics(self) -> None:
        """Reset all statistics counters across all layers"""
        StatisticsHelper.reset_stats(self.stats, preserve_fields=['layer_configuration'])
        if self.enable_algorithmic and self.algorithmic_recognizer:
            StatisticsHelper.reset_stats(self.algorithmic_recognizer.stats)
        if self.enable_semantic and self.semantic_recognizer:
            StatisticsHelper.reset_stats(self.semantic_recognizer.stats)
        if self.enable_llm and self.llm_recognizer:
            StatisticsHelper.reset_stats(self.llm_recognizer.stats, preserve_fields=['llm_provider'])