"""
Statistics Helper Utilities for Voice Assistant Services
Provides common statistics tracking patterns used across ASR, TTS, and Intent Recognition services.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Any


class StatisticsHelper:

    @staticmethod
    def init_base_stats(service='intent', **custom_fields) -> Dict[str, Any]:
        """Initialize statistics based on service type."""
        if service == 'intent':
            stats = {
                'total_queries': 0,
                'intent_distribution': defaultdict(int),
                'avg_confidence': [],
            }
        elif service in ['asr', 'tts']:
            stats = {'total_requests': 0}
        else:
            stats = {}

        stats.update(custom_fields)
        return stats

    @staticmethod
    def calculate_average(values_list: List[float]) -> float:
        """Calculate average from a list of values."""
        return sum(values_list) / len(values_list) if values_list else 0.0

    @staticmethod
    def calculate_success_rate(successful: int, total: int) -> float:
        """Calculate success rate as a ratio (0.0 to 1.0)."""
        return (successful / total) if total > 0 else 0.0

    @staticmethod
    def reset_stats(stats_dict: Dict, preserve_fields: Optional[List[str]] = None) -> None:
        """Reset statistics to initial state, optionally preserving certain fields."""
        preserved = {}
        if preserve_fields:
            for field in preserve_fields:
                if field in stats_dict:
                    preserved[field] = stats_dict[field]

        for key in list(stats_dict.keys()):
            if preserve_fields and key in preserve_fields:
                continue

            value = stats_dict[key]
            if isinstance(value, defaultdict):
                stats_dict[key] = defaultdict(int)
            elif isinstance(value, dict):
                stats_dict[key] = {}
            elif isinstance(value, list):
                stats_dict[key] = []
            elif isinstance(value, (int, float)):
                stats_dict[key] = 0

        stats_dict.update(preserved)