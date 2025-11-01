"""
Custom Logger Configuration for Voice Assistant
"""

import logging
import sys


class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    INTENT = '\033[96m'
    ALGORITHMIC = '\033[92m'
    SEMANTIC = '\033[93m'
    LLM = '\033[95m'
    TTS = '\033[94m'
    ASR = '\033[91m'
    APP = '\033[97m'
    TIME = '\033[90m'

    ERROR = '\033[91m'
    WARNING = '\033[93m'
    INFO = '\033[0m'
    DEBUG = '\033[90m'


class CleanFormatter(logging.Formatter):
    """Custom formatter with clean, readable output using short module names"""

    MODULE_NAMES = {
        'intentRecognizer.intent_recognizer': ('IntentRecognizer', Colors.INTENT),
        'intentRecognizer.algorithmic_recognizer': ('Algorithmic', Colors.ALGORITHMIC),
        'intentRecognizer.semantic_recognizer': ('Semantic', Colors.SEMANTIC),
        'intentRecognizer.llm_recognizer': ('LLM', Colors.LLM),
        'ttsModule.tts_service': ('TTS', Colors.TTS),
        'asrModule.asr_service': ('ASR', Colors.ASR),
        '__main__': ('App', Colors.APP),
        'app': ('App', Colors.APP),
    }

    def format(self, record):
        if record.name in self.MODULE_NAMES:
            module_name, color = self.MODULE_NAMES[record.name]
        else:
            module_name = record.name.split('.')[-1]
            color = Colors.RESET

        if record.levelno >= logging.ERROR:
            level_color = Colors.ERROR
        elif record.levelno >= logging.WARNING:
            level_color = Colors.WARNING
        elif record.levelno >= logging.INFO:
            level_color = Colors.INFO
        else:
            level_color = Colors.DEBUG

        timestamp = self.formatTime(record, '%Y-%m-%d %H:%M:%S')
        log_msg = f"{Colors.TIME}{timestamp}{Colors.RESET} {color}[{module_name}]{Colors.RESET} {level_color}{record.getMessage()}{Colors.RESET}"

        return log_msg


def setup_logging(level=logging.INFO):
    """
    Setup clean logging configuration for the entire application

    Args:
        level: Logging level (default: logging.INFO)
    """

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CleanFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    root_logger.handlers.clear()

    root_logger.addHandler(handler)

    # verbose third-party loggers
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('TTS').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)