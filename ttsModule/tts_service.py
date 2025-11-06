"""
Text-to-Speech Service using Coqui TTS VITS Model
Optimized for en/ljspeech/vits - single speaker
"""

import os
import sys
import tempfile
import contextlib
import io
from pathlib import Path
from typing import Optional
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.logger import ConditionalLogger
from utils.statistics import StatisticsHelper

try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


class TTSService:
    """Text-to-Speech service using VITS model for natural speech generation"""

    def __init__(
            self,
            model_name: str = "tts_models/en/ljspeech/vits",
            enable_logging: bool = False,
            device: str = "auto",
            output_dir: str = None
    ):
        if not TTS_AVAILABLE:
            raise ImportError(
                "TTS library not installed. Install with: pip install TTS torch torchaudio"
            )

        self.model_name = model_name
        self.tts_device = device
        self.logger = ConditionalLogger(__name__, enable_logging)

        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path(tempfile.gettempdir()) / "voice-assistant-tts"
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tts = None
        self.stats = StatisticsHelper.init_base_stats(
            service='asr-tts',
            successful_generations=0,
            failed_generations=0
        )

        self._initialize_tts()
        self._cleanup_on_start()

    def _initialize_tts(self):
        """Initialize VITS TTS model"""
        try:
            self.logger.info(f"Initializing TTS model: {self.model_name}")

            if self.tts_device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.tts_device

            import warnings
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', category=FutureWarning)

            # Suppress TTS verbose initialization output
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.tts = TTS(model_name=self.model_name, progress_bar=False).to(device)

            self.logger.info(f"TTS model initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize TTS model: {e}")
            raise RuntimeError(f"TTS initialization failed: {e}")

    def _cleanup_on_start(self):
        """Clean up audio directory on service initialization"""
        try:
            if self.output_dir.exists():
                audio_files = list(self.output_dir.glob("*.wav"))
                deleted_count = 0

                for file_path in audio_files:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to delete {file_path}: {e}")

                if deleted_count > 0:
                    self.logger.info(f"Cleaned up {deleted_count} old audio files on startup")
        except Exception as e:
            self.logger.error(f"Startup cleanup failed: {e}")

    def generate_speech(
            self,
            text: str,
            output_filename: Optional[str] = None,
            speed: float = 1.2
    ) -> Optional[str]:
        """
        Generate speech from text using VITS model

        Args:
            text: Text to convert to speech
            output_filename: Custom output filename (optional)
            speed: Speech speed multiplier (default: 1.1)

        Returns:
            Path to generated audio file or None if failed
        """
        self.stats['total_requests'] += 1

        if not text or not text.strip():
            self.logger.warning("Empty text provided for TTS generation")
            self.stats['failed_generations'] += 1
            return None

        try:
            import warnings
            warnings.filterwarnings('ignore')

            if output_filename:
                output_path = self.output_dir / output_filename
            else:
                output_path = self.output_dir / f"tts_output_{self.stats['total_requests']}.wav"

            # Suppress TTS verbose output during generation
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.tts.tts_to_file(
                    text=text,
                    file_path=str(output_path),
                    speed=speed
                )

            self.stats['successful_generations'] += 1

            preview = text[:40] + "..." if len(text) > 40 else text
            self.logger.info(f"Generated: '{preview}' -> {output_path.name}")

            return str(output_path)

        except Exception as e:
            self.logger.error(f"TTS generation failed: {e}")
            self.stats['failed_generations'] += 1
            return None

    def get_statistics(self) -> dict:
        """Get TTS service statistics"""
        return StatisticsHelper.build_stats_response(
            self.stats,
            success_rate=StatisticsHelper.calculate_success_rate(
                self.stats['successful_generations'],
                self.stats['total_requests']
            )
        )

    def reset_statistics(self):
        """Reset TTS service statistics"""
        StatisticsHelper.reset_stats(self.stats)