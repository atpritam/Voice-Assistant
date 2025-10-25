"""
Text-to-Speech Service using Coqui TTS VITS Model
Optimized for en/ljspeech/vits - single speaker
"""

import logging
import tempfile
import re
import contextlib
import io
from pathlib import Path
from typing import Optional
import torch

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
        self.enable_logging = enable_logging
        self.tts_device = device

        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path(tempfile.gettempdir()) / "voice-assistant-tts"
            self.output_dir.mkdir(parents=True, exist_ok=True)

        if enable_logging:
            self.logger = logging.getLogger(__name__)

        self.tts = None
        self.stats = {
            'total_requests': 0,
            'successful_generations': 0,
            'failed_generations': 0,
        }

        self._initialize_tts()
        self._cleanup_on_start()

    def _initialize_tts(self):
        """Initialize VITS TTS model"""
        try:
            if self.enable_logging:
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

            if self.enable_logging:
                self.logger.info(f"TTS model initialized successfully")

        except Exception as e:
            if self.enable_logging:
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
                        if self.enable_logging:
                            self.logger.warning(f"Failed to delete {file_path}: {e}")

                if self.enable_logging and deleted_count > 0:
                    self.logger.info(f"Cleaned up {deleted_count} old audio files on startup")
        except Exception as e:
            if self.enable_logging:
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
            if self.enable_logging:
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

            if self.enable_logging:
                preview = text[:40] + "..." if len(text) > 40 else text
                self.logger.info(f"Generated: '{preview}' -> {output_path.name}")

            return str(output_path)

        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"TTS generation failed: {e}")
            self.stats['failed_generations'] += 1
            return None

    def get_statistics(self) -> dict:
        """Get TTS service statistics"""
        success_rate = (
            self.stats['successful_generations'] / self.stats['total_requests']
            if self.stats['total_requests'] > 0
            else 0.0
        )

        return {
            'total_requests': self.stats['total_requests'],
            'successful_generations': self.stats['successful_generations'],
            'failed_generations': self.stats['failed_generations'],
            'success_rate': success_rate,
        }