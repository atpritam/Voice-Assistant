"""
Automatic Speech Recognition Service using OpenAI Whisper-tiny
With integrated audio preprocessing for improved accuracy
"""

import os
import logging
import tempfile
from typing import Optional, Tuple
import time
import torch
import math

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

from .audio_preprocessor import AudioPreprocessor

class ASRService:
    """Automatic Speech Recognition service using Whisper-tiny with preprocessing"""

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "auto",
        enable_logging: bool = False,
        enable_preprocessing: bool = True,
        noise_reduction_strength: float = 0.25
    ):
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "Whisper not installed. Install with: pip install openai-whisper"
            )

        self.model_size = model_size
        self.device = self._determine_device(device)
        self.enable_logging = enable_logging
        self.enable_preprocessing = enable_preprocessing

        if enable_logging:
            self.logger = logging.getLogger(__name__)

        self.model = None
        self.preprocessor = None

        self.stats = {
            'total_requests': 0,
            'successful_transcriptions': 0,
            'failed_transcriptions': 0,
            'total_processing_time': 0.0,
            'total_preprocessing_time': 0.0,
            'total_transcription_time': 0.0
        }

        self._initialize_model()

        if enable_preprocessing:
            self._initialize_preprocessor(noise_reduction_strength)

    def _determine_device(self, device: str) -> str:
        """Determine the best available device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _initialize_model(self):
        """Initialize Whisper model"""
        try:
            if self.enable_logging:
                self.logger.info(f"Initializing Whisper model: {self.model_size}")

            self.model = whisper.load_model(self.model_size, device=self.device)

            if self.enable_logging:
                self.logger.info("Whisper model initialized successfully")

        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Failed to initialize Whisper model: {e}")
            raise RuntimeError(f"Whisper initialization failed: {e}")

    def _initialize_preprocessor(self, noise_reduction_strength: float):
        """Initialize audio preprocessor"""
        try:
            self.preprocessor = AudioPreprocessor(
                target_sample_rate=16000,
                enable_noise_reduction=True,
                enable_normalization=True,
                enable_silence_trim=True,
                noise_reduction_strength=noise_reduction_strength,
                enable_logging=self.enable_logging
            )

            if self.enable_logging:
                self.logger.info("Audio preprocessor initialized")

        except ImportError as e:
            if self.enable_logging:
                self.logger.warning(
                    f"Audio preprocessor initialization failed: {e}\n"
                    "Install with: pip install librosa soundfile noisereduce\n"
                    "Continuing without preprocessing..."
                )
            self.enable_preprocessing = False
            self.preprocessor = None
        except Exception as e:
            if self.enable_logging:
                self.logger.warning(
                    f"Audio preprocessor initialization failed: {e}\n"
                    "Continuing without preprocessing..."
                )
            self.enable_preprocessing = False
            self.preprocessor = None

    def transcribe_audio(
        self,
        audio_path: str
    ) -> Tuple[Optional[str], float]:
        """
        Transcribe audio file to text

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (transcribed_text, confidence)
        """
        self.stats['total_requests'] += 1

        if not os.path.exists(audio_path):
            if self.enable_logging:
                self.logger.error(f"Audio file not found: {audio_path}")
            self.stats['failed_transcriptions'] += 1
            return None, 0.0

        try:
            total_start_time = time.time()
            processed_path = audio_path

            if self.enable_preprocessing and self.preprocessor:
                processed_path, preprocess_time = self.preprocessor.process_audio_file(audio_path)

                if processed_path is None:
                    if self.enable_logging:
                        self.logger.warning("Preprocessing failed, using original audio")
                    processed_path = audio_path
                else:
                    self.stats['total_preprocessing_time'] += preprocess_time
                    if self.enable_logging:
                        self.logger.debug(f"Preprocessing completed in {preprocess_time*1000:.1f}ms")

            transcription_start = time.time()
            result = self.model.transcribe(
                processed_path,
                language="en",
                fp16=False if self.device == "cpu" else True
            )
            transcription_time = time.time() - transcription_start
            self.stats['total_transcription_time'] += transcription_time

            transcribed_text = result["text"].strip()

            if not transcribed_text:
                if self.enable_logging:
                    self.logger.warning("No speech detected in audio")
                self.stats['failed_transcriptions'] += 1

                if processed_path != audio_path and os.path.exists(processed_path):
                    try:
                        os.unlink(processed_path)
                    except:
                        pass

                return None, 0.0

            segments = result.get("segments", [])
            avg_logprob = sum(s.get("avg_logprob", -1.0) for s in segments) / len(segments) if segments else -1.0
            confidence = self._normalize_confidence(avg_logprob)

            total_processing_time = time.time() - total_start_time

            self.stats['successful_transcriptions'] += 1
            self.stats['total_processing_time'] += total_processing_time

            if self.enable_logging:
                preview = transcribed_text[:40] + "..." if len(transcribed_text) > 40 else transcribed_text
                self.logger.info(
                    f"'{preview}' ({confidence:.2f}, {total_processing_time*1000:.0f}ms)"
                )

            if processed_path != audio_path and os.path.exists(processed_path):
                try:
                    os.unlink(processed_path)
                except:
                    pass

            return transcribed_text, confidence

        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Transcription failed: {e}")
            self.stats['failed_transcriptions'] += 1
            return None, 0.0

    def transcribe_audio_data(
        self,
        audio_data: bytes
    ) -> Tuple[Optional[str], float]:
        """
        Transcribe audio from raw bytes

        Args:
            audio_data: Raw audio bytes

        Returns:
            Tuple of (transcribed_text, confidence)
        """
        try:
            with tempfile.NamedTemporaryFile(
                suffix='.webm',
                delete=False,
                mode='wb'
            ) as temp_file:
                temp_file.write(audio_data)
                temp_file.flush()
                temp_path = temp_file.name

            try:
                result = self.transcribe_audio(temp_path)
                return result
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except OSError as e:
                        if self.enable_logging:
                            self.logger.warning(f"Failed to delete temp file {temp_path}: {e}")

        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Failed to transcribe audio data: {e}")
            return None, 0.0

    def _normalize_confidence(self, avg_logprob: float) -> float:
        """
        Normalize log probability to 0-1 confidence score
        """
        if avg_logprob == 0:
            return 0.5
        normalized = math.exp(avg_logprob)
        return min(max(normalized, 0.0), 1.0)

    def get_statistics(self) -> dict:
        """Get ASR service statistics"""
        success_rate = (
            self.stats['successful_transcriptions'] / self.stats['total_requests']
            if self.stats['total_requests'] > 0
            else 0.0
        )

        avg_total_time = (
            self.stats['total_processing_time'] / self.stats['successful_transcriptions']
            if self.stats['successful_transcriptions'] > 0
            else 0.0
        )

        stats_dict = {
            'total_requests': self.stats['total_requests'],
            'successful_transcriptions': self.stats['successful_transcriptions'],
            'failed_transcriptions': self.stats['failed_transcriptions'],
            'success_rate': success_rate,
            'total_processing_time': self.stats['total_processing_time'],
            'avg_total_processing_time_ms': avg_total_time * 1000,
            'preprocessing_enabled': self.enable_preprocessing
        }

        return stats_dict