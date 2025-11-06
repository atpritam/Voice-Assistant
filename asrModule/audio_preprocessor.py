"""
Audio Preprocessor for ASR
Lightweight preprocessing for improved Whisper accuracy
"""

import numpy as np
from typing import Tuple, Optional
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.logger import ConditionalLogger
from utils.statistics import StatisticsHelper

try:
    import librosa
    import soundfile as sf
    import noisereduce as nr

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


class AudioPreprocessor:
    """
    Lightweight audio preprocessing for ASR
    Focus: noise reduction, normalization, silence trimming
    """

    def __init__(
            self,
            target_sample_rate: int = 16000,
            enable_noise_reduction: bool = True,
            enable_normalization: bool = True,
            enable_silence_trim: bool = True,
            noise_reduction_strength: float = 0.25,
            enable_logging: bool = False
    ):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "Required packages not installed. Install with:\n"
                "pip install librosa soundfile noisereduce"
            )

        self.target_sample_rate = target_sample_rate
        self.enable_noise_reduction = enable_noise_reduction
        self.enable_normalization = enable_normalization
        self.enable_silence_trim = enable_silence_trim
        self.noise_reduction_strength = noise_reduction_strength
        self.logger = ConditionalLogger(__name__, enable_logging)

        self.stats = StatisticsHelper.init_base_stats(
            service='audio_preprocessor',
            total_processed=0,
            total_processing_time=0.0,
            processing_times=[]
        )

    def process_audio_file(
            self,
            audio_path: str
    ) -> Tuple[Optional[str], float]:
        """
        Process audio file with preprocessing pipeline

        Args:
            audio_path: Path to input audio file

        Returns:
            Tuple of (processed_audio_path, processing_time)
        """
        start_time = time.time()

        try:
            audio, sr = librosa.load(audio_path, sr=None, mono=True)

            self.logger.debug(
                f"Loaded audio: duration={len(audio) / sr:.2f}s, "
                f"sample_rate={sr}Hz, samples={len(audio)}"
            )

            processed_audio = self._apply_preprocessing(audio, sr)

            output_path = audio_path.replace('.webm', '_processed.wav')
            sf.write(output_path, processed_audio, self.target_sample_rate)

            processing_time = time.time() - start_time

            self.stats['total_processed'] += 1
            self.stats['total_processing_time'] += processing_time
            self.stats['processing_times'].append(processing_time)

            self.logger.info(
                f"Audio pre-processed successfully in {processing_time * 1000:.1f}ms"
            )

            return output_path, processing_time

        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {e}")
            return None, 0.0

    def _apply_preprocessing(
            self,
            audio: np.ndarray,
            sr: int
    ) -> np.ndarray:
        """Apply preprocessing pipeline to audio"""

        if sr != self.target_sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=self.target_sample_rate
            )
            sr = self.target_sample_rate

        if self.enable_silence_trim:
            audio = self._trim_silence(audio, sr)

        if self.enable_noise_reduction:
            audio = self._reduce_noise(audio, sr)

        if self.enable_normalization:
            audio = self._normalize_audio(audio)

        return audio

    def _trim_silence(
            self,
            audio: np.ndarray,
            sr: int,
            top_db: int = 30
    ) -> np.ndarray:
        """
        Trim leading and trailing silence
        Fast operation, minimal latency impact
        """
        try:
            trimmed, _ = librosa.effects.trim(
                audio,
                top_db=top_db,
                frame_length=512,
                hop_length=128
            )

            reduction = (1 - len(trimmed) / len(audio)) * 100
            self.logger.debug(f"Silence trimmed: {reduction:.1f}% reduction")

            return trimmed

        except Exception as e:
            self.logger.warning(f"Silence trimming failed: {e}")
            return audio

    def _reduce_noise(
            self,
            audio: np.ndarray,
            sr: int
    ) -> np.ndarray:
        """
        Reduce background noise using spectral gating
        Optimized for speed with stationary noise reduction
        """
        try:
            reduced = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=True,
                prop_decrease=self.noise_reduction_strength,
                freq_mask_smooth_hz=500,
                time_mask_smooth_ms=50
            )

            noise_level = np.mean(np.abs(audio - reduced))
            self.logger.debug(f"Noise reduced: level={noise_level:.4f}")

            return reduced

        except Exception as e:
            self.logger.warning(f"Noise reduction failed: {e}")
            return audio

    def _normalize_audio(
            self,
            audio: np.ndarray,
            target_level: float = -20.0
    ) -> np.ndarray:
        """
        Normalize audio to target dBFS level
        Very fast operation
        """
        try:
            if len(audio) == 0:
                return audio

            current_db = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-10)

            gain_db = target_level - current_db
            gain_linear = 10 ** (gain_db / 20)

            normalized = audio * gain_linear

            normalized = np.clip(normalized, -1.0, 1.0)

            self.logger.debug(
                f"Audio normalized: {current_db:.1f}dB -> {target_level:.1f}dB"
            )

            return normalized

        except Exception as e:
            self.logger.warning(f"Normalization failed: {e}")
            return audio

    def get_statistics(self) -> dict:
        """Get audio preprocessing statistics"""
        avg_processing_time = StatisticsHelper.calculate_average(
            self.stats['processing_times']
        )

        return {
            'total_processed': self.stats['total_processed'],
            'total_processing_time': self.stats['total_processing_time'],
            'avg_processing_time_ms': avg_processing_time * 1000,
        }