# src/core/emotion/emotion_infer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import logging

logger = logging.getLogger("src.core.emotion.emotion_infer")


@dataclass
class EmotionResult:
    emotion: Optional[str]
    confidence: float
    backend: str
    is_uncertain: bool
    details: Dict[str, Any]


class HSEmotionDetector:
    def __init__(self, device: str = "cpu"):
        self._printed_error = False
        logger.info("Loading HSEmotion model...")
        from hsemotion.facial_emotions import HSEmotionRecognizer

        self.model = HSEmotionRecognizer(
            model_name="enet_b0_8_best_vgaf", device=device
        )
        logger.info("HSEmotion model loaded")

    def analyze(self, img_bgr: np.ndarray) -> Tuple[Optional[str], float]:
        if img_bgr is None or img_bgr.size == 0:
            return None, 0.0
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            emotion, scores = self.model.predict_emotions(img_rgb, logits=False)
            conf = float(np.max(scores)) if scores is not None else 0.0
            conf = max(0.0, min(1.0, conf))
            return emotion, conf
        except Exception:
            if not self._printed_error:
                self._printed_error = True
                logger.error("HSEmotion Error (logged once):", exc_info=True)
            return None, 0.0


class EmotionInfer:
    """
    Inférence temps réel HSEmotion only.

    Logique:
    - Si conf < uncertain_min_conf => Uncertain
    - Si conf >= hse_threshold => "hsemotion"
    - Sinon => "hsemotion_low" (on affiche quand même l'émotion)
    """

    def __init__(
        self,
        device: str = "cpu",
        hse_threshold: float = 0.65,
        enable_uncertain: bool = True,
        uncertain_min_conf: float = 0.55,
    ):
        self.hse_threshold = float(hse_threshold)
        self.enable_uncertain = bool(enable_uncertain)
        self.uncertain_min_conf = float(uncertain_min_conf)
        self.hse = HSEmotionDetector(device=device)

    def infer(self, face_bgr: np.ndarray) -> EmotionResult:
        hse_emotion, hse_conf = self.hse.analyze(face_bgr)
        hse_conf = float(hse_conf or 0.0)

        if self.enable_uncertain:
            if (hse_emotion is None) or (hse_conf < self.uncertain_min_conf):
                return EmotionResult(
                    emotion=None,
                    confidence=hse_conf,  # ✅ on garde la vraie confiance
                    backend="hsemotion_uncertain",
                    is_uncertain=True,
                    details={"hse_emotion": hse_emotion, "hse_conf": hse_conf},
                )

        # Seuil principal
        if hse_emotion is not None and hse_conf >= self.hse_threshold:
            return EmotionResult(
                emotion=hse_emotion,
                confidence=hse_conf,
                backend="hsemotion",
                is_uncertain=False,
                details={},
            )

        # fallback: on renvoie quand même l'émotion si dispo
        if hse_emotion is not None:
            return EmotionResult(
                emotion=hse_emotion,
                confidence=hse_conf,
                backend="hsemotion_low",
                is_uncertain=False,
                details={},
            )

        return EmotionResult(
            emotion=None,
            confidence=0.0,
            backend="none",
            is_uncertain=True,
            details={},
        )
