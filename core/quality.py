"""
Quality assessment for pose detection.

This module evaluates how reliable a pose detection is, based on:
- Number of visible keypoints
- Keypoint confidence scores
- Presence of critical body parts (torso)
- Temporal stability (jitter)

A low quality score means the system should be cautious about making
definitive risk assessments. The quality score directly influences
whether NEEDS_HELP can be confirmed.

Key Design Decision:
- Missing torso (shoulders + hips) results in SEVERE quality penalty
- Quality < 0.4 prevents NEEDS_HELP confirmation (unless extreme geometry)
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import yaml
from pathlib import Path

# Import geometry utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.geometry import (
    compute_jitter, bbox_diagonal, point_is_valid, 
    midpoint_if_valid, distance
)

# Type aliases
Keypoint = Tuple[float, float, float]  # (x, y, confidence)
BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


@dataclass
class QualityResult:
    """
    Result of quality assessment for a pose detection.
    
    Attributes:
        score: Overall quality score [0, 1] where 1 = perfect quality
        has_torso: Whether torso keypoints are visible and reliable
        n_valid_keypoints: Number of keypoints above confidence threshold
        jitter: Normalized temporal jitter [0, inf) where 0 = stable
        reasons: List of reasons affecting quality (for debugging/logging)
    """
    score: float
    has_torso: bool
    n_valid_keypoints: int
    jitter: float
    reasons: List[str]
    
    def is_sufficient_for_confirmation(self, min_quality: float = 0.4) -> bool:
        """
        Check if quality is sufficient to confirm NEEDS_HELP.
        
        Args:
            min_quality: Minimum quality threshold (from config)
        
        Returns:
            True if quality is high enough to trust a NEEDS_HELP decision
        """
        return self.score >= min_quality


# COCO keypoint indices for torso detection
KEYPOINT_INDICES = {
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_hip': 11,
    'right_hip': 12,
}


class QualityAssessor:
    """
    Assesses the quality/reliability of pose detections.
    
    Quality is critical for:
    1. Deciding whether to trust a risk classification
    2. Influencing the temporal confirmation window
    3. Triggering the scheduler to improve observation
    
    The assessor penalizes heavily for:
    - Missing torso keypoints (can't reliably assess posture)
    - High jitter (unreliable tracking or rapid movement)
    - Low keypoint count (insufficient information)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the quality assessor.
        
        Args:
            config_path: Path to thresholds.yaml (optional)
        """
        self.config = self._load_config(config_path)
        self.prev_keypoints: Optional[List[Optional[Keypoint]]] = None
        self.prev_bbox: Optional[BBox] = None
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML or use defaults."""
        defaults = {
            'min_keypoints': 6,
            'min_confidence': 0.5,
            'torso_required': True,
            'torso_missing_penalty': 0.2,
            'jitter_penalty_threshold': 0.1,
            'jitter_penalty_factor': 0.3,
            'min_quality_for_confirmation': 0.4,
            'extreme_geometry_override': 0.25,
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    full_config = yaml.safe_load(f)
                    if 'quality' in full_config:
                        defaults.update(full_config['quality'])
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        return defaults
    
    def assess(
        self,
        keypoints: List[Optional[Keypoint]],
        bbox: BBox,
        frame_shape: Optional[Tuple[int, int]] = None
    ) -> QualityResult:
        """
        Assess the quality of a pose detection.
        
        This is the main entry point for quality assessment.
        
        Args:
            keypoints: List of 17 COCO keypoints [(x, y, conf), ...] or None
            bbox: Bounding box of the detection (x1, y1, x2, y2)
            frame_shape: Optional frame dimensions (height, width)
        
        Returns:
            QualityResult with overall score and component details
        """
        reasons = []
        score = 1.0
        
        # 1. Count valid keypoints
        n_valid = self._count_valid_keypoints(keypoints)
        
        if n_valid < self.config['min_keypoints']:
            penalty = 1.0 - (n_valid / self.config['min_keypoints'])
            score *= (1.0 - penalty * 0.5)  # Up to 50% penalty
            reasons.append(f"LOW_KEYPOINTS:{n_valid}")
        
        # 2. Check torso visibility (CRITICAL)
        has_torso = self._check_torso(keypoints)
        
        if not has_torso and self.config['torso_required']:
            # Severe penalty - multiply by torso_missing_penalty (default 0.2)
            score *= self.config['torso_missing_penalty']
            reasons.append("NO_TORSO")
        
        # 3. Calculate jitter (if we have previous frame data)
        jitter = self._calculate_jitter(keypoints, bbox)
        
        if jitter > self.config['jitter_penalty_threshold']:
            excess_jitter = jitter - self.config['jitter_penalty_threshold']
            jitter_penalty = min(0.5, excess_jitter * self.config['jitter_penalty_factor'])
            score *= (1.0 - jitter_penalty)
            reasons.append(f"HIGH_JITTER:{jitter:.2f}")
        
        # 4. Check average keypoint confidence
        avg_confidence = self._average_confidence(keypoints)
        if avg_confidence < self.config['min_confidence']:
            conf_penalty = 1.0 - (avg_confidence / self.config['min_confidence'])
            score *= (1.0 - conf_penalty * 0.3)  # Up to 30% penalty
            reasons.append(f"LOW_CONFIDENCE:{avg_confidence:.2f}")
        
        # Store for next frame's jitter calculation
        self.prev_keypoints = keypoints
        self.prev_bbox = bbox
        
        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, score))
        
        if not reasons:
            reasons.append("GOOD_QUALITY")
        
        return QualityResult(
            score=score,
            has_torso=has_torso,
            n_valid_keypoints=n_valid,
            jitter=jitter,
            reasons=reasons
        )
    
    def _count_valid_keypoints(self, keypoints: List[Optional[Keypoint]]) -> int:
        """Count keypoints above confidence threshold."""
        return sum(
            1 for kp in keypoints 
            if point_is_valid(kp, self.config['min_confidence'])
        )
    
    def _check_torso(self, keypoints: List[Optional[Keypoint]]) -> bool:
        """
        Check if torso keypoints are visible.
        
        Torso requires BOTH:
        - At least one shoulder visible
        - At least one hip visible
        
        This is the minimum for reliable posture assessment.
        """
        min_conf = self.config['min_confidence']
        
        left_shoulder = keypoints[KEYPOINT_INDICES['left_shoulder']]
        right_shoulder = keypoints[KEYPOINT_INDICES['right_shoulder']]
        left_hip = keypoints[KEYPOINT_INDICES['left_hip']]
        right_hip = keypoints[KEYPOINT_INDICES['right_hip']]
        
        has_shoulder = (
            point_is_valid(left_shoulder, min_conf) or 
            point_is_valid(right_shoulder, min_conf)
        )
        has_hip = (
            point_is_valid(left_hip, min_conf) or 
            point_is_valid(right_hip, min_conf)
        )
        
        return has_shoulder and has_hip
    
    def _calculate_jitter(
        self, 
        keypoints: List[Optional[Keypoint]], 
        bbox: BBox
    ) -> float:
        """
        Calculate normalized jitter from previous frame.
        
        Jitter is computed as average movement of visible keypoints,
        normalized by bounding box size to be scale-invariant.
        """
        if self.prev_keypoints is None or self.prev_bbox is None:
            return 0.0
        
        bbox_size = bbox_diagonal(bbox)
        if bbox_size < 1e-6:
            return 0.0
        
        min_conf = self.config['min_confidence']
        jitter_values = []
        
        for i, (curr_kp, prev_kp) in enumerate(zip(keypoints, self.prev_keypoints)):
            if point_is_valid(curr_kp, min_conf) and point_is_valid(prev_kp, min_conf):
                jitter = compute_jitter(
                    (curr_kp[0], curr_kp[1]),
                    (prev_kp[0], prev_kp[1]),
                    bbox_size
                )
                jitter_values.append(jitter)
        
        if not jitter_values:
            return 0.0
        
        return sum(jitter_values) / len(jitter_values)
    
    def _average_confidence(self, keypoints: List[Optional[Keypoint]]) -> float:
        """Calculate average confidence of valid keypoints."""
        confidences = [
            kp[2] for kp in keypoints 
            if kp is not None and kp[2] > 0
        ]
        
        if not confidences:
            return 0.0
        
        return sum(confidences) / len(confidences)
    
    def reset(self):
        """Reset temporal state (for new person or scene)."""
        self.prev_keypoints = None
        self.prev_bbox = None
    
    def can_confirm_needs_help(
        self, 
        quality: QualityResult,
        has_extreme_geometry: bool = False
    ) -> Tuple[bool, str]:
        """
        Determine if quality allows confirming NEEDS_HELP.
        
        This implements the key safety rule:
        - Quality >= 0.4: Can confirm normally
        - Quality 0.25-0.4 + extreme geometry: Can confirm (override)
        - Quality < 0.25: Cannot confirm (too unreliable)
        
        Args:
            quality: QualityResult from assess()
            has_extreme_geometry: True if pose geometry is extremely clear
        
        Returns:
            (can_confirm, reason_string)
        """
        min_quality = self.config['min_quality_for_confirmation']
        extreme_override = self.config['extreme_geometry_override']
        
        if quality.score >= min_quality:
            return (True, "QUALITY_SUFFICIENT")
        
        if has_extreme_geometry and quality.score >= extreme_override:
            return (True, "EXTREME_GEOMETRY_OVERRIDE")
        
        if quality.score < extreme_override:
            return (False, "QUALITY_TOO_LOW")
        
        return (False, "QUALITY_INSUFFICIENT")
