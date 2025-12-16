"""
Feature extraction for pose analysis.

This module extracts explainable geometric features from pose keypoints.
These features are designed to be:
1. Interpretable - each feature has clear physical meaning
2. Robust to partial visibility - graceful degradation when keypoints missing
3. Scale-invariant - normalized by bounding box when appropriate

Features extracted:
- Torso angle (orientation of spine relative to horizontal)
- Relative heights (hip/shoulder position within bbox)
- Aspect ratio (bbox width/height)
- Compactness (skeleton spread relative to bbox)
- Floor proximity (bbox bottom position in frame)
- Persistence tracking (frames with bbox at bottom)

These features form the input to the rule-based classifier.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from collections import deque
import yaml
from pathlib import Path

# Import geometry utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.geometry import (
    angle_from_horizontal, midpoint_if_valid, bbox_aspect_ratio,
    bbox_diagonal, bbox_dimensions, compute_skeleton_spread,
    point_is_valid, relative_position_in_frame
)

# Type aliases
Keypoint = Tuple[float, float, float]  # (x, y, confidence)
BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)
Point2D = Tuple[float, float]


@dataclass
class PoseFeatures:
    """
    Extracted features from a pose detection.
    
    All features are designed to be physically interpretable.
    Optional fields are None when the required keypoints are not visible.
    
    Attributes:
        torso_angle: Angle of torso from horizontal [0°=lying, 90°=standing]
        hip_height_ratio: Relative vertical position of hips in bbox [0=top, 1=bottom]
        shoulder_height_ratio: Relative vertical position of shoulders in bbox
        bbox_aspect_ratio: Width/height ratio of bounding box
        compactness: How compact the skeleton is [0=very compact, 1=spread out]
        bbox_bottom_ratio: Vertical position of bbox bottom in frame [0=top, 1=bottom]
        bbox_bottom_persistence: Frames with bbox at bottom of frame
        n_valid_keypoints: Number of keypoints above confidence threshold
        
        reason_parts: List of descriptive strings for logging/debugging
    """
    # Core pose features
    torso_angle: Optional[float] = None
    hip_height_ratio: Optional[float] = None
    shoulder_height_ratio: Optional[float] = None
    
    # Bounding box features
    bbox_aspect_ratio: float = 0.0
    compactness: Optional[float] = None
    
    # Floor detection proxy features
    bbox_bottom_ratio: float = 0.0
    bbox_bottom_persistence: int = 0
    
    # Quality indicators
    n_valid_keypoints: int = 0
    
    # Explainability
    reason_parts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'torso_angle': self.torso_angle,
            'hip_height_ratio': self.hip_height_ratio,
            'shoulder_height_ratio': self.shoulder_height_ratio,
            'bbox_aspect_ratio': self.bbox_aspect_ratio,
            'compactness': self.compactness,
            'bbox_bottom_ratio': self.bbox_bottom_ratio,
            'bbox_bottom_persistence': self.bbox_bottom_persistence,
            'n_valid_keypoints': self.n_valid_keypoints,
        }
    
    def get_reason_string(self) -> str:
        """Get concatenated reason string for logging."""
        return " + ".join(self.reason_parts) if self.reason_parts else "NO_FEATURES"


# COCO keypoint indices
KEYPOINT_INDICES = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16,
}


class FeatureExtractor:
    """
    Extracts geometric features from pose keypoints for classification.
    
    This class handles:
    1. Extracting individual pose features
    2. Tracking temporal features (bbox persistence at bottom)
    3. Generating reason strings for explainability
    
    Design principles:
    - Features are computed only when required keypoints are visible
    - Missing features return None (not estimated/guessed)
    - Temporal features use short sliding windows
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the feature extractor.
        
        Args:
            config_path: Path to thresholds.yaml (optional)
        """
        self.config = self._load_config(config_path)
        
        # Temporal tracking for floor detection proxy
        self.bbox_bottom_history: deque = deque(
            maxlen=self.config['bbox_bottom_persistence']
        )
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML or use defaults."""
        defaults = {
            'min_confidence': 0.5,
            'torso_angle_lying': 25.0,
            'torso_angle_sitting': 45.0,
            'torso_angle_standing': 70.0,
            'height_ratio_floor': 0.35,
            'aspect_ratio_lying': 1.8,
            'compactness_fetal': 0.4,
            'bbox_bottom_threshold': 0.85,
            'bbox_bottom_persistence': 5,
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    full_config = yaml.safe_load(f)
                    if 'pose' in full_config:
                        defaults.update(full_config['pose'])
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        return defaults
    
    def extract(
        self,
        keypoints: List[Optional[Keypoint]],
        bbox: BBox,
        frame_shape: Tuple[int, int]
    ) -> PoseFeatures:
        """
        Extract all features from a pose detection.
        
        This is the main entry point for feature extraction.
        
        Args:
            keypoints: List of 17 COCO keypoints [(x, y, conf), ...] or None
            bbox: Bounding box (x1, y1, x2, y2)
            frame_shape: Frame dimensions (height, width)
        
        Returns:
            PoseFeatures with all extracted features and reason strings
        """
        features = PoseFeatures()
        min_conf = self.config['min_confidence']
        
        # 1. Count valid keypoints
        features.n_valid_keypoints = sum(
            1 for kp in keypoints 
            if point_is_valid(kp, min_conf)
        )
        
        # 2. Torso angle (most important feature for lying detection)
        features.torso_angle = self._compute_torso_angle(keypoints)
        if features.torso_angle is not None:
            if features.torso_angle < self.config['torso_angle_lying']:
                features.reason_parts.append("TORSO_HORIZONTAL")
            elif features.torso_angle > self.config['torso_angle_standing']:
                features.reason_parts.append("TORSO_VERTICAL")
        
        # 3. Hip and shoulder height ratios (within bbox)
        features.hip_height_ratio = self._compute_height_ratio(
            keypoints, bbox, ['left_hip', 'right_hip']
        )
        features.shoulder_height_ratio = self._compute_height_ratio(
            keypoints, bbox, ['left_shoulder', 'right_shoulder']
        )
        
        if features.hip_height_ratio is not None:
            if features.hip_height_ratio < self.config['height_ratio_floor']:
                features.reason_parts.append("LOW_HEIGHT")
        
        # 4. Bounding box aspect ratio
        features.bbox_aspect_ratio = bbox_aspect_ratio(bbox)
        if features.bbox_aspect_ratio > self.config['aspect_ratio_lying']:
            features.reason_parts.append("WIDE_BBOX")
        
        # 5. Skeleton compactness
        features.compactness = self._compute_compactness(keypoints, bbox)
        if features.compactness is not None:
            if features.compactness < self.config['compactness_fetal']:
                features.reason_parts.append("COMPACT_POSE")
        
        # 6. Floor detection proxy - bbox bottom position
        features.bbox_bottom_ratio = self._compute_bbox_bottom_ratio(bbox, frame_shape)
        
        # 7. Track persistence of bbox at frame bottom
        is_at_bottom = features.bbox_bottom_ratio > self.config['bbox_bottom_threshold']
        self.bbox_bottom_history.append(is_at_bottom)
        features.bbox_bottom_persistence = sum(self.bbox_bottom_history)
        
        if features.bbox_bottom_persistence >= self.config['bbox_bottom_persistence'] * 0.8:
            features.reason_parts.append("PERSISTENT_AT_BOTTOM")
        
        return features
    
    def _compute_torso_angle(
        self, 
        keypoints: List[Optional[Keypoint]]
    ) -> Optional[float]:
        """
        Compute the angle of the torso from horizontal.
        
        Torso is defined by the line from mid-shoulders to mid-hips.
        Returns None if either midpoint cannot be computed.
        
        Returns:
            Angle in degrees [0=horizontal/lying, 90=vertical/standing]
        """
        min_conf = self.config['min_confidence']
        
        # Get shoulder and hip keypoints
        left_shoulder = keypoints[KEYPOINT_INDICES['left_shoulder']]
        right_shoulder = keypoints[KEYPOINT_INDICES['right_shoulder']]
        left_hip = keypoints[KEYPOINT_INDICES['left_hip']]
        right_hip = keypoints[KEYPOINT_INDICES['right_hip']]
        
        # Compute midpoints (returns None if keypoints invalid)
        mid_shoulders = midpoint_if_valid(left_shoulder, right_shoulder, min_conf)
        mid_hips = midpoint_if_valid(left_hip, right_hip, min_conf)
        
        # Fallback: use single keypoint if only one side visible
        if mid_shoulders is None:
            if point_is_valid(left_shoulder, min_conf):
                mid_shoulders = (left_shoulder[0], left_shoulder[1])
            elif point_is_valid(right_shoulder, min_conf):
                mid_shoulders = (right_shoulder[0], right_shoulder[1])
        
        if mid_hips is None:
            if point_is_valid(left_hip, min_conf):
                mid_hips = (left_hip[0], left_hip[1])
            elif point_is_valid(right_hip, min_conf):
                mid_hips = (right_hip[0], right_hip[1])
        
        # Cannot compute angle without both points
        if mid_shoulders is None or mid_hips is None:
            return None
        
        return angle_from_horizontal(mid_shoulders, mid_hips)
    
    def _compute_height_ratio(
        self,
        keypoints: List[Optional[Keypoint]],
        bbox: BBox,
        keypoint_names: List[str]
    ) -> Optional[float]:
        """
        Compute the relative vertical position of keypoints within the bbox.
        
        Returns a ratio [0, 1] where:
        - 0 = keypoints at top of bbox
        - 1 = keypoints at bottom of bbox
        
        This helps distinguish standing (hips high in bbox) from 
        sitting/lying (hips low or spread across bbox).
        """
        min_conf = self.config['min_confidence']
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        
        if bbox_height < 1:
            return None
        
        # Collect valid y-coordinates
        y_values = []
        for name in keypoint_names:
            kp = keypoints[KEYPOINT_INDICES[name]]
            if point_is_valid(kp, min_conf):
                y_values.append(kp[1])
        
        if not y_values:
            return None
        
        avg_y = sum(y_values) / len(y_values)
        
        # Normalize to [0, 1] within bbox
        ratio = (avg_y - y1) / bbox_height
        return max(0.0, min(1.0, ratio))
    
    def _compute_compactness(
        self,
        keypoints: List[Optional[Keypoint]],
        bbox: BBox
    ) -> Optional[float]:
        """
        Compute skeleton compactness (spread relative to bbox).
        
        A very compact skeleton (low ratio) might indicate:
        - Fetal position
        - Curled up on floor
        - Crouching
        
        Returns:
            Compactness ratio [0, 1] where 0 = very compact, 1 = spread out
        """
        min_conf = self.config['min_confidence']
        
        skeleton_spread = compute_skeleton_spread(keypoints, min_conf)
        if skeleton_spread is None:
            return None
        
        bbox_diag = bbox_diagonal(bbox)
        if bbox_diag < 1:
            return None
        
        # Normalize: skeleton spread / bbox diagonal
        # Cap at 1.0 (skeleton shouldn't be larger than bbox)
        ratio = skeleton_spread / bbox_diag
        return min(1.0, ratio)
    
    def _compute_bbox_bottom_ratio(
        self,
        bbox: BBox,
        frame_shape: Tuple[int, int]
    ) -> float:
        """
        Compute the relative position of bbox bottom edge in the frame.
        
        This is a proxy for floor contact without depth sensing:
        - Ratio close to 1.0 = bbox bottom near frame bottom = likely on floor
        - Ratio < 0.7 = bbox not at bottom = likely standing/sitting on chair
        
        Returns:
            Ratio [0, 1] where 1 = bbox at very bottom of frame
        """
        frame_height, frame_width = frame_shape
        x1, y1, x2, y2 = bbox
        
        if frame_height < 1:
            return 0.0
        
        return y2 / frame_height
    
    def reset(self):
        """Reset temporal state (for new person or scene)."""
        self.bbox_bottom_history.clear()
    
    def is_extreme_geometry(self, features: PoseFeatures) -> bool:
        """
        Check if the pose geometry is extremely clear (allows quality override).
        
        "Extreme geometry" means multiple strong indicators of lying/floor position.
        This allows NEEDS_HELP confirmation even with slightly lower quality.
        
        Args:
            features: Extracted pose features
        
        Returns:
            True if geometry is unambiguous
        """
        indicators = 0
        
        # Torso nearly horizontal
        if features.torso_angle is not None and features.torso_angle < 15.0:
            indicators += 1
        
        # Very wide bbox (horizontal orientation)
        if features.bbox_aspect_ratio > 2.0:
            indicators += 1
        
        # Hips very low in bbox
        if features.hip_height_ratio is not None and features.hip_height_ratio < 0.25:
            indicators += 1
        
        # Persistent at bottom of frame
        if features.bbox_bottom_persistence >= 4:
            indicators += 1
        
        # Extreme = 3+ indicators
        return indicators >= 3
