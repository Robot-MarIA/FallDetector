"""
Pose classification and risk scoring.

This module classifies poses into risk states based on geometric features.
The classifier uses explicit rules with configurable thresholds, making
decisions transparent and auditable.

Internal pose categories:
- LYING: Person in horizontal position
- SITTING_FLOOR: Sitting on ground (not on elevated surface)
- ALL_FOURS: Hands and knees position
- KNEELING: On knees, torso somewhat upright
- NORMAL: Standing, walking, sitting on chair

Output states:
- OK: No risk detected
- RISK: Warning state (kneeling, unusual signals)
- NEEDS_HELP: High-risk posture confirmed (lying, floor sitting, all fours)
- UNKNOWN: Insufficient quality or conflicting signals

Key Design Decisions:
- Risk score is continuous [0, 1] for nuanced decisions
- NEEDS_HELP requires quality check (except extreme geometry)
- Reason strings explain every classification decision
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import yaml
from pathlib import Path

# Import local modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.features import PoseFeatures
from core.quality import QualityResult


class InternalPose(Enum):
    """
    Internal pose categories with physical meaning.
    
    These are more granular than output states for better
    decision-making and future extensibility.
    """
    LYING = "lying"
    SITTING_FLOOR = "sitting_floor"
    ALL_FOURS = "all_fours"
    KNEELING = "kneeling"
    NORMAL = "normal"
    UNKNOWN = "unknown"


class RiskState(Enum):
    """
    Output risk states for the system.
    
    These are the states exposed to external systems (ROS2, UI, logs).
    """
    OK = "OK"
    RISK = "RISK"
    NEEDS_HELP = "NEEDS_HELP"
    UNKNOWN = "UNKNOWN"


@dataclass
class ClassificationResult:
    """
    Result of pose classification.
    
    Attributes:
        state: Output risk state (OK, RISK, NEEDS_HELP, UNKNOWN)
        internal_pose: Detailed internal pose category
        risk_score: Continuous risk score [0, 1]
        confidence: How confident we are in this classification [0, 1]
        reason: Human-readable explanation of the decision
        can_confirm: Whether quality allows final confirmation
    """
    state: RiskState
    internal_pose: InternalPose
    risk_score: float
    confidence: float
    reason: str
    can_confirm: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'state': self.state.value,
            'internal_pose': self.internal_pose.value,
            'risk_score': self.risk_score,
            'confidence': self.confidence,
            'reason': self.reason,
            'can_confirm': self.can_confirm,
        }


class PoseClassifier:
    """
    Classifies poses into risk states using rule-based logic.
    
    The classifier:
    1. Determines internal pose category from features
    2. Computes a continuous risk score
    3. Maps to output state considering quality constraints
    4. Generates explainable reason strings
    
    All thresholds are configurable via YAML.
    """
    
    # Mapping from internal poses to base risk weights
    POSE_RISK_WEIGHTS = {
        InternalPose.LYING: 0.9,
        InternalPose.SITTING_FLOOR: 0.8,
        InternalPose.ALL_FOURS: 0.85,
        InternalPose.KNEELING: 0.5,
        InternalPose.NORMAL: 0.1,
        InternalPose.UNKNOWN: 0.3,  # Uncertain = moderate base risk
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pose classifier.
        
        Args:
            config_path: Path to thresholds.yaml (optional)
        """
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML or use defaults."""
        defaults = {
            # Pose detection thresholds
            'torso_angle_lying': 25.0,
            'torso_angle_sitting': 45.0,
            'torso_angle_standing': 70.0,
            'height_ratio_floor': 0.35,
            'height_ratio_sitting': 0.55,
            'aspect_ratio_lying': 1.8,
            'compactness_fetal': 0.4,
            'bbox_bottom_threshold': 0.85,
            'bbox_bottom_persistence': 5,
            
            # Risk thresholds
            'risk_needs_help': 0.7,
            'risk_warning': 0.4,
            'risk_ok': 0.2,
            
            # Quality gates
            'min_quality_for_confirmation': 0.4,
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    full_config = yaml.safe_load(f)
                    if 'pose' in full_config:
                        defaults.update(full_config['pose'])
                    if 'risk' in full_config:
                        defaults.update(full_config['risk'])
                    if 'quality' in full_config:
                        defaults.update(full_config['quality'])
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        return defaults
    
    def classify(
        self,
        features: PoseFeatures,
        quality: QualityResult,
        has_extreme_geometry: bool = False
    ) -> ClassificationResult:
        """
        Classify a pose based on extracted features.
        
        This is the main entry point for classification.
        
        Args:
            features: Extracted pose features from FeatureExtractor
            quality: Quality assessment from QualityAssessor
            has_extreme_geometry: Whether geometry is extremely clear
        
        Returns:
            ClassificationResult with state, score, and reason
        """
        reason_parts = list(features.reason_parts)  # Copy feature reasons
        
        # 1. Check if we have enough information
        if not quality.has_torso and quality.score < 0.3:
            return ClassificationResult(
                state=RiskState.UNKNOWN,
                internal_pose=InternalPose.UNKNOWN,
                risk_score=0.3,
                confidence=0.2,
                reason="INSUFFICIENT_VISIBILITY: " + " + ".join(quality.reasons),
                can_confirm=False,
            )
        
        # 2. Determine internal pose category
        internal_pose, pose_confidence = self._determine_internal_pose(features)
        reason_parts.append(f"POSE:{internal_pose.value}")
        
        # 3. Calculate risk score
        risk_score = self._calculate_risk_score(
            internal_pose, 
            features, 
            quality,
            pose_confidence
        )
        
        # 4. Check quality constraints for confirmation
        can_confirm, quality_reason = self._check_can_confirm(
            quality, 
            has_extreme_geometry
        )
        
        if not can_confirm:
            reason_parts.append(quality_reason)
        
        # 5. Map to output state
        state = self._map_to_state(risk_score, can_confirm)
        
        # 6. Build final reason string
        reason = " + ".join(reason_parts)
        
        return ClassificationResult(
            state=state,
            internal_pose=internal_pose,
            risk_score=risk_score,
            confidence=pose_confidence * quality.score,
            reason=reason,
            can_confirm=can_confirm,
        )
    
    def _determine_internal_pose(
        self, 
        features: PoseFeatures
    ) -> Tuple[InternalPose, float]:
        """
        Determine the internal pose category from features.
        
        RULE ORDER (CRITICAL):
        1. HARD BLOCK: Vertical torso (> standing threshold) = NOT floor posture
        2. Check for LYING (horizontal orientation + wide bbox)
        3. Check for ALL_FOURS (compact + low hip ratio)
        4. Check for SITTING_FLOOR (low hip ratio + non-horizontal torso)
        5. Check for KNEELING
        6. Default to NORMAL or UNKNOWN
        
        IMPORTANT: bbox_bottom_ratio is NOT used for floor detection
        because letterboxing/resize makes it unreliable.
        Floor postures must be based on body geometry only.
        
        Returns:
            (internal_pose, confidence)
        """
        confidence = 0.5
        
        # =========================================================
        # HARD BLOCK: VERTICAL TORSO = NOT A FLOOR POSTURE
        # If torso is clearly vertical, block LYING/SITTING_FLOOR/ALL_FOURS
        # =========================================================
        torso_is_vertical = False
        if features.torso_angle is not None:
            if features.torso_angle > self.config['torso_angle_standing']:
                torso_is_vertical = True
                # Torso clearly vertical -> NORMAL
                confidence = 0.85
                return (InternalPose.NORMAL, confidence)
        
        # =========================================================
        # LYING: Horizontal torso + horizontal bbox
        # =========================================================
        if features.torso_angle is not None:
            if features.torso_angle < self.config['torso_angle_lying']:
                # Torso nearly horizontal
                if features.bbox_aspect_ratio > self.config['aspect_ratio_lying']:
                    # Wide bbox confirms horizontal body
                    confidence = 0.9
                    return (InternalPose.LYING, confidence)
                elif features.bbox_aspect_ratio > 1.3:
                    # Somewhat wide, probably lying
                    confidence = 0.75
                    return (InternalPose.LYING, confidence)
        
        # =========================================================
        # ALL_FOURS: Compact body + hips very low + non-vertical torso
        # Based on body geometry, NOT bbox position in frame
        # =========================================================
        if features.compactness is not None and features.hip_height_ratio is not None:
            is_compact = features.compactness < self.config['compactness_fetal']
            hips_very_low = features.hip_height_ratio < self.config['height_ratio_floor']
            torso_not_upright = (
                features.torso_angle is not None and 
                features.torso_angle < self.config['torso_angle_sitting']
            )
            
            if is_compact and hips_very_low and torso_not_upright:
                confidence = 0.75
                return (InternalPose.ALL_FOURS, confidence)
        
        # =========================================================
        # SITTING_FLOOR: Low hip ratio + semi-upright torso
        # CRITICAL: Requires BOTH low hips AND non-standing torso angle
        # If torso is upright, person is NOT on floor (standing/chair)
        # =========================================================
        if features.hip_height_ratio is not None and features.torso_angle is not None:
            hips_low = features.hip_height_ratio < self.config['height_ratio_sitting']
            torso_semi_upright = (
                features.torso_angle > self.config['torso_angle_lying'] and
                features.torso_angle < self.config['torso_angle_standing']
            )
            
            if hips_low and torso_semi_upright:
                # Additional check: shoulder height should also be relatively low
                if features.shoulder_height_ratio is not None:
                    if features.shoulder_height_ratio < 0.65:
                        confidence = 0.7
                        return (InternalPose.SITTING_FLOOR, confidence)
                else:
                    # No shoulder data, rely on hip ratio + torso angle
                    confidence = 0.6
                    return (InternalPose.SITTING_FLOOR, confidence)
        
        # =========================================================
        # KNEELING: Medium torso angle, moderate height
        # =========================================================
        if features.torso_angle is not None:
            is_kneeling_angle = (
                self.config['torso_angle_lying'] < features.torso_angle < 
                self.config['torso_angle_standing']
            )
            
            if is_kneeling_angle and features.hip_height_ratio is not None:
                # Hips at medium height (not standing, not full floor)
                if 0.3 < features.hip_height_ratio < 0.6:
                    confidence = 0.6
                    return (InternalPose.KNEELING, confidence)
        
        # =========================================================
        # NORMAL: Fallbacks for standing/sitting on chair
        # =========================================================
        
        # Upright-ish torso (already checked strict vertical above)
        if features.torso_angle is not None:
            if features.torso_angle > self.config['torso_angle_sitting']:
                confidence = 0.75
                return (InternalPose.NORMAL, confidence)
        
        # Tall narrow bbox typically = standing
        if features.bbox_aspect_ratio < 0.7:
            confidence = 0.65
            return (InternalPose.NORMAL, confidence)
        
        # Default to UNKNOWN
        confidence = 0.3
        return (InternalPose.UNKNOWN, confidence)
    
    def _calculate_risk_score(
        self,
        internal_pose: InternalPose,
        features: PoseFeatures,
        quality: QualityResult,
        pose_confidence: float
    ) -> float:
        """
        Calculate continuous risk score [0, 1].
        
        The risk score combines:
        - Base risk for the pose type
        - Feature-based modifiers (body geometry only)
        - Quality-based uncertainty
        
        NOTE: bbox_bottom_persistence removed as modifier because
        it's unreliable with letterboxing/resize.
        
        Returns:
            Risk score [0, 1] where 1 = maximum risk
        """
        # Start with base risk for pose type
        base_risk = self.POSE_RISK_WEIGHTS.get(internal_pose, 0.3)
        
        # Feature-based modifiers (body geometry only)
        modifier = 0.0
        
        # Horizontal torso increases risk
        if features.torso_angle is not None:
            if features.torso_angle < 20.0:
                modifier += 0.1  # Very horizontal
            elif features.torso_angle < 30.0:
                modifier += 0.05
        
        # Very low hip position increases risk
        if features.hip_height_ratio is not None:
            if features.hip_height_ratio < 0.25:
                modifier += 0.1  # Very low hips
            elif features.hip_height_ratio < 0.35:
                modifier += 0.05
        
        # Low quality adds uncertainty - bias toward caution
        if quality.score < 0.5:
            modifier += 0.05
        
        # Compute final score
        risk_score = base_risk + modifier
        
        # Weight by pose confidence
        # Low confidence pulls toward neutral (0.3)
        risk_score = risk_score * pose_confidence + 0.3 * (1 - pose_confidence)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, risk_score))
    
    def _check_can_confirm(
        self,
        quality: QualityResult,
        has_extreme_geometry: bool
    ) -> Tuple[bool, str]:
        """
        Check if quality allows confirming NEEDS_HELP.
        
        Returns:
            (can_confirm, reason_string)
        """
        min_quality = self.config['min_quality_for_confirmation']
        
        if quality.score >= min_quality:
            return (True, "QUALITY_OK")
        
        if has_extreme_geometry and quality.score >= 0.25:
            return (True, "EXTREME_GEO_OVERRIDE")
        
        return (False, "QUALITY_BLOCKS_CONFIRMATION")
    
    def _map_to_state(
        self, 
        risk_score: float, 
        can_confirm: bool
    ) -> RiskState:
        """
        Map risk score to output state.
        
        Uses configurable thresholds:
        - risk > risk_needs_help → NEEDS_HELP (if can_confirm)
        - risk > risk_warning → RISK
        - risk < risk_ok → OK
        - otherwise → UNKNOWN
        """
        if risk_score > self.config['risk_needs_help']:
            if can_confirm:
                return RiskState.NEEDS_HELP
            else:
                return RiskState.RISK  # Downgrade if can't confirm
        
        if risk_score > self.config['risk_warning']:
            return RiskState.RISK
        
        if risk_score < self.config['risk_ok']:
            return RiskState.OK
        
        return RiskState.UNKNOWN


def get_state_display(state: RiskState) -> Dict[str, str]:
    """
    Get display properties for a risk state.
    
    Returns:
        Dictionary with 'label', 'color', 'icon' for UI display
    """
    display_map = {
        RiskState.OK: {
            'label': 'OK',
            'color': '#22c55e',  # Green
            'icon': '✓',
        },
        RiskState.RISK: {
            'label': 'RIESGO',
            'color': '#f59e0b',  # Orange
            'icon': '⚠',
        },
        RiskState.NEEDS_HELP: {
            'label': 'AYUDA',
            'color': '#ef4444',  # Red
            'icon': '🚨',
        },
        RiskState.UNKNOWN: {
            'label': 'DESCONOCIDO',
            'color': '#6b7280',  # Gray
            'icon': '?',
        },
    }
    return display_map.get(state, display_map[RiskState.UNKNOWN])
