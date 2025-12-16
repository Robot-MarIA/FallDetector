"""
Temporal analysis and adaptive confirmation.

This module handles temporal consistency of pose classifications.
Unlike fixed-window confirmation, this uses ADAPTIVE confirmation times
based on:
- Stability of the detected pose (low jitter = faster confirmation)
- Quality of detection (high quality = faster confirmation)
- Clarity of risk signals (clear danger = faster confirmation)

Key Design Decision:
- Confirmation window is NOT fixed at 2.5 seconds
- Window dynamically adjusts between min_window and max_window
- Clear, stable, high-quality detections confirm faster
- Ambiguous, unstable, low-quality detections require longer observation

This prevents both:
- False positives (confirming too fast on uncertain data)
- Delayed response (waiting too long on obvious emergencies)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Deque
from collections import deque
from enum import Enum
import time
import yaml
from pathlib import Path

# Import local modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.classifier import RiskState, ClassificationResult


@dataclass
class TemporalState:
    """
    Temporal state tracking for a single person.
    
    Attributes:
        current_state: Most recent classification state
        confirmed_state: State after temporal confirmation
        confirmation_progress: Progress toward confirmation [0, 1]
        window_seconds: Current adaptive confirmation window
        stability: Pose stability measure [0, 1]
        state_history: Recent state history for analysis
        last_update_time: Timestamp of last update
    """
    current_state: RiskState = RiskState.UNKNOWN
    confirmed_state: RiskState = RiskState.OK
    confirmation_progress: float = 0.0
    window_seconds: float = 2.5
    stability: float = 0.5
    state_history: Deque = field(default_factory=lambda: deque(maxlen=30))
    last_update_time: float = field(default_factory=time.time)
    
    # Tracking for stability calculation
    position_history: Deque = field(default_factory=lambda: deque(maxlen=10))


class TemporalAnalyzer:
    """
    Analyzes temporal consistency and manages adaptive confirmation.
    
    The confirmation window adapts based on three factors:
    1. Stability factor: How stable is the pose over time?
    2. Quality factor: How reliable are the detections?
    3. Risk clarity factor: How clear are the risk signals?
    
    Window calculation:
        window = base_window * (1 / combined_factor)
        combined_factor = stability^w1 * quality^w2 * risk_clarity^w3
        window = clamp(window, min_window, max_window)
    
    This means:
    - High stability + high quality + clear risk → SHORT window (fast confirmation)
    - Low stability + low quality + ambiguous → LONG window (slow confirmation)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the temporal analyzer.
        
        Args:
            config_path: Path to thresholds.yaml (optional)
        """
        self.config = self._load_config(config_path)
        self.state = TemporalState()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML or use defaults."""
        defaults = {
            'base_window_seconds': 2.5,
            'min_window_seconds': 1.0,
            'max_window_seconds': 5.0,
            'stability_weight': 0.4,
            'quality_weight': 0.3,
            'risk_clarity_weight': 0.3,
            'stability_threshold': 0.8,
            'pose_change_threshold': 0.15,
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    full_config = yaml.safe_load(f)
                    if 'temporal' in full_config:
                        defaults.update(full_config['temporal'])
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        return defaults
    
    def update(
        self,
        classification: ClassificationResult,
        quality_score: float,
        bbox_center: Optional[tuple] = None,
        timestamp: Optional[float] = None
    ) -> 'TemporalResult':
        """
        Update temporal state with new classification.
        
        This is the main entry point for temporal analysis.
        
        Args:
            classification: Result from PoseClassifier
            quality_score: Quality score [0, 1]
            bbox_center: Optional (x, y) for stability tracking
            timestamp: Optional timestamp (uses current time if None)
        
        Returns:
            TemporalResult with confirmed state and analysis details
        """
        now = timestamp or time.time()
        dt = now - self.state.last_update_time
        self.state.last_update_time = now
        
        # 1. Update stability tracking
        if bbox_center is not None:
            self.state.position_history.append(bbox_center)
        self.state.stability = self._calculate_stability()
        
        # 2. Add to state history
        self.state.state_history.append({
            'state': classification.state,
            'risk_score': classification.risk_score,
            'quality': quality_score,
            'timestamp': now,
        })
        
        # 3. Check for state change
        state_changed = classification.state != self.state.current_state
        self.state.current_state = classification.state
        
        # 4. Calculate adaptive confirmation window
        risk_clarity = self._calculate_risk_clarity(classification)
        self.state.window_seconds = self._calculate_adaptive_window(
            self.state.stability,
            quality_score,
            risk_clarity
        )
        
        # 5. Update confirmation progress
        if state_changed:
            # Reset progress on state change
            self.state.confirmation_progress = 0.0
        else:
            # Accumulate progress toward confirmation
            progress_rate = 1.0 / max(self.state.window_seconds, 0.1)
            self.state.confirmation_progress += dt * progress_rate
            self.state.confirmation_progress = min(1.0, self.state.confirmation_progress)
        
        # 6. Check if confirmed
        is_confirmed = self.state.confirmation_progress >= 1.0
        
        if is_confirmed:
            # Special handling for NEEDS_HELP: require can_confirm
            if classification.state == RiskState.NEEDS_HELP:
                if classification.can_confirm:
                    self.state.confirmed_state = RiskState.NEEDS_HELP
                else:
                    # Can't confirm NEEDS_HELP due to quality, downgrade to RISK
                    self.state.confirmed_state = RiskState.RISK
            else:
                self.state.confirmed_state = classification.state
        
        # 7. Build reason string
        reason_parts = []
        if is_confirmed:
            reason_parts.append("CONFIRMED")
        else:
            reason_parts.append(f"CONFIRMING:{self.state.confirmation_progress:.0%}")
        
        reason_parts.append(f"WINDOW:{self.state.window_seconds:.1f}s")
        reason_parts.append(f"STABILITY:{self.state.stability:.2f}")
        
        return TemporalResult(
            current_state=self.state.current_state,
            confirmed_state=self.state.confirmed_state,
            is_confirmed=is_confirmed,
            confirmation_progress=self.state.confirmation_progress,
            window_seconds=self.state.window_seconds,
            stability=self.state.stability,
            reason=" | ".join(reason_parts),
        )
    
    def _calculate_stability(self) -> float:
        """
        Calculate pose stability from position history.
        
        Stability is inversely related to position variance.
        High stability = pose is not moving much.
        
        Returns:
            Stability score [0, 1] where 1 = perfectly stable
        """
        if len(self.state.position_history) < 2:
            return 0.5  # Default for insufficient data
        
        positions = list(self.state.position_history)
        
        # Calculate variance in x and y
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        
        var_x = sum((x - mean_x) ** 2 for x in xs) / len(xs)
        var_y = sum((y - mean_y) ** 2 for y in ys) / len(ys)
        
        total_variance = (var_x + var_y) ** 0.5
        
        # Normalize: assume max reasonable variance of 100 pixels
        # This should be normalized by bbox size in production
        normalized_variance = min(1.0, total_variance / 100.0)
        
        # Invert: low variance = high stability
        stability = 1.0 - normalized_variance
        
        return stability
    
    def _calculate_risk_clarity(self, classification: ClassificationResult) -> float:
        """
        Calculate how clear/unambiguous the risk signal is.
        
        Clear signals:
        - Very high risk score (> 0.8) or very low (< 0.2)
        - High classification confidence
        
        Ambiguous signals:
        - Risk score near 0.5
        - Low confidence
        
        Returns:
            Clarity score [0, 1] where 1 = very clear signal
        """
        # Distance from ambiguous middle (0.5)
        risk_distance = abs(classification.risk_score - 0.5) * 2
        
        # Combine with classification confidence
        clarity = (risk_distance + classification.confidence) / 2
        
        return max(0.1, min(1.0, clarity))
    
    def _calculate_adaptive_window(
        self,
        stability: float,
        quality: float,
        risk_clarity: float
    ) -> float:
        """
        Calculate the adaptive confirmation window.
        
        The window shrinks when:
        - Stability is high (pose not moving)
        - Quality is high (reliable detection)
        - Risk signals are clear (unambiguous)
        
        The window grows when any of these are low.
        
        Returns:
            Window in seconds, clamped to [min, max]
        """
        base = self.config['base_window_seconds']
        min_window = self.config['min_window_seconds']
        max_window = self.config['max_window_seconds']
        
        # Weights for each factor
        w_stability = self.config['stability_weight']
        w_quality = self.config['quality_weight']
        w_clarity = self.config['risk_clarity_weight']
        
        # Ensure non-zero values
        stability = max(0.1, stability)
        quality = max(0.1, quality)
        risk_clarity = max(0.1, risk_clarity)
        
        # Combined factor: geometric weighted mean
        combined = (
            (stability ** w_stability) *
            (quality ** w_quality) *
            (risk_clarity ** w_clarity)
        )
        
        # Window is inversely proportional to combined factor
        # High combined → short window
        # Low combined → long window
        window = base / max(0.1, combined)
        
        # Clamp to configured range
        return max(min_window, min(max_window, window))
    
    def get_state_persistence(self, state: RiskState) -> float:
        """
        Calculate how long a state has persisted in recent history.
        
        Args:
            state: The state to check
        
        Returns:
            Duration in seconds that state has been continuous
        """
        if not self.state.state_history:
            return 0.0
        
        # Walk backwards through history
        duration = 0.0
        prev_time = None
        
        for entry in reversed(self.state.state_history):
            if entry['state'] != state:
                break
            
            if prev_time is not None:
                duration += prev_time - entry['timestamp']
            
            prev_time = entry['timestamp']
        
        return duration
    
    def reset(self):
        """Reset temporal state (for new person or scene)."""
        self.state = TemporalState()


@dataclass
class TemporalResult:
    """
    Result of temporal analysis.
    
    Attributes:
        current_state: Latest classification (may not be confirmed)
        confirmed_state: State after temporal confirmation
        is_confirmed: Whether current state is confirmed
        confirmation_progress: Progress toward confirmation [0, 1]
        window_seconds: Current adaptive window duration
        stability: Pose stability [0, 1]
        reason: Explanation string for logging
    """
    current_state: RiskState
    confirmed_state: RiskState
    is_confirmed: bool
    confirmation_progress: float
    window_seconds: float
    stability: float
    reason: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'current_state': self.current_state.value,
            'confirmed_state': self.confirmed_state.value,
            'is_confirmed': self.is_confirmed,
            'confirmation_progress': self.confirmation_progress,
            'window_seconds': self.window_seconds,
            'stability': self.stability,
        }
