"""
Adaptive scheduler for resource optimization.

This module manages the trade-off between:
- Power consumption (lower FPS, lower resolution)
- Detection reliability (higher FPS for better temporal analysis)

The scheduler operates in three modes:
- LOW_POWER: Minimal resources when situation is stable and safe
- CHECKING: Elevated monitoring when uncertainty or potential risk detected
- CONFIRMING: Maximum attention during critical confirmation phase

Key Design Decision (PRUDENT BEHAVIOR):
- UNKNOWN state with elevated risk → CHECKING mode (never LOW_POWER)
- The system prefers to improve observation before making decisions
- Only returns to LOW_POWER after confirmed OK with high quality

This prevents the system from "going to sleep" when it should be paying attention.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from enum import Enum
import time
import yaml
from pathlib import Path

# Import local modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.classifier import RiskState
from core.temporal import TemporalResult


class SchedulerMode(Enum):
    """
    Scheduler operating modes.
    
    Each mode defines target FPS, resolution, and inference frequency.
    """
    LOW_POWER = "LOW_POWER"
    CHECKING = "CHECKING"
    CONFIRMING = "CONFIRMING"


@dataclass
class SchedulerSettings:
    """
    Current scheduler settings based on mode.
    
    Attributes:
        mode: Current operating mode
        target_fps: Target frames per second
        resolution_scale: Scale factor for input resolution [0, 1]
        inference_skip: Process every Nth frame (1 = all frames)
        reason: Explanation for current mode
    """
    mode: SchedulerMode
    target_fps: float
    resolution_scale: float
    inference_skip: int
    reason: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'mode': self.mode.value,
            'target_fps': self.target_fps,
            'resolution_scale': self.resolution_scale,
            'inference_skip': self.inference_skip,
        }


class AdaptiveScheduler:
    """
    Manages adaptive resource allocation based on system state.
    
    The scheduler implements a state machine:
    
    LOW_POWER ←→ CHECKING ←→ CONFIRMING
        ↑___________________________|
    
    Transitions are governed by:
    - Risk score thresholds
    - Quality stability
    - Confirmed state duration
    
    PRUDENT BEHAVIOR:
    The scheduler is biased toward CHECKING over LOW_POWER when uncertain.
    Key rule: UNKNOWN + risk > threshold → CHECKING (never LOW_POWER)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the adaptive scheduler.
        
        Args:
            config_path: Path to scheduler.yaml (optional)
        """
        self.config = self._load_config(config_path)
        
        self.current_mode = SchedulerMode.CHECKING  # Start conservatively
        self.mode_start_time = time.time()
        self.last_mode_change = time.time()
        
        # Tracking for transition decisions
        self.consecutive_ok_start: Optional[float] = None
        self.quality_history: list = []
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML or use defaults."""
        defaults = {
            'modes': {
                'LOW_POWER': {
                    'fps': 2,
                    'resolution_scale': 0.5,
                    'inference_skip': 2,
                },
                'CHECKING': {
                    'fps': 12,
                    'resolution_scale': 0.75,
                    'inference_skip': 1,
                    'duration_seconds': 3.0,
                },
                'CONFIRMING': {
                    'fps': 15,
                    'resolution_scale': 1.0,
                    'inference_skip': 1,
                },
            },
            'transitions': {
                'to_checking': {
                    'risk_threshold': 0.3,
                    'quality_unstable_threshold': 0.5,
                    'unknown_with_risk': True,
                    'unknown_risk_threshold': 0.4,
                },
                'to_confirming': {
                    'risk_threshold': 0.6,
                    'quality_min': 0.4,
                    'consecutive_risk_frames': 3,
                },
                'to_low_power': {
                    'consecutive_ok_seconds': 5.0,
                    'require_high_quality': True,
                    'quality_min': 0.6,
                    'risk_max': 0.2,
                },
            },
            'timing': {
                'mode_switch_cooldown': 1.0,
            },
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    loaded = yaml.safe_load(f)
                    # Deep merge
                    self._deep_merge(defaults, loaded)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        return defaults
    
    def _deep_merge(self, base: Dict, override: Dict):
        """Deep merge override into base dictionary."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def update(
        self,
        risk_score: float,
        quality_score: float,
        confirmed_state: RiskState,
        current_state: RiskState,
        is_confirmed: bool,
        timestamp: Optional[float] = None
    ) -> SchedulerSettings:
        """
        Update scheduler based on current system state.
        
        This is the main entry point for scheduler decisions.
        
        Args:
            risk_score: Current risk score [0, 1]
            quality_score: Current quality score [0, 1]
            confirmed_state: Temporally confirmed state
            current_state: Instantaneous state (may not be confirmed)
            is_confirmed: Whether current state is temporally confirmed
            timestamp: Optional timestamp (uses current time if None)
        
        Returns:
            SchedulerSettings with current mode and parameters
        """
        now = timestamp or time.time()
        
        # Track quality for stability analysis
        self.quality_history.append(quality_score)
        if len(self.quality_history) > 30:
            self.quality_history.pop(0)
        
        quality_stable = self._is_quality_stable()
        
        # Determine target mode based on state
        target_mode = self._determine_target_mode(
            risk_score=risk_score,
            quality_score=quality_score,
            quality_stable=quality_stable,
            confirmed_state=confirmed_state,
            current_state=current_state,
            is_confirmed=is_confirmed,
            now=now,
        )
        
        # Check cooldown before switching
        cooldown = self.config['timing']['mode_switch_cooldown']
        can_switch = (now - self.last_mode_change) >= cooldown
        
        reason_parts = []
        
        if target_mode != self.current_mode and can_switch:
            old_mode = self.current_mode
            self.current_mode = target_mode
            self.mode_start_time = now
            self.last_mode_change = now
            reason_parts.append(f"SWITCHED:{old_mode.value}→{target_mode.value}")
        else:
            reason_parts.append(f"MODE:{self.current_mode.value}")
        
        # Add context to reason
        if current_state == RiskState.UNKNOWN and risk_score > 0.3:
            reason_parts.append("UNKNOWN_WITH_RISK")
        elif risk_score > 0.6:
            reason_parts.append("HIGH_RISK")
        elif not quality_stable:
            reason_parts.append("QUALITY_UNSTABLE")
        
        # Get settings for current mode
        mode_config = self.config['modes'][self.current_mode.value]
        
        return SchedulerSettings(
            mode=self.current_mode,
            target_fps=mode_config['fps'],
            resolution_scale=mode_config['resolution_scale'],
            inference_skip=mode_config['inference_skip'],
            reason=" | ".join(reason_parts),
        )
    
    def _determine_target_mode(
        self,
        risk_score: float,
        quality_score: float,
        quality_stable: bool,
        confirmed_state: RiskState,
        current_state: RiskState,
        is_confirmed: bool,
        now: float,
    ) -> SchedulerMode:
        """
        Determine the target scheduler mode.
        
        Implements the transition rules from config.
        """
        trans = self.config['transitions']
        
        # --- Check for CONFIRMING mode ---
        # High risk with sufficient quality → CONFIRMING
        to_conf = trans['to_confirming']
        if (risk_score > to_conf['risk_threshold'] and 
            quality_score >= to_conf['quality_min']):
            return SchedulerMode.CONFIRMING
        
        # Already in CONFIRMING and still have elevated risk
        if self.current_mode == SchedulerMode.CONFIRMING:
            if risk_score > trans['to_checking']['risk_threshold']:
                return SchedulerMode.CONFIRMING
        
        # --- Check for CHECKING mode (PRUDENT BEHAVIOR) ---
        to_check = trans['to_checking']
        
        # KEY RULE: UNKNOWN + elevated risk → CHECKING (never LOW_POWER)
        if current_state == RiskState.UNKNOWN:
            if to_check['unknown_with_risk'] and risk_score > to_check['unknown_risk_threshold']:
                self.consecutive_ok_start = None  # Reset OK tracking
                return SchedulerMode.CHECKING
        
        # Medium risk → CHECKING
        if risk_score > to_check['risk_threshold']:
            self.consecutive_ok_start = None
            return SchedulerMode.CHECKING
        
        # Quality unstable → CHECKING (need more observation)
        if not quality_stable:
            self.consecutive_ok_start = None
            return SchedulerMode.CHECKING
        
        # Current state is RISK → CHECKING
        if current_state == RiskState.RISK:
            self.consecutive_ok_start = None
            return SchedulerMode.CHECKING
        
        # --- Check for LOW_POWER mode ---
        to_low = trans['to_low_power']
        
        # Only go to LOW_POWER if:
        # 1. Confirmed state is OK
        # 2. Risk is very low
        # 3. Quality is good and stable
        # 4. Has been OK for required duration
        
        if confirmed_state != RiskState.OK:
            self.consecutive_ok_start = None
            return SchedulerMode.CHECKING
        
        if risk_score > to_low['risk_max']:
            self.consecutive_ok_start = None
            return SchedulerMode.CHECKING
        
        if to_low['require_high_quality'] and quality_score < to_low['quality_min']:
            self.consecutive_ok_start = None
            return SchedulerMode.CHECKING
        
        # Track consecutive OK time
        if self.consecutive_ok_start is None:
            self.consecutive_ok_start = now
        
        ok_duration = now - self.consecutive_ok_start
        
        if ok_duration >= to_low['consecutive_ok_seconds']:
            return SchedulerMode.LOW_POWER
        
        # Not enough time yet, stay in CHECKING
        return SchedulerMode.CHECKING
    
    def _is_quality_stable(self) -> bool:
        """
        Check if quality has been stable recently.
        
        Quality is unstable if variance exceeds threshold.
        """
        if len(self.quality_history) < 5:
            return True  # Not enough data, assume stable
        
        recent = self.quality_history[-10:]
        mean = sum(recent) / len(recent)
        variance = sum((q - mean) ** 2 for q in recent) / len(recent)
        
        threshold = self.config['transitions']['to_checking']['quality_unstable_threshold']
        
        # Variance > threshold^2 means unstable
        return variance < (threshold * 0.5) ** 2
    
    def get_mode_duration(self) -> float:
        """Get how long we've been in current mode (seconds)."""
        return time.time() - self.mode_start_time
    
    def force_mode(self, mode: SchedulerMode, reason: str = "FORCED"):
        """
        Force scheduler into a specific mode.
        
        Useful for external triggers (e.g., user pressing panic button).
        
        Args:
            mode: Mode to switch to
            reason: Reason for forced switch
        """
        self.current_mode = mode
        self.mode_start_time = time.time()
        self.last_mode_change = time.time()
    
    def reset(self):
        """Reset scheduler state (for new session)."""
        self.current_mode = SchedulerMode.CHECKING
        self.mode_start_time = time.time()
        self.last_mode_change = time.time()
        self.consecutive_ok_start = None
        self.quality_history.clear()


def get_mode_display(mode: SchedulerMode) -> Dict[str, str]:
    """
    Get display properties for a scheduler mode.
    
    Returns:
        Dictionary with 'label', 'color', 'icon' for UI display
    """
    display_map = {
        SchedulerMode.LOW_POWER: {
            'label': 'BAJO CONSUMO',
            'color': '#22c55e',  # Green
            'icon': '🔋',
        },
        SchedulerMode.CHECKING: {
            'label': 'VERIFICANDO',
            'color': '#f59e0b',  # Orange
            'icon': '👁',
        },
        SchedulerMode.CONFIRMING: {
            'label': 'CONFIRMANDO',
            'color': '#ef4444',  # Red
            'icon': '⚡',
        },
    }
    return display_map.get(mode, display_map[SchedulerMode.CHECKING])
