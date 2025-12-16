"""
Output publisher abstraction for results.

This module provides a unified interface for publishing detection results:
- Console output - for PC development
- ROS2 topics - for robot deployment (stub)

The abstraction allows swapping output destinations without changing
the rest of the pipeline.

Design Pattern:
- OutputPublisher (ABC): Abstract interface
- ConsolePublisher: Prints to stdout
- ROS2Publisher: Stub for ROS2 topic publishing
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime

# Import local modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.classifier import RiskState, ClassificationResult
from core.temporal import TemporalResult
from core.scheduler import SchedulerMode, SchedulerSettings


@dataclass
class SystemState:
    """
    Complete system state for publishing.
    
    Bundles all relevant information for output.
    """
    timestamp: float
    risk_state: RiskState
    confirmed_state: RiskState
    risk_score: float
    quality_score: float
    is_confirmed: bool
    scheduler_mode: SchedulerMode
    reason: str
    
    # Optional detailed info
    internal_pose: Optional[str] = None
    torso_angle: Optional[float] = None
    n_persons: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'risk_state': self.risk_state.value,
            'confirmed_state': self.confirmed_state.value,
            'risk_score': round(self.risk_score, 3),
            'quality_score': round(self.quality_score, 3),
            'is_confirmed': self.is_confirmed,
            'scheduler_mode': self.scheduler_mode.value,
            'reason': self.reason,
            'internal_pose': self.internal_pose,
            'torso_angle': round(self.torso_angle, 1) if self.torso_angle else None,
            'n_persons': self.n_persons,
        }


class OutputPublisher(ABC):
    """
    Abstract base class for output publishers.
    
    All publishers must implement:
    - publish(): Send system state to destination
    """
    
    @abstractmethod
    def publish(self, state: SystemState):
        """
        Publish the current system state.
        
        Args:
            state: Complete system state to publish
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close/cleanup the publisher."""
        pass


class ConsolePublisher(OutputPublisher):
    """
    Publisher that outputs to console (stdout).
    
    This is the primary implementation for PC development.
    Provides formatted, human-readable output.
    
    Args:
        verbose: If True, print detailed information
        color: If True, use ANSI colors for state highlighting
    """
    
    # ANSI color codes
    COLORS = {
        RiskState.OK: '\033[92m',        # Green
        RiskState.RISK: '\033[93m',      # Yellow
        RiskState.NEEDS_HELP: '\033[91m', # Red
        RiskState.UNKNOWN: '\033[90m',   # Gray
    }
    RESET = '\033[0m'
    
    def __init__(self, verbose: bool = False, color: bool = True):
        self.verbose = verbose
        self.color = color
        self.last_state: Optional[RiskState] = None
    
    def publish(self, state: SystemState):
        """Print state to console."""
        # Only print on state change or in verbose mode
        state_changed = state.confirmed_state != self.last_state
        self.last_state = state.confirmed_state
        
        if not state_changed and not self.verbose:
            return
        
        # Build output line
        if self.color:
            color = self.COLORS.get(state.confirmed_state, '')
            reset = self.RESET
        else:
            color = ''
            reset = ''
        
        time_str = datetime.fromtimestamp(state.timestamp).strftime('%H:%M:%S')
        
        line = (
            f"[{time_str}] "
            f"{color}{state.confirmed_state.value:11s}{reset} "
            f"risk={state.risk_score:.2f} "
            f"quality={state.quality_score:.2f} "
            f"mode={state.scheduler_mode.value}"
        )
        
        if state.is_confirmed:
            line += " [CONFIRMED]"
        
        print(line)
        
        if self.verbose:
            print(f"  Reason: {state.reason}")
            if state.torso_angle is not None:
                print(f"  Torso angle: {state.torso_angle:.1f}°")
    
    def close(self):
        """Nothing to close for console output."""
        pass


class ROS2Publisher(OutputPublisher):
    """
    Stub for ROS2 topic publisher.
    
    This class will be implemented when migrating to ROS2.
    It publishes system state to custom ROS2 message topics.
    
    Planned topics:
    - /fall_detector/state (custom msg with risk state)
    - /fall_detector/risk_score (Float32)
    - /fall_detector/needs_help (Bool, latched)
    
    TODO: Implement when ROS2 integration is needed.
    """
    
    def __init__(
        self,
        node_name: str = "fall_detector_publisher",
        state_topic: str = "/fall_detector/state"
    ):
        self.node_name = node_name
        self.state_topic = state_topic
        
        raise NotImplementedError(
            "ROS2Publisher is a stub for future implementation. "
            "Use ConsolePublisher for PC development."
        )
    
    def publish(self, state: SystemState):
        """Publish state to ROS2 topics."""
        raise NotImplementedError("ROS2 integration not yet implemented")
    
    def close(self):
        """Shutdown ROS2 node."""
        pass


def create_publisher(
    publisher_type: str = "console",
    **kwargs
) -> OutputPublisher:
    """
    Factory function to create appropriate publisher.
    
    Args:
        publisher_type: "console" or "ros2"
        **kwargs: Additional arguments for the publisher
    
    Returns:
        OutputPublisher instance
    """
    publishers = {
        'console': ConsolePublisher,
        'ros2': ROS2Publisher,
    }
    
    publisher_class = publishers.get(publisher_type.lower())
    if publisher_class is None:
        raise ValueError(f"Unknown publisher type: {publisher_type}")
    
    return publisher_class(**kwargs)
