"""
Visualization utilities for pose overlay.

This module provides visual overlays for debugging and demonstration:
- Skeleton drawing with confidence-based coloring
- State badge display
- Score gauges
- Scheduler mode indicator

The visualization is optional and can be disabled for production.
"""

from typing import Optional, Tuple, List, Dict
import numpy as np

# OpenCV import with error handling
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None

# Import local modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.classifier import RiskState
from core.scheduler import SchedulerMode


# COCO skeleton connections for drawing
SKELETON_CONNECTIONS = [
    # Face
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Upper body
    (5, 6),  # Shoulders
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    # Torso
    (5, 11), (6, 12), (11, 12),
    # Lower body
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]

# Colors for different states (BGR format for OpenCV)
STATE_COLORS = {
    RiskState.OK: (0, 200, 0),          # Green
    RiskState.RISK: (0, 165, 255),      # Orange
    RiskState.NEEDS_HELP: (0, 0, 255),  # Red
    RiskState.UNKNOWN: (128, 128, 128), # Gray
}

MODE_COLORS = {
    SchedulerMode.LOW_POWER: (0, 200, 0),
    SchedulerMode.CHECKING: (0, 165, 255),
    SchedulerMode.CONFIRMING: (0, 0, 255),
}


class PoseVisualizer:
    """
    Visualizes pose detection results on video frames.
    
    Features:
    - Skeleton overlay with confidence-based coloring
    - Current state badge
    - Risk and quality score display
    - Scheduler mode indicator
    - Bounding box with state coloring
    """
    
    def __init__(
        self,
        show_skeleton: bool = True,
        show_bbox: bool = True,
        show_scores: bool = True,
        show_state: bool = True,
        show_mode: bool = True,
        skeleton_thickness: int = 2,
        keypoint_radius: int = 4,
        font_scale: float = 0.6
    ):
        if not OPENCV_AVAILABLE:
            raise RuntimeError("OpenCV not available for visualization")
        
        self.show_skeleton = show_skeleton
        self.show_bbox = show_bbox
        self.show_scores = show_scores
        self.show_state = show_state
        self.show_mode = show_mode
        self.skeleton_thickness = skeleton_thickness
        self.keypoint_radius = keypoint_radius
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw(
        self,
        frame: np.ndarray,
        keypoints: Optional[List[Optional[Tuple[float, float, float]]]] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        state: RiskState = RiskState.UNKNOWN,
        risk_score: float = 0.0,
        quality_score: float = 0.0,
        scheduler_mode: SchedulerMode = SchedulerMode.CHECKING,
        is_confirmed: bool = False,
        reason: str = ""
    ) -> np.ndarray:
        """
        Draw all visual overlays on a frame.
        
        Args:
            frame: Input frame (will be modified in-place)
            keypoints: List of 17 keypoints or None
            bbox: Bounding box (x1, y1, x2, y2)
            state: Current risk state
            risk_score: Risk score [0, 1]
            quality_score: Quality score [0, 1]
            scheduler_mode: Current scheduler mode
            is_confirmed: Whether state is temporally confirmed
            reason: Reason string for display
        
        Returns:
            Frame with overlays drawn
        """
        overlay = frame.copy()
        
        state_color = STATE_COLORS.get(state, STATE_COLORS[RiskState.UNKNOWN])
        
        # Draw bounding box
        if self.show_bbox and bbox is not None:
            self._draw_bbox(overlay, bbox, state_color, is_confirmed)
        
        # Draw skeleton
        if self.show_skeleton and keypoints is not None:
            self._draw_skeleton(overlay, keypoints)
        
        # Draw state badge
        if self.show_state:
            self._draw_state_badge(overlay, state, is_confirmed)
        
        # Draw scores
        if self.show_scores:
            self._draw_scores(overlay, risk_score, quality_score)
        
        # Draw scheduler mode
        if self.show_mode:
            self._draw_mode(overlay, scheduler_mode)
        
        # Draw reason (at bottom)
        if reason:
            self._draw_reason(overlay, reason)
        
        return overlay
    
    def _draw_bbox(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        color: Tuple[int, int, int],
        is_confirmed: bool
    ):
        """Draw bounding box with state coloring."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        thickness = 3 if is_confirmed else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    def _draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: List[Optional[Tuple[float, float, float]]]
    ):
        """Draw skeleton with confidence-based coloring."""
        # Draw connections
        for i, j in SKELETON_CONNECTIONS:
            if i < len(keypoints) and j < len(keypoints):
                kp1 = keypoints[i]
                kp2 = keypoints[j]
                
                if kp1 is not None and kp2 is not None:
                    # Color based on average confidence
                    avg_conf = (kp1[2] + kp2[2]) / 2
                    color = self._confidence_color(avg_conf)
                    
                    pt1 = (int(kp1[0]), int(kp1[1]))
                    pt2 = (int(kp2[0]), int(kp2[1]))
                    cv2.line(frame, pt1, pt2, color, self.skeleton_thickness)
        
        # Draw keypoints
        for kp in keypoints:
            if kp is not None:
                color = self._confidence_color(kp[2])
                center = (int(kp[0]), int(kp[1]))
                cv2.circle(frame, center, self.keypoint_radius, color, -1)
    
    def _confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """Get color based on confidence (green = high, red = low)."""
        # Interpolate between red and green
        g = int(confidence * 255)
        r = int((1 - confidence) * 255)
        return (0, g, r)
    
    def _draw_state_badge(
        self,
        frame: np.ndarray,
        state: RiskState,
        is_confirmed: bool
    ):
        """Draw state badge in top-left corner."""
        color = STATE_COLORS.get(state, STATE_COLORS[RiskState.UNKNOWN])
        
        # State text
        text = state.value
        if is_confirmed:
            text += " ✓"
        
        # Background rectangle
        (text_w, text_h), baseline = cv2.getTextSize(text, self.font, self.font_scale * 1.5, 2)
        cv2.rectangle(frame, (10, 10), (20 + text_w, 20 + text_h + baseline), color, -1)
        
        # Text (white on colored background)
        cv2.putText(frame, text, (15, 15 + text_h), self.font, self.font_scale * 1.5, (255, 255, 255), 2)
    
    def _draw_scores(
        self,
        frame: np.ndarray,
        risk_score: float,
        quality_score: float
    ):
        """Draw risk and quality score bars."""
        h, w = frame.shape[:2]
        bar_width = 150
        bar_height = 15
        x_start = w - bar_width - 20
        
        # Risk score bar
        y = 20
        self._draw_score_bar(frame, x_start, y, bar_width, bar_height, 
                            risk_score, "Risk", (0, 0, 200))
        
        # Quality score bar
        y = 45
        self._draw_score_bar(frame, x_start, y, bar_width, bar_height,
                            quality_score, "Quality", (200, 100, 0))
    
    def _draw_score_bar(
        self,
        frame: np.ndarray,
        x: int, y: int,
        width: int, height: int,
        value: float,
        label: str,
        color: Tuple[int, int, int]
    ):
        """Draw a single score bar."""
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
        
        # Filled portion
        fill_width = int(width * value)
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
        
        # Border
        cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 1)
        
        # Label
        text = f"{label}: {value:.2f}"
        cv2.putText(frame, text, (x, y - 3), self.font, self.font_scale * 0.7, (255, 255, 255), 1)
    
    def _draw_mode(self, frame: np.ndarray, mode: SchedulerMode):
        """Draw scheduler mode indicator."""
        h, w = frame.shape[:2]
        color = MODE_COLORS.get(mode, (128, 128, 128))
        
        text = f"Mode: {mode.value}"
        (text_w, text_h), _ = cv2.getTextSize(text, self.font, self.font_scale, 1)
        
        x = w - text_w - 20
        y = 80
        
        cv2.putText(frame, text, (x, y), self.font, self.font_scale, color, 2)
    
    def _draw_reason(self, frame: np.ndarray, reason: str):
        """Draw reason string at bottom of frame."""
        h, w = frame.shape[:2]
        
        # Truncate if too long
        max_chars = w // 8
        if len(reason) > max_chars:
            reason = reason[:max_chars-3] + "..."
        
        y = h - 15
        cv2.putText(frame, reason, (10, y), self.font, self.font_scale * 0.7, (200, 200, 200), 1)


def create_visualizer(**kwargs) -> PoseVisualizer:
    """Factory function to create a visualizer."""
    return PoseVisualizer(**kwargs)
