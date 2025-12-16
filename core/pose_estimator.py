"""
Pose estimator with person selection.

This module wraps the inference backend and handles:
- Running pose estimation on frames
- Selecting the primary person when multiple are detected
- Providing structured keypoint data

Person Selection Strategy:
When multiple people are detected, the system must choose one
to focus on. Two strategies are supported:
1. Largest bounding box (likely closest/most visible person)
2. Most centered in frame (likely the subject of interest)

The selected person is the one analyzed for risk detection.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import yaml
from pathlib import Path
import numpy as np

# Import local modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.inference_backend import InferenceBackend, PoseDetection, create_inference_backend


# COCO keypoint names for reference
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


@dataclass
class SelectedPose:
    """
    Selected pose after person selection.
    
    Attributes:
        keypoints: 17 COCO keypoints as (x, y, confidence) or None if missing
        bbox: Bounding box (x1, y1, x2, y2)
        confidence: Detection confidence
        selection_reason: Why this person was selected
        n_persons_detected: Total number of people in frame
    """
    keypoints: List[Optional[Tuple[float, float, float]]]
    bbox: Tuple[float, float, float, float]
    confidence: float
    selection_reason: str
    n_persons_detected: int
    
    def get_keypoint(self, name: str) -> Optional[Tuple[float, float, float]]:
        """Get a keypoint by name."""
        if name in KEYPOINT_NAMES:
            idx = KEYPOINT_NAMES.index(name)
            return self.keypoints[idx]
        return None
    
    def get_bbox_center(self) -> Tuple[float, float]:
        """Get bounding box center."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class PoseEstimator:
    """
    High-level pose estimator with person selection.
    
    This class:
    1. Runs pose estimation using the configured backend
    2. Selects the primary person based on configured strategy
    3. Returns structured pose data for analysis
    
    The selection strategy is important because:
    - In home care, the person nearest the camera is usually the subject
    - In robot scenarios, the centered person is often the target
    
    Args:
        backend: Inference backend to use
        config_path: Path to thresholds.yaml for selection config
    """
    
    def __init__(
        self,
        backend: Optional[InferenceBackend] = None,
        config_path: Optional[str] = None
    ):
        self.backend = backend
        self.config = self._load_config(config_path)
        
        # Create default backend if none provided
        if self.backend is None:
            self.backend = create_inference_backend("ultralytics")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML or use defaults."""
        defaults = {
            'method': 'largest_bbox',  # or 'most_centered'
            'center_weight': 0.3,
            'min_bbox_area': 0.01,
            'min_confidence': 0.5,
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    full_config = yaml.safe_load(f)
                    if 'person_selection' in full_config:
                        defaults.update(full_config['person_selection'])
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        return defaults
    
    def estimate(
        self,
        frame: np.ndarray,
        frame_shape: Optional[Tuple[int, int]] = None
    ) -> Optional[SelectedPose]:
        """
        Run pose estimation and select primary person.
        
        Args:
            frame: Input image (BGR format)
            frame_shape: Optional (height, width) for center calculation
        
        Returns:
            SelectedPose for the primary person, or None if no valid detection
        """
        if frame_shape is None:
            frame_shape = (frame.shape[0], frame.shape[1])
        
        # Run inference
        detections = self.backend.infer(frame)
        
        if not detections:
            return None
        
        # Filter by minimum area
        frame_area = frame_shape[0] * frame_shape[1]
        min_area = self.config['min_bbox_area'] * frame_area
        
        valid_detections = [
            d for d in detections 
            if d.get_bbox_area() >= min_area
        ]
        
        if not valid_detections:
            return None
        
        # Select primary person
        selected, reason = self._select_person(valid_detections, frame_shape)
        
        if selected is None:
            return None
        
        # Convert keypoints (mark low-confidence as None)
        keypoints = []
        min_conf = self.config['min_confidence']
        
        for kp in selected.keypoints:
            if kp[2] >= min_conf:
                keypoints.append(kp)
            else:
                keypoints.append(None)
        
        return SelectedPose(
            keypoints=keypoints,
            bbox=selected.bbox,
            confidence=selected.confidence,
            selection_reason=reason,
            n_persons_detected=len(detections),
        )
    
    def _select_person(
        self,
        detections: List[PoseDetection],
        frame_shape: Tuple[int, int]
    ) -> Tuple[Optional[PoseDetection], str]:
        """
        Select the primary person from multiple detections.
        
        Implements two strategies:
        1. largest_bbox: Select person with largest bounding box
        2. most_centered: Select person closest to frame center
        
        Returns:
            (selected_detection, reason_string)
        """
        if len(detections) == 1:
            return detections[0], "ONLY_PERSON"
        
        method = self.config['method']
        frame_height, frame_width = frame_shape
        frame_center = (frame_width / 2, frame_height / 2)
        
        if method == 'largest_bbox':
            # Sort by bbox area (largest first)
            sorted_detections = sorted(
                detections,
                key=lambda d: d.get_bbox_area(),
                reverse=True
            )
            return sorted_detections[0], "LARGEST_BBOX"
        
        elif method == 'most_centered':
            # Sort by distance to center (closest first)
            def center_distance(d: PoseDetection) -> float:
                cx, cy = d.get_bbox_center()
                return ((cx - frame_center[0]) ** 2 + (cy - frame_center[1]) ** 2) ** 0.5
            
            sorted_detections = sorted(detections, key=center_distance)
            return sorted_detections[0], "MOST_CENTERED"
        
        elif method == 'hybrid':
            # Combine area and center distance with weighting
            frame_diagonal = (frame_width ** 2 + frame_height ** 2) ** 0.5
            max_area = max(d.get_bbox_area() for d in detections)
            center_weight = self.config['center_weight']
            
            def hybrid_score(d: PoseDetection) -> float:
                # Normalize area score (0-1)
                area_score = d.get_bbox_area() / max_area if max_area > 0 else 0
                
                # Normalize center score (0-1, inverted so center = high score)
                cx, cy = d.get_bbox_center()
                center_dist = ((cx - frame_center[0]) ** 2 + (cy - frame_center[1]) ** 2) ** 0.5
                center_score = 1.0 - (center_dist / (frame_diagonal / 2))
                center_score = max(0, center_score)
                
                # Weighted combination
                return (1 - center_weight) * area_score + center_weight * center_score
            
            sorted_detections = sorted(detections, key=hybrid_score, reverse=True)
            return sorted_detections[0], "HYBRID_SELECTION"
        
        # Fallback to largest
        sorted_detections = sorted(
            detections,
            key=lambda d: d.get_bbox_area(),
            reverse=True
        )
        return sorted_detections[0], "FALLBACK_LARGEST"
    
    def get_all_detections(self, frame: np.ndarray) -> List[PoseDetection]:
        """
        Get all detections without selection (for debugging/visualization).
        
        Args:
            frame: Input image
        
        Returns:
            List of all PoseDetection objects
        """
        return self.backend.infer(frame)
    
    def warmup(self):
        """Warmup the inference backend."""
        if self.backend:
            self.backend.warmup()
