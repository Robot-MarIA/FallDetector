"""
Inference backend abstraction for pose estimation.

This module provides a unified interface for different inference backends:
- Ultralytics YOLO (PyTorch) - for PC development
- TensorRT - for NVIDIA Jetson deployment (stub)
- DeepStream - for optimized Jetson pipelines (stub)

The abstraction allows swapping inference backends without changing
the rest of the pipeline, enabling seamless migration from
PC development to Jetson deployment.

Design Pattern:
- InferenceBackend (ABC): Abstract interface
- UltralyticsBackend: Concrete implementation using ultralytics library
- TensorRTBackend: Stub for future TensorRT optimization
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class PoseDetection:
    """
    Container for a single pose detection.
    
    Attributes:
        keypoints: List of 17 COCO keypoints as (x, y, confidence)
        bbox: Bounding box (x1, y1, x2, y2)
        confidence: Overall detection confidence
        person_id: Optional ID for tracking (if available)
    """
    keypoints: List[Tuple[float, float, float]]
    bbox: Tuple[float, float, float, float]
    confidence: float
    person_id: Optional[int] = None
    
    def get_bbox_area(self) -> float:
        """Calculate bounding box area."""
        x1, y1, x2, y2 = self.bbox
        return abs(x2 - x1) * abs(y2 - y1)
    
    def get_bbox_center(self) -> Tuple[float, float]:
        """Calculate bounding box center."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class InferenceBackend(ABC):
    """
    Abstract base class for inference backends.
    
    All inference backends must implement:
    - infer(): Run pose estimation on a frame
    - get_model_info(): Get model metadata
    """
    
    @abstractmethod
    def infer(self, frame: np.ndarray) -> List[PoseDetection]:
        """
        Run pose estimation on a frame.
        
        Args:
            frame: Input image as numpy array (BGR format)
        
        Returns:
            List of PoseDetection objects for each person detected
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        pass
    
    @abstractmethod
    def warmup(self):
        """Perform warmup inference for consistent performance."""
        pass


class UltralyticsBackend(InferenceBackend):
    """
    Inference backend using Ultralytics YOLO-Pose.
    
    This is the primary implementation for PC development.
    Uses PyTorch for inference with optional GPU acceleration.
    
    Args:
        model_path: Path to YOLO-Pose model (default: yolo11n-pose.pt)
        device: Device for inference ("cuda", "cpu", or "auto")
        conf_threshold: Minimum confidence for detections
    
    Example:
        backend = UltralyticsBackend("yolo11n-pose.pt")
        detections = backend.infer(frame)
    """
    
    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        device: str = "auto",
        conf_threshold: float = 0.5
    ):
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "Ultralytics not available. Install with: pip install ultralytics"
            )
        
        print(f"Loading YOLO-Pose model: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        # Set device
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"  Device: {self.device}")
        print(f"  Confidence threshold: {self.conf_threshold}")
    
    def infer(self, frame: np.ndarray) -> List[PoseDetection]:
        """
        Run YOLO-Pose inference on a frame.
        
        Args:
            frame: Input image (BGR format from OpenCV)
        
        Returns:
            List of PoseDetection objects
        """
        if self.model is None:
            return []
        
        # Run inference
        results = self.model(
            frame,
            device=self.device,
            conf=self.conf_threshold,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            if result.keypoints is None:
                continue
            
            # Get keypoints and boxes
            keypoints_data = result.keypoints.data.cpu().numpy()
            boxes_data = result.boxes.data.cpu().numpy()
            
            for i in range(len(keypoints_data)):
                # Extract keypoints (17 COCO keypoints, each with x, y, conf)
                kps = keypoints_data[i]
                keypoints = [
                    (float(kps[j][0]), float(kps[j][1]), float(kps[j][2]))
                    for j in range(len(kps))
                ]
                
                # Extract bounding box
                box = boxes_data[i]
                bbox = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                confidence = float(box[4])
                
                detections.append(PoseDetection(
                    keypoints=keypoints,
                    bbox=bbox,
                    confidence=confidence
                ))
        
        return detections
    
    def get_model_info(self) -> dict:
        """Get YOLO model information."""
        return {
            'model_path': self.model_path,
            'device': self.device,
            'backend': 'ultralytics',
            'conf_threshold': self.conf_threshold,
        }
    
    def warmup(self):
        """Perform warmup inference."""
        if self.model is not None:
            print("Warming up model...")
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.infer(dummy)
            print("Warmup complete")


class TensorRTBackend(InferenceBackend):
    """
    Stub for TensorRT inference backend.
    
    This class will be implemented for NVIDIA Jetson deployment.
    TensorRT provides optimized inference for NVIDIA GPUs.
    
    TODO: Implement when Jetson deployment is needed.
    
    Expected workflow:
    1. Export YOLO model to ONNX
    2. Convert ONNX to TensorRT engine
    3. Run inference using TensorRT runtime
    """
    
    def __init__(
        self,
        engine_path: str = "yolo11n-pose.engine",
        workspace_size: int = 1 << 30  # 1GB
    ):
        self.engine_path = engine_path
        self.workspace_size = workspace_size
        
        raise NotImplementedError(
            "TensorRTBackend is a stub for future implementation. "
            "Use UltralyticsBackend for PC development. "
            "For Jetson deployment, implement TensorRT engine loading and inference."
        )
    
    def infer(self, frame: np.ndarray) -> List[PoseDetection]:
        """Run TensorRT inference."""
        raise NotImplementedError("TensorRT integration not yet implemented")
    
    def get_model_info(self) -> dict:
        """Get TensorRT engine information."""
        return {
            'engine_path': self.engine_path,
            'backend': 'tensorrt',
        }
    
    def warmup(self):
        """TensorRT warmup."""
        pass


class DeepStreamBackend(InferenceBackend):
    """
    Stub for NVIDIA DeepStream inference backend.
    
    DeepStream provides optimized video analytics pipelines
    for Jetson devices with hardware-accelerated decoding,
    inference, and encoding.
    
    TODO: Implement when Jetson deployment with video streams is needed.
    """
    
    def __init__(self, config_path: str = "deepstream_config.txt"):
        self.config_path = config_path
        
        raise NotImplementedError(
            "DeepStreamBackend is a stub for future implementation. "
            "Use UltralyticsBackend for PC development."
        )
    
    def infer(self, frame: np.ndarray) -> List[PoseDetection]:
        """Run DeepStream inference."""
        raise NotImplementedError("DeepStream integration not yet implemented")
    
    def get_model_info(self) -> dict:
        """Get DeepStream pipeline information."""
        return {
            'config_path': self.config_path,
            'backend': 'deepstream',
        }
    
    def warmup(self):
        """DeepStream warmup."""
        pass


def create_inference_backend(
    backend_type: str = "ultralytics",
    **kwargs
) -> InferenceBackend:
    """
    Factory function to create appropriate inference backend.
    
    Args:
        backend_type: "ultralytics", "tensorrt", or "deepstream"
        **kwargs: Additional arguments for the backend
    
    Returns:
        InferenceBackend instance
    
    Example:
        backend = create_inference_backend("ultralytics", model_path="yolo11n-pose.pt")
    """
    backends = {
        'ultralytics': UltralyticsBackend,
        'tensorrt': TensorRTBackend,
        'deepstream': DeepStreamBackend,
    }
    
    backend_class = backends.get(backend_type.lower())
    if backend_class is None:
        raise ValueError(f"Unknown backend type: {backend_type}. Available: {list(backends.keys())}")
    
    return backend_class(**kwargs)
