"""
Frame source abstraction for video input.

This module provides a unified interface for different frame sources:
- OpenCV (webcam, video file) - for PC development
- ROS2 Image topics - for robot deployment (stub)

The abstraction allows swapping frame sources without changing
the rest of the pipeline, enabling seamless migration from
PC testing to ROS2 deployment.

Design Pattern:
- FrameSource (ABC): Abstract interface
- OpenCVFrameSource: Concrete implementation for PC
- ROS2ImageSource: Stub for future ROS2 integration
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Generator
import time

import numpy as np

# OpenCV import with error handling
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None


@dataclass
class FrameData:
    """
    Container for a video frame with metadata.
    
    Attributes:
        frame: The image as numpy array (BGR format for OpenCV)
        timestamp: Capture timestamp (seconds since epoch)
        frame_number: Sequential frame number
        source_fps: Native FPS of the source
        width: Frame width in pixels
        height: Frame height in pixels
    """
    frame: np.ndarray
    timestamp: float
    frame_number: int
    source_fps: float
    width: int
    height: int


class FrameSource(ABC):
    """
    Abstract base class for frame sources.
    
    All frame sources must implement:
    - get_frame(): Get the next frame
    - release(): Release resources
    - is_open(): Check if source is available
    """
    
    @abstractmethod
    def get_frame(self) -> Optional[FrameData]:
        """
        Get the next frame from the source.
        
        Returns:
            FrameData if successful, None if no frame available
        """
        pass
    
    @abstractmethod
    def release(self):
        """Release resources (close video file, release camera, etc.)."""
        pass
    
    @abstractmethod
    def is_open(self) -> bool:
        """Check if the source is open and available."""
        pass
    
    @abstractmethod
    def get_fps(self) -> float:
        """Get the native FPS of the source."""
        pass
    
    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """Get the resolution (width, height) of the source."""
        pass
    
    def frames(self) -> Generator[FrameData, None, None]:
        """
        Generator that yields frames until source is exhausted.
        
        Yields:
            FrameData for each available frame
        """
        while self.is_open():
            frame_data = self.get_frame()
            if frame_data is None:
                break
            yield frame_data


class OpenCVFrameSource(FrameSource):
    """
    Frame source using OpenCV for webcam or video file input.
    
    This is the primary implementation for PC development and testing.
    
    Args:
        source: Either an integer (camera index) or string (video file path)
        target_fps: Optional target FPS (for throttling high-speed sources)
        resolution_scale: Optional scale factor for resolution (0.5 = half size)
    
    Example:
        # Webcam
        source = OpenCVFrameSource(0)
        
        # Video file
        source = OpenCVFrameSource("path/to/video.mp4")
        
        # With throttling and scaling
        source = OpenCVFrameSource(0, target_fps=15, resolution_scale=0.75)
    """
    
    def __init__(
        self,
        source: int | str,
        target_fps: Optional[float] = None,
        resolution_scale: float = 1.0
    ):
        if not OPENCV_AVAILABLE:
            raise RuntimeError("OpenCV not available. Install with: pip install opencv-python")
        
        self.source_id = source
        self.target_fps = target_fps
        self.resolution_scale = resolution_scale
        
        # Open video capture
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            source_type = "webcam" if isinstance(source, int) else "video file"
            raise RuntimeError(f"Could not open {source_type}: {source}")
        
        # Get source properties
        self._native_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Frame counting
        self._frame_number = 0
        self._last_frame_time = 0.0
        
        # Calculate target frame interval for throttling
        if target_fps and target_fps < self._native_fps:
            self._target_interval = 1.0 / target_fps
        else:
            self._target_interval = 0.0
        
        print(f"Opened {source}: {self._width}x{self._height} @ {self._native_fps:.1f} FPS")
        if resolution_scale != 1.0:
            scaled_w = int(self._width * resolution_scale)
            scaled_h = int(self._height * resolution_scale)
            print(f"  Scaling to: {scaled_w}x{scaled_h}")
    
    def get_frame(self) -> Optional[FrameData]:
        """Get the next frame from OpenCV capture."""
        if not self.is_open():
            return None
        
        # Throttle if target FPS is set
        if self._target_interval > 0:
            elapsed = time.time() - self._last_frame_time
            if elapsed < self._target_interval:
                time.sleep(self._target_interval - elapsed)
        
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            return None
        
        timestamp = time.time()
        self._last_frame_time = timestamp
        self._frame_number += 1
        
        # Apply resolution scaling if needed
        if self.resolution_scale != 1.0:
            new_width = int(frame.shape[1] * self.resolution_scale)
            new_height = int(frame.shape[0] * self.resolution_scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        return FrameData(
            frame=frame,
            timestamp=timestamp,
            frame_number=self._frame_number,
            source_fps=self._native_fps,
            width=frame.shape[1],
            height=frame.shape[0],
        )
    
    def release(self):
        """Release OpenCV capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("Frame source released")
    
    def is_open(self) -> bool:
        """Check if OpenCV capture is open."""
        return self.cap is not None and self.cap.isOpened()
    
    def get_fps(self) -> float:
        """Get the native FPS of the source."""
        return self._native_fps
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get the current resolution (after scaling)."""
        if self.resolution_scale != 1.0:
            return (
                int(self._width * self.resolution_scale),
                int(self._height * self.resolution_scale)
            )
        return (self._width, self._height)
    
    def set_resolution_scale(self, scale: float):
        """
        Update resolution scale dynamically.
        
        This allows the scheduler to adjust resolution on the fly.
        
        Args:
            scale: Scale factor [0.1, 1.0]
        """
        self.resolution_scale = max(0.1, min(1.0, scale))
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


class ROS2ImageSource(FrameSource):
    """
    Stub for ROS2 image topic subscription.
    
    This class will be implemented when migrating to ROS2.
    It subscribes to a sensor_msgs/Image topic and provides
    frames through the same interface.
    
    TODO: Implement when ROS2 integration is needed.
    """
    
    def __init__(
        self,
        topic_name: str = "/camera/image_raw",
        queue_size: int = 1
    ):
        self.topic_name = topic_name
        self.queue_size = queue_size
        self._is_open = False
        
        # Placeholder for ROS2 node and subscription
        # self.node = None
        # self.subscription = None
        # self.latest_frame = None
        
        raise NotImplementedError(
            "ROS2ImageSource is a stub for future implementation. "
            "Use OpenCVFrameSource for PC development."
        )
    
    def get_frame(self) -> Optional[FrameData]:
        """Get the latest frame from ROS2 topic."""
        raise NotImplementedError("ROS2 integration not yet implemented")
    
    def release(self):
        """Release ROS2 subscription."""
        raise NotImplementedError("ROS2 integration not yet implemented")
    
    def is_open(self) -> bool:
        """Check if ROS2 subscription is active."""
        return self._is_open
    
    def get_fps(self) -> float:
        """Get estimated FPS from ROS2 topic."""
        return 30.0  # Placeholder
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get resolution from ROS2 camera info."""
        return (640, 480)  # Placeholder


def create_frame_source(
    source_type: str,
    source: int | str = 0,
    **kwargs
) -> FrameSource:
    """
    Factory function to create appropriate frame source.
    
    Args:
        source_type: "opencv" or "ros2"
        source: Camera index or video path (for opencv)
        **kwargs: Additional arguments for the source
    
    Returns:
        FrameSource instance
    
    Example:
        source = create_frame_source("opencv", 0, target_fps=15)
        source = create_frame_source("opencv", "video.mp4")
    """
    if source_type.lower() == "opencv":
        return OpenCVFrameSource(source, **kwargs)
    elif source_type.lower() == "ros2":
        return ROS2ImageSource(**kwargs)
    else:
        raise ValueError(f"Unknown source type: {source_type}")
