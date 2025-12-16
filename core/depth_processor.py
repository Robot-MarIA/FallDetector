"""
Depth Processing Module for Fall Detection.

Provides:
- Floor plane estimation using RANSAC
- Depth sampling with median filtering
- Height-above-floor calculation for keypoints
- Temporal smoothing of floor plane

Physical setup:
- Camera height: 0.63m above floor
- Typical distance: 1.5m
- Legs/hips visible, feet may be out of frame
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from collections import deque

try:
    import pyrealsense2 as rs
    RS_AVAILABLE = True
except ImportError:
    RS_AVAILABLE = False


@dataclass
class FloorPlane:
    """Estimated floor plane: ax + by + cz + d = 0"""
    a: float
    b: float
    c: float
    d: float
    quality: float  # 0-1, based on inlier ratio
    
    def distance_to_point(self, x: float, y: float, z: float) -> float:
        """Signed distance from point to plane (positive = above)."""
        num = self.a * x + self.b * y + self.c * z + self.d
        denom = np.sqrt(self.a**2 + self.b**2 + self.c**2)
        return num / denom if denom > 0 else 0.0


@dataclass
class DepthFeatures:
    """Depth-derived features for classification."""
    hip_height_m: Optional[float] = None
    knee_height_m: Optional[float] = None
    shoulder_height_m: Optional[float] = None
    floor_quality: float = 0.0
    depth_valid_ratio: float = 0.0
    depth_mode: str = "none"  # "plane", "fallback", "none"
    
    # Temporal features
    vertical_drop_m: float = 0.0
    vertical_velocity_mps: float = 0.0
    floor_contact_time_s: float = 0.0


class DepthProcessor:
    """
    Processes RealSense depth for fall detection.
    
    Key features:
    - RANSAC floor plane estimation from lower ROI
    - Robust depth sampling with median filter
    - Height-above-floor for keypoints
    - Temporal smoothing and tracking
    """
    
    def __init__(
        self,
        camera_height_m: float = 0.63,
        sample_window: int = 7,
        floor_roi_ratio: float = 0.35,
        min_floor_quality: float = 0.55,
        ema_alpha: float = 0.3,
        temporal_window_s: float = 1.2,
        fps: float = 30.0,
    ):
        self.camera_height_m = camera_height_m
        self.sample_window = sample_window
        self.floor_roi_ratio = floor_roi_ratio
        self.min_floor_quality = min_floor_quality
        self.ema_alpha = ema_alpha
        self.temporal_window_s = temporal_window_s
        
        # Filters
        self.temporal_filter = None
        self.spatial_filter = None
        self.hole_filter = None
        
        if RS_AVAILABLE:
            self.temporal_filter = rs.temporal_filter()
            self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
            self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
            
            self.spatial_filter = rs.spatial_filter()
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
            
            self.hole_filter = rs.hole_filling_filter()
        
        # Smoothed floor plane
        self._floor_plane: Optional[FloorPlane] = None
        
        # Temporal buffer for hip height
        buffer_size = int(temporal_window_s * fps)
        self._hip_height_buffer: deque = deque(maxlen=buffer_size)
        self._last_hip_height: Optional[float] = None
        self._floor_contact_start: Optional[float] = None
        
        # Intrinsics (set when processing)
        self._intrinsics = None
    
    def set_intrinsics(self, intrinsics):
        """Set camera intrinsics for 3D projection."""
        self._intrinsics = intrinsics
    
    def filter_depth(self, depth_frame):
        """Apply temporal + spatial + hole filling filters."""
        if not RS_AVAILABLE:
            return depth_frame
        
        filtered = depth_frame
        if self.temporal_filter:
            filtered = self.temporal_filter.process(filtered)
        if self.spatial_filter:
            filtered = self.spatial_filter.process(filtered)
        if self.hole_filter:
            filtered = self.hole_filter.process(filtered)
        
        return filtered
    
    def depth_at(
        self,
        depth_image: np.ndarray,
        x: int,
        y: int,
        win: int = None
    ) -> Optional[float]:
        """
        Get robust depth at (x, y) using median of window.
        
        Returns depth in METERS, or None if not valid.
        """
        if win is None:
            win = self.sample_window
        
        h, w = depth_image.shape[:2]
        half = win // 2
        
        # Clamp to image bounds
        x1 = max(0, x - half)
        x2 = min(w, x + half + 1)
        y1 = max(0, y - half)
        y2 = min(h, y + half + 1)
        
        patch = depth_image[y1:y2, x1:x2].flatten()
        
        # Filter valid values (RealSense depth in mm, 0 = invalid)
        valid = patch[(patch > 200) & (patch < 6000)]  # 0.2m - 6m
        
        if len(valid) < (win * win) // 4:
            return None
        
        # Median in meters
        return float(np.median(valid)) / 1000.0
    
    def pixel_to_3d(
        self,
        x: float,
        y: float,
        depth_m: float
    ) -> Optional[Tuple[float, float, float]]:
        """
        Convert pixel + depth to 3D point in camera frame.
        
        Returns (X, Y, Z) in meters.
        """
        if self._intrinsics is None or depth_m is None:
            return None
        
        # RealSense convention: X right, Y down, Z forward
        fx = self._intrinsics.fx
        fy = self._intrinsics.fy
        cx = self._intrinsics.ppx
        cy = self._intrinsics.ppy
        
        X = (x - cx) * depth_m / fx
        Y = (y - cy) * depth_m / fy
        Z = depth_m
        
        return (X, Y, Z)
    
    def estimate_floor_plane(
        self,
        depth_image: np.ndarray,
        max_iterations: int = 50,
        distance_threshold: float = 0.03,  # 3cm
    ) -> Optional[FloorPlane]:
        """
        Estimate floor plane using RANSAC on lower ROI.
        
        Uses bottom 35% of frame where floor is likely visible.
        """
        if self._intrinsics is None:
            return None
        
        h, w = depth_image.shape[:2]
        roi_top = int(h * (1 - self.floor_roi_ratio))
        
        # Sample points from ROI
        roi = depth_image[roi_top:, :]
        
        # Get valid points
        points_3d = []
        sample_step = 8  # Don't need every pixel
        
        for py in range(0, roi.shape[0], sample_step):
            for px in range(0, roi.shape[1], sample_step):
                depth_mm = roi[py, px]
                if 200 < depth_mm < 6000:
                    depth_m = depth_mm / 1000.0
                    pt3d = self.pixel_to_3d(px, roi_top + py, depth_m)
                    if pt3d:
                        points_3d.append(pt3d)
        
        if len(points_3d) < 50:
            return None
        
        points = np.array(points_3d)
        
        # RANSAC plane fitting
        best_plane = None
        best_inliers = 0
        n_points = len(points)
        
        for _ in range(max_iterations):
            # Random 3 points
            idx = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = points[idx]
            
            # Compute plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm
            
            # Plane: a*x + b*y + c*z + d = 0
            a, b, c = normal
            d = -np.dot(normal, p1)
            
            # Count inliers
            distances = np.abs(points @ normal + d)
            inliers = np.sum(distances < distance_threshold)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = (a, b, c, d)
        
        if best_plane is None:
            return None
        
        quality = best_inliers / n_points
        
        # Create plane and smooth with EMA
        new_plane = FloorPlane(
            a=best_plane[0],
            b=best_plane[1],
            c=best_plane[2],
            d=best_plane[3],
            quality=quality
        )
        
        # EMA smoothing
        if self._floor_plane is not None:
            alpha = self.ema_alpha
            new_plane = FloorPlane(
                a=alpha * new_plane.a + (1 - alpha) * self._floor_plane.a,
                b=alpha * new_plane.b + (1 - alpha) * self._floor_plane.b,
                c=alpha * new_plane.c + (1 - alpha) * self._floor_plane.c,
                d=alpha * new_plane.d + (1 - alpha) * self._floor_plane.d,
                quality=quality
            )
        
        self._floor_plane = new_plane
        return new_plane
    
    def get_keypoint_height(
        self,
        depth_image: np.ndarray,
        x: float,
        y: float,
        plane: Optional[FloorPlane]
    ) -> Optional[float]:
        """
        Get height of keypoint above floor plane.
        
        Returns height in meters, or None if not computable.
        """
        depth_m = self.depth_at(depth_image, int(x), int(y))
        if depth_m is None:
            return None
        
        pt3d = self.pixel_to_3d(x, y, depth_m)
        if pt3d is None:
            return None
        
        if plane is None:
            return None
        
        return plane.distance_to_point(*pt3d)
    
    def process(
        self,
        depth_image: np.ndarray,
        keypoints: List[Optional[Tuple[float, float, float]]],
        timestamp: float
    ) -> DepthFeatures:
        """
        Process depth and keypoints to extract height features.
        
        Args:
            depth_image: Raw depth in mm (uint16)
            keypoints: 17 COCO keypoints [(x, y, conf), ...]
            timestamp: Current time in seconds
        
        Returns:
            DepthFeatures with heights and temporal info
        """
        features = DepthFeatures()
        
        # Estimate floor plane
        plane = self.estimate_floor_plane(depth_image)
        
        if plane is None or plane.quality < self.min_floor_quality:
            features.depth_mode = "none"
            features.floor_quality = plane.quality if plane else 0.0
            return features
        
        features.floor_quality = plane.quality
        features.depth_mode = "plane"
        
        # Keypoint indices (COCO)
        LEFT_HIP, RIGHT_HIP = 11, 12
        LEFT_KNEE, RIGHT_KNEE = 13, 14
        LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
        
        # Get hip height (average of left/right if both valid)
        hip_heights = []
        valid_count = 0
        total_kp = 0
        
        for idx in [LEFT_HIP, RIGHT_HIP]:
            if idx < len(keypoints) and keypoints[idx] is not None:
                kp = keypoints[idx]
                if kp[2] >= 0.3:  # confidence threshold
                    total_kp += 1
                    h = self.get_keypoint_height(depth_image, kp[0], kp[1], plane)
                    if h is not None:
                        hip_heights.append(h)
                        valid_count += 1
        
        if hip_heights:
            features.hip_height_m = float(np.mean(hip_heights))
        
        # Get knee height
        knee_heights = []
        for idx in [LEFT_KNEE, RIGHT_KNEE]:
            if idx < len(keypoints) and keypoints[idx] is not None:
                kp = keypoints[idx]
                if kp[2] >= 0.3:
                    total_kp += 1
                    h = self.get_keypoint_height(depth_image, kp[0], kp[1], plane)
                    if h is not None:
                        knee_heights.append(h)
                        valid_count += 1
        
        if knee_heights:
            features.knee_height_m = float(np.mean(knee_heights))
        
        # Get shoulder height (optional)
        shoulder_heights = []
        for idx in [LEFT_SHOULDER, RIGHT_SHOULDER]:
            if idx < len(keypoints) and keypoints[idx] is not None:
                kp = keypoints[idx]
                if kp[2] >= 0.3:
                    h = self.get_keypoint_height(depth_image, kp[0], kp[1], plane)
                    if h is not None:
                        shoulder_heights.append(h)
        
        if shoulder_heights:
            features.shoulder_height_m = float(np.mean(shoulder_heights))
        
        # Depth valid ratio
        features.depth_valid_ratio = valid_count / max(total_kp, 1)
        
        # Temporal tracking
        if features.hip_height_m is not None:
            self._hip_height_buffer.append((timestamp, features.hip_height_m))
            
            # Vertical drop (max - min in window)
            if len(self._hip_height_buffer) > 5:
                heights = [h for _, h in self._hip_height_buffer]
                features.vertical_drop_m = max(heights) - min(heights)
                
                # Velocity (smoothed derivative)
                if self._last_hip_height is not None:
                    dt = timestamp - self._hip_height_buffer[-2][0] if len(self._hip_height_buffer) > 1 else 0.033
                    if dt > 0:
                        raw_vel = (features.hip_height_m - self._last_hip_height) / dt
                        # Negative = falling
                        features.vertical_velocity_mps = -raw_vel
            
            self._last_hip_height = features.hip_height_m
            
            # Floor contact time
            HIP_FLOOR_THRESH = 0.30
            if features.hip_height_m < HIP_FLOOR_THRESH:
                if self._floor_contact_start is None:
                    self._floor_contact_start = timestamp
                features.floor_contact_time_s = timestamp - self._floor_contact_start
            else:
                self._floor_contact_start = None
                features.floor_contact_time_s = 0.0
        
        return features
