"""
Geometry utilities for pose analysis.

This module provides mathematical functions for computing angles, distances,
and spatial relationships between keypoints. All functions are designed to
handle missing keypoints gracefully by returning None.

These utilities form the foundation for feature extraction and pose classification.
"""

import math
from typing import Optional, Tuple, List

# Type alias for a 2D point (x, y)
Point2D = Tuple[float, float]

# Type alias for a keypoint with confidence (x, y, confidence)
Keypoint = Tuple[float, float, float]


def angle_from_horizontal(p1: Point2D, p2: Point2D) -> float:
    """
    Calculate the angle of the line from p1 to p2 relative to horizontal.
    
    Returns angle in degrees [0, 90] where:
    - 0° = perfectly horizontal (lying down)
    - 90° = perfectly vertical (standing)
    
    Args:
        p1: First point (x, y) - typically upper body point
        p2: Second point (x, y) - typically lower body point
    
    Returns:
        Angle in degrees from horizontal (always positive)
    
    Example:
        >>> angle_from_horizontal((0, 0), (10, 0))  # horizontal
        0.0
        >>> angle_from_horizontal((0, 0), (0, 10))  # vertical
        90.0
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    # atan2 returns angle from positive x-axis
    # We want angle from horizontal, so we take absolute value
    angle_rad = math.atan2(abs(dy), abs(dx))
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg


def angle_between_vectors(
    p1: Point2D, 
    p_vertex: Point2D, 
    p2: Point2D
) -> float:
    """
    Calculate the angle at p_vertex formed by vectors (p_vertex -> p1) and (p_vertex -> p2).
    
    Useful for limb angles (e.g., elbow angle, knee angle).
    
    Args:
        p1: First endpoint
        p_vertex: Vertex point (where angle is measured)
        p2: Second endpoint
    
    Returns:
        Angle in degrees [0, 180]
    """
    # Vectors from vertex to endpoints
    v1 = (p1[0] - p_vertex[0], p1[1] - p_vertex[1])
    v2 = (p2[0] - p_vertex[0], p2[1] - p_vertex[1])
    
    # Magnitudes
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 0.0  # Degenerate case
    
    # Dot product
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    
    # Clamp to valid range for acos
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    
    return math.degrees(math.acos(cos_angle))


def distance(p1: Point2D, p2: Point2D) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
    
    Returns:
        Distance in pixels
    """
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def midpoint(p1: Point2D, p2: Point2D) -> Point2D:
    """
    Calculate the midpoint between two points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
    
    Returns:
        Midpoint (x, y)
    """
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def midpoint_if_valid(
    kp1: Optional[Keypoint], 
    kp2: Optional[Keypoint],
    min_confidence: float = 0.5
) -> Optional[Point2D]:
    """
    Calculate midpoint only if both keypoints are valid (above confidence threshold).
    
    This is crucial for handling partial body visibility - we don't want to
    compute midpoints with unreliable keypoints.
    
    Args:
        kp1: First keypoint (x, y, confidence) or None
        kp2: Second keypoint (x, y, confidence) or None
        min_confidence: Minimum confidence threshold
    
    Returns:
        Midpoint (x, y) if both keypoints are valid, None otherwise
    """
    if kp1 is None or kp2 is None:
        return None
    
    if kp1[2] < min_confidence or kp2[2] < min_confidence:
        return None
    
    return midpoint((kp1[0], kp1[1]), (kp2[0], kp2[1]))


def normalize_by_bbox(
    value: float, 
    bbox_size: float,
    epsilon: float = 1e-6
) -> float:
    """
    Normalize a value by bounding box size.
    
    Used for jitter calculation and distance normalization to make
    measurements scale-invariant.
    
    Args:
        value: Value to normalize (e.g., movement in pixels)
        bbox_size: Reference size (e.g., bbox diagonal or height)
        epsilon: Small value to prevent division by zero
    
    Returns:
        Normalized value (dimensionless ratio)
    """
    return value / max(bbox_size, epsilon)


def bbox_diagonal(bbox: Tuple[float, float, float, float]) -> float:
    """
    Calculate the diagonal length of a bounding box.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
    
    Returns:
        Diagonal length in pixels
    """
    x1, y1, x2, y2 = bbox
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def bbox_dimensions(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Get width and height of a bounding box.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
    
    Returns:
        (width, height)
    """
    x1, y1, x2, y2 = bbox
    return (abs(x2 - x1), abs(y2 - y1))


def bbox_aspect_ratio(bbox: Tuple[float, float, float, float]) -> float:
    """
    Calculate aspect ratio of bounding box (width / height).
    
    A high aspect ratio (> 1.5) suggests horizontal orientation (lying).
    A low aspect ratio (< 0.7) suggests vertical orientation (standing).
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
    
    Returns:
        Aspect ratio (width / height)
    """
    width, height = bbox_dimensions(bbox)
    if height < 1e-6:
        return 0.0
    return width / height


def bbox_center(bbox: Tuple[float, float, float, float]) -> Point2D:
    """
    Calculate the center point of a bounding box.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
    
    Returns:
        Center point (x, y)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    """
    Calculate the area of a bounding box.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
    
    Returns:
        Area in square pixels
    """
    width, height = bbox_dimensions(bbox)
    return width * height


def relative_position_in_frame(
    point: Point2D, 
    frame_shape: Tuple[int, int]
) -> Tuple[float, float]:
    """
    Calculate relative position of a point within the frame.
    
    Args:
        point: Point (x, y) in pixels
        frame_shape: Frame dimensions (height, width)
    
    Returns:
        (relative_x, relative_y) where 0.0 = left/top, 1.0 = right/bottom
    """
    height, width = frame_shape
    rel_x = point[0] / max(width, 1)
    rel_y = point[1] / max(height, 1)
    return (rel_x, rel_y)


def compute_skeleton_spread(
    keypoints: List[Optional[Keypoint]],
    min_confidence: float = 0.5
) -> Optional[float]:
    """
    Compute the spread of visible keypoints (bounding box of valid keypoints).
    
    Used to calculate "compactness" - a compact skeleton might indicate
    fetal position or curled-up posture.
    
    Args:
        keypoints: List of keypoints [(x, y, conf), ...] or None
        min_confidence: Minimum confidence for valid keypoints
    
    Returns:
        Diagonal of the keypoint bounding box, or None if < 2 valid keypoints
    """
    valid_points = [
        (kp[0], kp[1]) 
        for kp in keypoints 
        if kp is not None and kp[2] >= min_confidence
    ]
    
    if len(valid_points) < 2:
        return None
    
    xs = [p[0] for p in valid_points]
    ys = [p[1] for p in valid_points]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    return math.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)


def compute_jitter(
    current_point: Point2D,
    previous_point: Point2D,
    bbox_size: float
) -> float:
    """
    Compute normalized jitter (movement) between frames.
    
    High jitter can indicate unreliable detection or actual movement.
    Normalizing by bbox size makes this scale-invariant.
    
    Args:
        current_point: Current position (x, y)
        previous_point: Previous position (x, y)
        bbox_size: Reference size for normalization
    
    Returns:
        Normalized jitter [0, inf) where 0 = no movement
    """
    movement = distance(current_point, previous_point)
    return normalize_by_bbox(movement, bbox_size)


def point_is_valid(kp: Optional[Keypoint], min_confidence: float = 0.5) -> bool:
    """
    Check if a keypoint is valid (exists and has sufficient confidence).
    
    Args:
        kp: Keypoint (x, y, confidence) or None
        min_confidence: Minimum confidence threshold
    
    Returns:
        True if keypoint is valid
    """
    return kp is not None and kp[2] >= min_confidence
