"""
Unit tests for geometry utilities.

These are sanity tests to verify basic geometric calculations.
Run with: pytest tests/test_geometry.py -v
"""

import pytest
import math
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.geometry import (
    angle_from_horizontal,
    angle_between_vectors,
    distance,
    midpoint,
    midpoint_if_valid,
    bbox_aspect_ratio,
    bbox_area,
    normalize_by_bbox,
    point_is_valid,
)


class TestAngleFromHorizontal:
    """Tests for angle_from_horizontal function."""
    
    def test_horizontal_line_is_zero_degrees(self):
        """A perfectly horizontal line should have 0° angle."""
        p1 = (0, 0)
        p2 = (100, 0)  # Horizontal to the right
        
        angle = angle_from_horizontal(p1, p2)
        
        assert abs(angle - 0.0) < 0.01, f"Expected 0°, got {angle}°"
    
    def test_vertical_line_is_90_degrees(self):
        """A perfectly vertical line should have 90° angle."""
        p1 = (0, 0)
        p2 = (0, 100)  # Vertical downward
        
        angle = angle_from_horizontal(p1, p2)
        
        assert abs(angle - 90.0) < 0.01, f"Expected 90°, got {angle}°"
    
    def test_45_degree_line(self):
        """A 45° diagonal should return 45°."""
        p1 = (0, 0)
        p2 = (100, 100)  # 45° diagonal
        
        angle = angle_from_horizontal(p1, p2)
        
        assert abs(angle - 45.0) < 0.01, f"Expected 45°, got {angle}°"
    
    def test_negative_direction_same_angle(self):
        """Direction shouldn't matter for angle magnitude."""
        p1 = (100, 0)
        p2 = (0, 0)  # Horizontal to the left
        
        angle = angle_from_horizontal(p1, p2)
        
        assert abs(angle - 0.0) < 0.01, f"Expected 0°, got {angle}°"
    
    def test_lying_person_torso_angle(self):
        """Simulate a lying person's torso (nearly horizontal)."""
        # Shoulders at (100, 200), hips at (300, 220) - slight incline
        shoulders = (100, 200)
        hips = (300, 220)
        
        angle = angle_from_horizontal(shoulders, hips)
        
        # Should be small angle (nearly horizontal = lying)
        assert angle < 15.0, f"Expected < 15° for lying, got {angle}°"


class TestDistance:
    """Tests for distance function."""
    
    def test_zero_distance(self):
        """Same point should have zero distance."""
        p = (50, 50)
        
        assert distance(p, p) == 0.0
    
    def test_horizontal_distance(self):
        """Horizontal distance calculation."""
        p1 = (0, 0)
        p2 = (100, 0)
        
        assert distance(p1, p2) == 100.0
    
    def test_pythagorean_distance(self):
        """3-4-5 right triangle."""
        p1 = (0, 0)
        p2 = (3, 4)
        
        assert distance(p1, p2) == 5.0


class TestMidpoint:
    """Tests for midpoint functions."""
    
    def test_simple_midpoint(self):
        """Basic midpoint calculation."""
        p1 = (0, 0)
        p2 = (100, 100)
        
        mid = midpoint(p1, p2)
        
        assert mid == (50, 50)
    
    def test_midpoint_if_valid_with_valid_keypoints(self):
        """Should return midpoint when both keypoints are valid."""
        kp1 = (0, 0, 0.9)  # High confidence
        kp2 = (100, 100, 0.8)  # High confidence
        
        mid = midpoint_if_valid(kp1, kp2, min_confidence=0.5)
        
        assert mid == (50, 50)
    
    def test_midpoint_if_valid_with_low_confidence(self):
        """Should return None when confidence is too low."""
        kp1 = (0, 0, 0.9)  # High confidence
        kp2 = (100, 100, 0.3)  # Low confidence
        
        mid = midpoint_if_valid(kp1, kp2, min_confidence=0.5)
        
        assert mid is None
    
    def test_midpoint_if_valid_with_none(self):
        """Should return None when keypoint is None."""
        kp1 = (0, 0, 0.9)
        kp2 = None
        
        mid = midpoint_if_valid(kp1, kp2)
        
        assert mid is None


class TestBboxFunctions:
    """Tests for bounding box functions."""
    
    def test_aspect_ratio_square(self):
        """Square bbox should have aspect ratio 1."""
        bbox = (0, 0, 100, 100)
        
        ratio = bbox_aspect_ratio(bbox)
        
        assert abs(ratio - 1.0) < 0.01
    
    def test_aspect_ratio_horizontal(self):
        """Wide bbox (lying person) should have ratio > 1."""
        bbox = (0, 0, 200, 100)  # 200 wide, 100 tall
        
        ratio = bbox_aspect_ratio(bbox)
        
        assert ratio == 2.0  # 200/100
    
    def test_aspect_ratio_vertical(self):
        """Tall bbox (standing person) should have ratio < 1."""
        bbox = (0, 0, 100, 200)  # 100 wide, 200 tall
        
        ratio = bbox_aspect_ratio(bbox)
        
        assert ratio == 0.5  # 100/200
    
    def test_bbox_area(self):
        """Basic area calculation."""
        bbox = (10, 20, 110, 120)  # 100x100
        
        area = bbox_area(bbox)
        
        assert area == 10000


class TestPointIsValid:
    """Tests for point_is_valid function."""
    
    def test_valid_point(self):
        """High confidence point is valid."""
        kp = (100, 200, 0.9)
        
        assert point_is_valid(kp, min_confidence=0.5) is True
    
    def test_invalid_point_low_confidence(self):
        """Low confidence point is invalid."""
        kp = (100, 200, 0.3)
        
        assert point_is_valid(kp, min_confidence=0.5) is False
    
    def test_none_point(self):
        """None is invalid."""
        assert point_is_valid(None) is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
