"""
Unit tests for quality assessment.

These tests verify the quality scoring logic, especially:
- Torso missing penalty
- Keypoint count requirements
- Quality gates for NEEDS_HELP confirmation

Run with: pytest tests/test_quality.py -v
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.quality import QualityAssessor, QualityResult


def create_keypoints(
    valid_indices: list,
    confidence: float = 0.9,
    total: int = 17
) -> list:
    """
    Helper to create keypoint list with specified valid indices.
    
    Args:
        valid_indices: Indices that should be valid (high confidence)
        confidence: Confidence for valid keypoints
        total: Total number of keypoints
    
    Returns:
        List of (x, y, confidence) tuples
    """
    keypoints = []
    for i in range(total):
        if i in valid_indices:
            keypoints.append((100.0 + i * 10, 100.0 + i * 10, confidence))
        else:
            keypoints.append((0.0, 0.0, 0.0))  # Invalid
    return keypoints


class TestQualityAssessor:
    """Tests for QualityAssessor class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.assessor = QualityAssessor()
    
    def test_full_visibility_high_quality(self):
        """All keypoints visible should give high quality."""
        # All 17 keypoints valid
        keypoints = create_keypoints(list(range(17)), confidence=0.9)
        bbox = (50, 50, 250, 450)
        
        result = self.assessor.assess(keypoints, bbox)
        
        assert result.score >= 0.8, f"Expected high quality, got {result.score}"
        assert result.has_torso is True
        assert result.n_valid_keypoints == 17
    
    def test_missing_torso_severe_penalty(self):
        """Missing shoulders and hips should severely penalize quality."""
        # Only head keypoints (0-4), no shoulders (5,6) or hips (11,12)
        keypoints = create_keypoints([0, 1, 2, 3, 4], confidence=0.9)
        bbox = (50, 50, 250, 450)
        
        result = self.assessor.assess(keypoints, bbox)
        
        # Quality should be very low (< 0.3 per design)
        assert result.score < 0.3, f"Expected low quality without torso, got {result.score}"
        assert result.has_torso is False
        assert "NO_TORSO" in result.reasons
    
    def test_only_one_shoulder_and_hip_has_torso(self):
        """Single shoulder + single hip should count as having torso."""
        # Left shoulder (5) and left hip (11) only
        keypoints = create_keypoints([5, 11], confidence=0.9)
        bbox = (50, 50, 250, 450)
        
        result = self.assessor.assess(keypoints, bbox)
        
        assert result.has_torso is True
    
    def test_low_keypoint_count_penalty(self):
        """Few visible keypoints should reduce quality."""
        # Only 3 keypoints (below default min of 6)
        keypoints = create_keypoints([5, 6, 11], confidence=0.9)  # Both shoulders + hip
        bbox = (50, 50, 250, 450)
        
        result = self.assessor.assess(keypoints, bbox)
        
        # Should have penalty but still have torso
        assert result.has_torso is True
        assert "LOW_KEYPOINTS" in result.reasons


class TestQualityGates:
    """Tests for quality gates in NEEDS_HELP confirmation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.assessor = QualityAssessor()
    
    def test_sufficient_quality_can_confirm(self):
        """Quality >= 0.4 should allow confirmation."""
        keypoints = create_keypoints(list(range(17)), confidence=0.9)
        bbox = (50, 50, 250, 450)
        
        result = self.assessor.assess(keypoints, bbox)
        
        can_confirm, reason = self.assessor.can_confirm_needs_help(
            result, 
            has_extreme_geometry=False
        )
        
        assert can_confirm is True
        assert "SUFFICIENT" in reason
    
    def test_low_quality_blocks_confirmation(self):
        """Quality < 0.4 should block confirmation without extreme geometry."""
        # Create low quality detection (no torso)
        keypoints = create_keypoints([0, 1, 2], confidence=0.9)
        bbox = (50, 50, 250, 450)
        
        result = self.assessor.assess(keypoints, bbox)
        
        can_confirm, reason = self.assessor.can_confirm_needs_help(
            result,
            has_extreme_geometry=False
        )
        
        assert can_confirm is False
    
    def test_extreme_geometry_overrides_low_quality(self):
        """Extreme geometry should allow confirmation with lower quality."""
        # Create medium-low quality detection
        keypoints = create_keypoints([5, 6, 11, 12], confidence=0.7)  # Torso only
        bbox = (50, 50, 250, 450)
        
        result = self.assessor.assess(keypoints, bbox)
        
        # Even with lower quality, extreme geometry should override
        # (but quality must still be >= 0.25)
        if result.score >= 0.25:
            can_confirm, reason = self.assessor.can_confirm_needs_help(
                result,
                has_extreme_geometry=True  # Simulate very clear lying position
            )
            
            assert can_confirm is True
            assert "OVERRIDE" in reason


class TestJitterCalculation:
    """Tests for jitter-based quality penalties."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.assessor = QualityAssessor()
    
    def test_no_jitter_first_frame(self):
        """First frame should have zero jitter."""
        keypoints = create_keypoints(list(range(17)), confidence=0.9)
        bbox = (50, 50, 250, 450)
        
        result = self.assessor.assess(keypoints, bbox)
        
        assert result.jitter == 0.0
    
    def test_stable_detection_low_jitter(self):
        """Stable detections should have low jitter."""
        keypoints = create_keypoints(list(range(17)), confidence=0.9)
        bbox = (50, 50, 250, 450)
        
        # First frame
        self.assessor.assess(keypoints, bbox)
        
        # Second frame - slight movement
        keypoints_moved = [
            (kp[0] + 2, kp[1] + 2, kp[2]) if kp[2] > 0 else kp
            for kp in keypoints
        ]
        result = self.assessor.assess(keypoints_moved, bbox)
        
        # Small movement should result in low jitter
        assert result.jitter < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
