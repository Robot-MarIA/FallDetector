"""
Unit tests for temporal confirmation.

These tests verify the adaptive temporal confirmation logic:
- Confirmation after sufficient persistence
- Adaptive window calculation
- NEEDS_HELP only confirmed when allowed by quality

Run with: pytest tests/test_temporal.py -v
"""

import pytest
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.temporal import TemporalAnalyzer, TemporalResult
from core.classifier import RiskState, ClassificationResult, InternalPose


def create_classification(
    state: RiskState,
    risk_score: float = 0.5,
    can_confirm: bool = True
) -> ClassificationResult:
    """Helper to create ClassificationResult."""
    return ClassificationResult(
        state=state,
        internal_pose=InternalPose.NORMAL,
        risk_score=risk_score,
        confidence=0.8,
        reason="TEST",
        can_confirm=can_confirm,
    )


class TestTemporalConfirmation:
    """Tests for temporal confirmation behavior."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = TemporalAnalyzer()
    
    def test_initial_state_not_confirmed(self):
        """First classification should not be immediately confirmed."""
        classification = create_classification(RiskState.NEEDS_HELP, risk_score=0.9)
        
        result = self.analyzer.update(
            classification=classification,
            quality_score=0.9,
            timestamp=time.time()
        )
        
        # Should not be confirmed yet
        assert result.is_confirmed is False
        assert result.confirmation_progress < 1.0
    
    def test_confirmation_after_persistence(self):
        """State should be confirmed after sufficient time."""
        t = time.time()
        
        # Simulate persistent NEEDS_HELP for 5 seconds with high quality
        for i in range(50):  # 50 frames over ~5 seconds
            classification = create_classification(
                RiskState.NEEDS_HELP, 
                risk_score=0.9,
                can_confirm=True
            )
            
            result = self.analyzer.update(
                classification=classification,
                quality_score=0.9,
                bbox_center=(320, 240),
                timestamp=t + i * 0.1  # 10 FPS
            )
            
            if result.is_confirmed and result.confirmed_state == RiskState.NEEDS_HELP:
                # Confirmed - test passed
                assert True
                return
        
        # Should have confirmed within 5 seconds
        assert result.is_confirmed, "Should have confirmed after 5 seconds"
    
    def test_state_change_resets_progress(self):
        """Changing state should reset confirmation progress."""
        t = time.time()
        
        # Start with RISK
        for i in range(10):
            classification = create_classification(RiskState.RISK, risk_score=0.5)
            result = self.analyzer.update(
                classification=classification,
                quality_score=0.9,
                timestamp=t + i * 0.1
            )
        
        progress_before = result.confirmation_progress
        assert progress_before > 0
        
        # Change to OK
        classification = create_classification(RiskState.OK, risk_score=0.1)
        result = self.analyzer.update(
            classification=classification,
            quality_score=0.9,
            timestamp=t + 1.5  # 1.5 seconds later
        )
        
        # Progress should reset
        assert result.confirmation_progress < progress_before


class TestAdaptiveWindow:
    """Tests for adaptive confirmation window."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = TemporalAnalyzer()
    
    def test_high_quality_shorter_window(self):
        """High quality should result in shorter confirmation window."""
        classification = create_classification(RiskState.RISK, risk_score=0.7)
        
        # High quality update
        result_high = self.analyzer.update(
            classification=classification,
            quality_score=0.95,
            bbox_center=(320, 240),
            timestamp=time.time()
        )
        
        window_high = result_high.window_seconds
        
        # Reset and try with low quality
        self.analyzer.reset()
        
        result_low = self.analyzer.update(
            classification=classification,
            quality_score=0.4,
            bbox_center=(320, 240),
            timestamp=time.time()
        )
        
        window_low = result_low.window_seconds
        
        # Low quality should have longer window
        assert window_low > window_high, \
            f"Low quality window ({window_low}) should be > high quality ({window_high})"
    
    def test_window_within_bounds(self):
        """Window should always be within configured min/max."""
        classification = create_classification(RiskState.RISK, risk_score=0.5)
        
        # Test with various quality levels
        for quality in [0.1, 0.5, 0.9, 1.0]:
            result = self.analyzer.update(
                classification=classification,
                quality_score=quality,
                timestamp=time.time()
            )
            
            # Default bounds are 1.0 to 5.0 seconds
            assert result.window_seconds >= 1.0, \
                f"Window {result.window_seconds} below minimum"
            assert result.window_seconds <= 5.0, \
                f"Window {result.window_seconds} above maximum"
            
            self.analyzer.reset()


class TestNeedsHelpConfirmation:
    """Tests for NEEDS_HELP confirmation rules."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = TemporalAnalyzer()
    
    def test_needs_help_blocked_by_quality(self):
        """NEEDS_HELP should not confirm if can_confirm is False."""
        t = time.time()
        
        # Simulate persistent NEEDS_HELP but with can_confirm=False
        for i in range(60):
            classification = create_classification(
                RiskState.NEEDS_HELP,
                risk_score=0.9,
                can_confirm=False  # Quality too low
            )
            
            result = self.analyzer.update(
                classification=classification,
                quality_score=0.3,  # Low quality
                timestamp=t + i * 0.1
            )
        
        # Should be confirmed, but downgraded to RISK (not NEEDS_HELP)
        if result.is_confirmed:
            assert result.confirmed_state != RiskState.NEEDS_HELP, \
                "NEEDS_HELP should not confirm when can_confirm=False"


class TestStabilityCalculation:
    """Tests for pose stability calculation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = TemporalAnalyzer()
    
    def test_stable_position_high_stability(self):
        """Stable bbox center should give high stability."""
        t = time.time()
        center = (320, 240)
        
        for i in range(15):
            classification = create_classification(RiskState.OK, risk_score=0.1)
            result = self.analyzer.update(
                classification=classification,
                quality_score=0.9,
                bbox_center=center,  # Same position
                timestamp=t + i * 0.1
            )
        
        # Stability should be high
        assert result.stability > 0.8, f"Expected high stability, got {result.stability}"
    
    def test_moving_position_lower_stability(self):
        """Moving bbox center should reduce stability."""
        t = time.time()
        
        for i in range(15):
            # Moving center
            center = (320 + i * 10, 240 + i * 10)
            
            classification = create_classification(RiskState.OK, risk_score=0.1)
            result = self.analyzer.update(
                classification=classification,
                quality_score=0.9,
                bbox_center=center,
                timestamp=t + i * 0.1
            )
        
        # Stability should be lower than perfectly stable
        assert result.stability < 0.8, f"Expected lower stability for movement, got {result.stability}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
