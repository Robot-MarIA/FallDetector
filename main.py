"""
Pose Risk Detection System - Main Entry Point

This is the main orchestrator that connects all components:
- Frame source (webcam/video)
- Pose estimation (YOLO-Pose)
- Quality assessment
- Feature extraction
- Classification
- Temporal confirmation
- Adaptive scheduling
- Visualization and logging

Usage:
    python main.py --source webcam --show
    python main.py --source video --path video.mp4 --output logs/
    python main.py --help

Designed for PC development with architecture ready for ROS2/Jetson migration.
"""

import argparse
import time
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import components
from core.frame_source import OpenCVFrameSource, FrameData
from core.inference_backend import create_inference_backend
from core.pose_estimator import PoseEstimator
from core.quality import QualityAssessor
from core.features import FeatureExtractor
from core.classifier import PoseClassifier, RiskState
from core.temporal import TemporalAnalyzer
from core.scheduler import AdaptiveScheduler, SchedulerMode
from core.outputs import ConsolePublisher, SystemState
from utils.dashboard import DashboardVisualizer
from utils.logging import ExplainableLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pose Risk Detection System - Detects risky postures from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --source webcam --show
  python main.py --source video --path fall_video.mp4
  python main.py --source webcam --output logs/ --verbose
        """
    )
    
    # Input source
    parser.add_argument(
        '--source', 
        type=str, 
        default='webcam',
        choices=['webcam', 'video'],
        help='Input source type (default: webcam)'
    )
    parser.add_argument(
        '--path', 
        type=str, 
        default=None,
        help='Path to video file (required if source=video)'
    )
    parser.add_argument(
        '--camera', 
        type=int, 
        default=0,
        help='Camera index for webcam (default: 0)'
    )
    
    # Model
    parser.add_argument(
        '--model', 
        type=str, 
        default='yolo11n-pose.pt',
        help='Path to YOLO-Pose model (default: yolo11n-pose.pt)'
    )
    
    # Configuration
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help='Path to config directory (default: config/)'
    )
    
    # Output
    parser.add_argument(
        '--output', 
        type=str, 
        default='logs',
        help='Output directory for logs (default: logs/)'
    )
    parser.add_argument(
        '--show', 
        action='store_true',
        help='Show visualization window'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose console output'
    )
    parser.add_argument(
        '--no-log', 
        action='store_true',
        help='Disable file logging'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.source == 'video' and args.path is None:
        parser.error("--path is required when source=video")
    
    return args


def main():
    """Main entry point."""
    args = parse_args()
    
    # Determine config path
    config_dir = Path(args.config) if args.config else PROJECT_ROOT / 'config'
    thresholds_path = config_dir / 'thresholds.yaml'
    scheduler_path = config_dir / 'scheduler.yaml'
    
    print("=" * 60)
    print("Pose Risk Detection System")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"Model: {args.model}")
    print(f"Config: {config_dir}")
    print("=" * 60)
    
    # Initialize components
    print("\nInitializing components...")
    
    # 1. Frame source
    if args.source == 'webcam':
        source = OpenCVFrameSource(args.camera)
    else:
        source = OpenCVFrameSource(args.path)
    
    # 2. Inference backend
    backend = create_inference_backend('ultralytics', model_path=args.model)
    backend.warmup()
    
    # 3. Pose estimator
    estimator = PoseEstimator(
        backend=backend,
        config_path=str(thresholds_path) if thresholds_path.exists() else None
    )
    
    # 4. Analysis pipeline
    quality_assessor = QualityAssessor(
        config_path=str(thresholds_path) if thresholds_path.exists() else None
    )
    feature_extractor = FeatureExtractor(
        config_path=str(thresholds_path) if thresholds_path.exists() else None
    )
    classifier = PoseClassifier(
        config_path=str(thresholds_path) if thresholds_path.exists() else None
    )
    
    # 5. Temporal and scheduler
    temporal = TemporalAnalyzer(
        config_path=str(thresholds_path) if thresholds_path.exists() else None
    )
    scheduler = AdaptiveScheduler(
        config_path=str(scheduler_path) if scheduler_path.exists() else None
    )
    
    # 6. Output and visualization
    publisher = ConsolePublisher(verbose=args.verbose)
    visualizer = DashboardVisualizer() if args.show else None
    
    # 7. Logger
    if not args.no_log:
        logger = ExplainableLogger(output_dir=args.output)
    else:
        logger = None
    
    # OpenCV window
    if args.show:
        import cv2
        cv2.namedWindow('Fall Detection - YOLO11 Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Fall Detection - YOLO11 Pose', 1280, 720)
    
    print("\nStarting detection loop... (Press 'q' to quit)")
    print("-" * 60)
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0.0
    
    try:
        for frame_data in source.frames():
            frame_count += 1
            
            # Apply scheduler's inference skip
            settings = scheduler.update(
                risk_score=0.0,  # Will be updated below
                quality_score=0.5,
                confirmed_state=RiskState.OK,
                current_state=RiskState.OK,
                is_confirmed=False,
            )
            
            if frame_count % settings.inference_skip != 0:
                # Skip inference this frame
                if args.show:
                    import cv2
                    cv2.imshow('Fall Detection - YOLO11 Pose', frame_data.frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue
            
            # Run pose estimation
            pose = estimator.estimate(
                frame_data.frame,
                frame_shape=(frame_data.height, frame_data.width)
            )
            
            if pose is None:
                # No person detected
                if args.show:
                    import cv2
                    h, w = frame_data.frame.shape[:2]
                    overlay = cv2.resize(frame_data.frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
                    # Draw semi-transparent panel
                    panel = overlay.copy()
                    cv2.rectangle(panel, (0, 0), (1280, 70), (30, 30, 35), -1)
                    cv2.addWeighted(panel, 0.85, overlay, 0.15, 0, overlay)
                    cv2.putText(overlay, "NO PERSON DETECTED", (480, 45),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 180, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Fall Detection - YOLO11 Pose', overlay)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue
            
            # Quality assessment
            quality = quality_assessor.assess(
                keypoints=pose.keypoints,
                bbox=pose.bbox,
                frame_shape=(frame_data.height, frame_data.width)
            )
            
            # Feature extraction
            features = feature_extractor.extract(
                keypoints=pose.keypoints,
                bbox=pose.bbox,
                frame_shape=(frame_data.height, frame_data.width)
            )
            
            # Check for extreme geometry (quality override)
            has_extreme_geo = feature_extractor.is_extreme_geometry(features)
            
            # Classification
            classification = classifier.classify(
                features=features,
                quality=quality,
                has_extreme_geometry=has_extreme_geo
            )
            
            # Temporal confirmation
            bbox_center = pose.get_bbox_center()
            temporal_result = temporal.update(
                classification=classification,
                quality_score=quality.score,
                bbox_center=bbox_center,
                timestamp=frame_data.timestamp
            )
            
            # Update scheduler with real values
            settings = scheduler.update(
                risk_score=classification.risk_score,
                quality_score=quality.score,
                confirmed_state=temporal_result.confirmed_state,
                current_state=temporal_result.current_state,
                is_confirmed=temporal_result.is_confirmed,
                timestamp=frame_data.timestamp
            )
            
            # Publish state
            system_state = SystemState(
                timestamp=frame_data.timestamp,
                risk_state=temporal_result.current_state,
                confirmed_state=temporal_result.confirmed_state,
                risk_score=classification.risk_score,
                quality_score=quality.score,
                is_confirmed=temporal_result.is_confirmed,
                scheduler_mode=settings.mode,
                reason=classification.reason,
                internal_pose=classification.internal_pose.value,
                torso_angle=features.torso_angle,
                n_persons=pose.n_persons_detected,
            )
            publisher.publish(system_state)
            
            # Log
            if logger:
                logger.log(
                    timestamp=frame_data.timestamp,
                    frame_number=frame_count,
                    risk_state=temporal_result.current_state.value,
                    confirmed_state=temporal_result.confirmed_state.value,
                    internal_pose=classification.internal_pose.value,
                    risk_score=classification.risk_score,
                    quality_score=quality.score,
                    is_confirmed=temporal_result.is_confirmed,
                    torso_angle=features.torso_angle,
                    hip_height_ratio=features.hip_height_ratio,
                    shoulder_height_ratio=features.shoulder_height_ratio,
                    bbox_aspect_ratio=features.bbox_aspect_ratio,
                    compactness=features.compactness,
                    bbox_bottom_ratio=features.bbox_bottom_ratio,
                    n_valid_keypoints=features.n_valid_keypoints,
                    scheduler_mode=settings.mode.value,
                    target_fps=settings.target_fps,
                    reason=classification.reason,
                    temporal_reason=temporal_result.reason,
                    n_persons_detected=pose.n_persons_detected,
                    selection_reason=pose.selection_reason,
                )
            
            # Visualization with new Dashboard UI
            if args.show and visualizer:
                import cv2
                overlay = visualizer.draw(
                    frame=frame_data.frame,
                    keypoints=pose.keypoints,
                    state=temporal_result.confirmed_state,
                    risk_score=classification.risk_score,
                    quality_score=quality.score,
                    is_confirmed=temporal_result.is_confirmed,
                    internal_pose=classification.internal_pose.value,
                    scheduler_mode=settings.mode,
                )
                
                cv2.imshow('Fall Detection - YOLO11 Pose', overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        print("\n" + "-" * 60)
        print("Cleaning up...")
        
        source.release()
        publisher.close()
        
        if logger:
            logger.close()
            summary = logger.get_summary()
            print(f"\nSession summary:")
            print(f"  Total frames: {summary.get('n_entries', 0)}")
            if 'state_counts' in summary:
                print(f"  States: {summary['state_counts']}")
        
        if args.show:
            import cv2
            cv2.destroyAllWindows()
        
        print("\nDone!")


if __name__ == '__main__':
    main()
