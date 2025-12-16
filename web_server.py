"""
Web Dashboard Server for Fall Detection System.

Video Sources: Intel RealSense D435i (RGB + Depth) or Webcam fallback.
Depth Features: RANSAC floor plane, height-above-floor, fall detection.

Endpoints:
- GET / : HTML dashboard
- GET /video : MJPEG RGB stream
- GET /depth : MJPEG Depth stream (colorized)
- WS /ws : Real-time data WebSocket
"""

import asyncio
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Generator
from dataclasses import dataclass

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

# RealSense
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    rs = None

import sys
sys.path.insert(0, str(Path(__file__).parent))

from core.inference_backend import create_inference_backend
from core.pose_estimator import PoseEstimator
from core.quality import QualityAssessor
from core.features import FeatureExtractor
from core.classifier import PoseClassifier, RiskState
from core.temporal import TemporalAnalyzer
from core.depth_processor import DepthProcessor, DepthFeatures


# ============================================================================
# FRAME DATA
# ============================================================================

@dataclass
class FrameData:
    frame: np.ndarray  # RGB/BGR
    depth_raw: Optional[np.ndarray]  # Depth in mm (uint16)
    depth_colorized: Optional[np.ndarray]  # Depth colorized (BGR)
    width: int
    height: int
    timestamp: float


# ============================================================================
# REALSENSE SOURCE (RGB + DEPTH at correct resolutions)
# ============================================================================

class RealSenseSource:
    """
    Intel RealSense D435i: 
    - Color: 1280x720 @30fps BGR
    - Depth: 848x480 @30fps z16 (aligned to color)
    """
    
    def __init__(self):
        self.pipeline = None
        self.align = None
        self.colorizer = None
        self.intrinsics = None
        self._running = False
        self.color_w, self.color_h = 1280, 720
        self.depth_w, self.depth_h = 848, 480
        
    def start(self) -> bool:
        if not REALSENSE_AVAILABLE:
            print("[RealSense] pyrealsense2 not available")
            return False
        
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Color at 1280x720
            config.enable_stream(rs.stream.color, self.color_w, self.color_h, rs.format.bgr8, 30)
            # Depth at 848x480 (better quality than 1280x720 for D435i)
            config.enable_stream(rs.stream.depth, self.depth_w, self.depth_h, rs.format.z16, 30)
            
            profile = self.pipeline.start(config)
            
            # Align depth to color
            self.align = rs.align(rs.stream.color)
            
            # Colorizer
            self.colorizer = rs.colorizer()
            self.colorizer.set_option(rs.option.color_scheme, 0)  # Jet
            
            # Get depth intrinsics (after alignment, use color intrinsics)
            color_profile = profile.get_stream(rs.stream.color)
            self.intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            
            device = profile.get_device()
            name = device.get_info(rs.camera_info.name)
            print(f"[RealSense] Connected: {name}")
            print(f"[RealSense] Color: {self.color_w}x{self.color_h} | Depth: {self.depth_w}x{self.depth_h}")
            
            self._running = True
            return True
            
        except Exception as e:
            print(f"[RealSense] Failed: {e}")
            return False
    
    def get_frame(self) -> Optional[FrameData]:
        if not self._running or not self.pipeline:
            return None
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned = self.align.process(frames)
            
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            
            if not color_frame:
                return None
            
            color = np.asanyarray(color_frame.get_data())
            
            depth_raw = None
            depth_colorized = None
            if depth_frame:
                depth_raw = np.asanyarray(depth_frame.get_data())
                depth_colorized = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
            
            return FrameData(
                frame=color,
                depth_raw=depth_raw,
                depth_colorized=depth_colorized,
                width=self.color_w,
                height=self.color_h,
                timestamp=time.time()
            )
        except Exception as e:
            print(f"[RealSense] Frame error: {e}")
            return None
    
    def frames(self) -> Generator[FrameData, None, None]:
        while self._running:
            fd = self.get_frame()
            if fd:
                yield fd
    
    def stop(self):
        self._running = False
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
        print("[RealSense] Stopped")


class WebcamSource:
    """Fallback webcam (no depth)."""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self._running = False
        self.width, self.height = 1280, 720
    
    def start(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[Webcam] {self.width}x{self.height}")
            self._running = True
            return True
        except:
            return False
    
    def get_frame(self) -> Optional[FrameData]:
        if not self._running or not self.cap:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return FrameData(
            frame=frame, 
            depth_raw=None, 
            depth_colorized=None,
            width=self.width, 
            height=self.height, 
            timestamp=time.time()
        )
    
    def frames(self) -> Generator[FrameData, None, None]:
        while self._running:
            fd = self.get_frame()
            if fd:
                yield fd
    
    def stop(self):
        self._running = False
        if self.cap:
            self.cap.release()
        print("[Webcam] Stopped")


def create_video_source(use_realsense: bool = True, camera_index: int = 0):
    if use_realsense:
        src = RealSenseSource()
        if src.start():
            return src
        print("[Source] RealSense failed, trying webcam...")
    
    src = WebcamSource(camera_index)
    if src.start():
        return src
    
    raise RuntimeError("No video source")


# ============================================================================
# GLOBAL STATE
# ============================================================================

class DetectionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.depth: Optional[np.ndarray] = None
        self.frame_width: int = 1280
        self.frame_height: int = 720
        self.keypoints: List[Dict] = []
        self.state: str = "ANALYZING"
        self.position: str = "unknown"
        self.confidence: float = 0.0
        self.risk_score: float = 0.0
        self.quality_score: float = 0.0
        self.fps: float = 0.0
        self.log: List[Dict] = []
        self.is_confirmed: bool = False
        
        # Depth features for debug
        self.hip_height_m: Optional[float] = None
        self.floor_quality: float = 0.0
        self.depth_mode: str = "none"
        
    def update(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)
    
    def get_json(self) -> Dict:
        with self.lock:
            return {
                "state": self.state,
                "position": self.position,
                "confidence": round(self.confidence, 2),
                "risk_score": round(self.risk_score, 2),
                "quality_score": round(self.quality_score, 2),
                "fps": round(self.fps, 1),
                "frame_width": self.frame_width,
                "frame_height": self.frame_height,
                "keypoints": self.keypoints[:],
                "log": self.log[:8],
                # Depth debug
                "hip_height_m": round(self.hip_height_m, 3) if self.hip_height_m else None,
                "floor_quality": round(self.floor_quality, 2),
                "depth_mode": self.depth_mode,
            }
    
    def get_rgb_jpeg(self) -> Optional[bytes]:
        with self.lock:
            if self.frame is None:
                return None
            _, jpg = cv2.imencode('.jpg', self.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return jpg.tobytes()
    
    def get_depth_jpeg(self) -> Optional[bytes]:
        with self.lock:
            if self.depth is None:
                return None
            _, jpg = cv2.imencode('.jpg', self.depth, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return jpg.tobytes()


STATE = DetectionState()


# ============================================================================
# DETECTION LOOP WITH DEPTH
# ============================================================================

def map_state_with_depth(
    risk_state: RiskState,
    is_confirmed: bool,
    risk_score: float,
    depth_features: DepthFeatures,
    config: Dict
) -> tuple:
    """
    Map state using depth evidence.
    
    Returns: (state_str, is_confirmed, risk_score, reason)
    """
    hip_thresh = config.get('hip_floor_thresh_m', 0.30)
    persistence_thresh = config.get('persistence_confirm_s', 1.0)
    drop_thresh = config.get('drop_thresh_m', 0.35)
    vel_thresh = config.get('vel_thresh_mps', 0.60)
    min_floor_quality = config.get('min_floor_quality', 0.55)
    min_depth_ratio = config.get('min_depth_valid_ratio', 0.50)
    sofa_min = config.get('sofa_min_height_m', 0.35)
    sofa_max = config.get('sofa_max_height_m', 0.80)
    
    reason_parts = []
    
    # Check if depth is reliable
    if depth_features.depth_mode == "none" or depth_features.floor_quality < min_floor_quality:
        # Fallback: use 2D only, but cannot confirm RED
        if risk_state == RiskState.NEEDS_HELP:
            return ("ANALYZING", False, risk_score, "DEPTH_UNRELIABLE_FALLBACK")
        elif risk_state == RiskState.OK:
            return ("OK", is_confirmed, risk_score, "2D_OK")
        else:
            return ("ANALYZING", False, risk_score, "DEPTH_UNRELIABLE")
    
    # Depth is reliable
    hip_h = depth_features.hip_height_m
    
    # Check if on elevated surface (sofa/bed = OK)
    if hip_h is not None and sofa_min <= hip_h <= sofa_max:
        if depth_features.vertical_drop_m < drop_thresh:
            # On sofa/bed, no fall event
            reason_parts.append("ON_ELEVATED_SURFACE")
            return ("OK", True, 0.15, " + ".join(reason_parts))
    
    # Check for floor contact
    on_floor = False
    if hip_h is not None and hip_h < hip_thresh:
        if depth_features.floor_contact_time_s >= persistence_thresh:
            on_floor = True
            reason_parts.append("FLOOR_CONTACT_PERSISTENT")
    
    # Check for drop event
    had_drop = False
    if depth_features.vertical_drop_m > drop_thresh:
        reason_parts.append(f"DROP_EVENT({depth_features.vertical_drop_m:.2f}m)")
        had_drop = True
    if depth_features.vertical_velocity_mps > vel_thresh:
        reason_parts.append(f"FAST_DESCENT({depth_features.vertical_velocity_mps:.2f}m/s)")
        had_drop = True
    
    # Decision
    if on_floor:
        if had_drop:
            reason_parts.append("FALL_CONFIRMED")
            return ("FALL", True, 0.95, " + ".join(reason_parts))
        else:
            reason_parts.append("FLOOR_POSTURE_NO_DROP")
            return ("FALL", True, 0.85, " + ".join(reason_parts))
    
    # Not on floor
    if risk_state == RiskState.OK:
        return ("OK", True, risk_score, "DEPTH_OK")
    
    return ("ANALYZING", False, risk_score, "ANALYZING")


def detection_loop(use_realsense: bool = True, camera_index: int = 0):
    """Detection loop with depth integration."""
    global STATE
    
    source = create_video_source(use_realsense, camera_index)
    
    # Config
    config_dir = Path(__file__).parent / 'config'
    thresholds_path = config_dir / 'thresholds.yaml'
    
    # Load depth config
    depth_config = {}
    if thresholds_path.exists():
        import yaml
        with open(thresholds_path) as f:
            cfg = yaml.safe_load(f)
            if 'depth' in cfg:
                depth_config = cfg['depth']
    
    # Initialize depth processor
    depth_processor = DepthProcessor(
        camera_height_m=0.63,
        sample_window=depth_config.get('depth_sample_win', 7),
        min_floor_quality=depth_config.get('min_floor_quality', 0.55),
        temporal_window_s=depth_config.get('temporal_window_s', 1.2),
    )
    
    # Set intrinsics if RealSense
    if isinstance(source, RealSenseSource) and source.intrinsics:
        depth_processor.set_intrinsics(source.intrinsics)
    
    # Detection components
    backend = create_inference_backend('ultralytics', model_path='yolo11n-pose.pt')
    backend.warmup()
    
    estimator = PoseEstimator(backend=backend, config_path=str(thresholds_path) if thresholds_path.exists() else None)
    quality_assessor = QualityAssessor(config_path=str(thresholds_path) if thresholds_path.exists() else None)
    feature_extractor = FeatureExtractor(config_path=str(thresholds_path) if thresholds_path.exists() else None)
    classifier = PoseClassifier(config_path=str(thresholds_path) if thresholds_path.exists() else None)
    temporal = TemporalAnalyzer(config_path=str(thresholds_path) if thresholds_path.exists() else None)
    
    log_buffer = []
    last_state = None
    fps_history = []
    last_time = time.time()
    
    print("[Detection] Loop started with depth processing")
    
    try:
        for fd in source.frames():
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps_history.append(1.0 / dt)
                if len(fps_history) > 30:
                    fps_history.pop(0)
            fps = sum(fps_history) / max(len(fps_history), 1)
            
            STATE.update(
                frame=fd.frame.copy(),
                depth=fd.depth_colorized.copy() if fd.depth_colorized is not None else None,
                frame_width=fd.width,
                frame_height=fd.height,
                fps=fps
            )
            
            # Pose estimation
            pose = estimator.estimate(fd.frame, frame_shape=(fd.height, fd.width))
            
            if pose is None:
                STATE.update(
                    keypoints=[], 
                    state="ANALYZING", 
                    position="unknown",
                    hip_height_m=None,
                    floor_quality=0.0,
                    depth_mode="none"
                )
                continue
            
            # Keypoints JSON
            kp_json = [
                {"x": round(k[0], 1), "y": round(k[1], 1), "c": round(k[2], 2)} if k else {"x": 0, "y": 0, "c": 0}
                for k in pose.keypoints
            ]
            
            # Depth features
            depth_features = DepthFeatures()
            if fd.depth_raw is not None:
                depth_features = depth_processor.process(fd.depth_raw, pose.keypoints, fd.timestamp)
            
            # 2D analysis
            quality = quality_assessor.assess(pose.keypoints, pose.bbox, (fd.height, fd.width))
            features = feature_extractor.extract(pose.keypoints, pose.bbox, (fd.height, fd.width))
            classification = classifier.classify(features, quality, feature_extractor.is_extreme_geometry(features))
            
            bbox_center = pose.get_bbox_center()
            temporal_result = temporal.update(classification, quality.score, bbox_center, fd.timestamp)
            
            # Map state with depth
            frontend_state, confirmed, risk, reason = map_state_with_depth(
                temporal_result.confirmed_state,
                temporal_result.is_confirmed,
                classification.risk_score,
                depth_features,
                depth_config
            )
            
            # Log on state change
            if frontend_state != last_state:
                ts = datetime.now().strftime("%H:%M:%S")
                if frontend_state == "OK":
                    log_buffer.insert(0, {"text": "Postura correcta", "level": "ok", "t": ts})
                elif frontend_state == "FALL":
                    log_buffer.insert(0, {"text": f"Caida: {reason}", "level": "danger", "t": ts})
                else:
                    log_buffer.insert(0, {"text": "Analizando", "level": "warn", "t": ts})
                log_buffer = log_buffer[:8]
                last_state = frontend_state
            
            STATE.update(
                keypoints=kp_json,
                state=frontend_state,
                position=classification.internal_pose.value,
                confidence=quality.score,
                risk_score=risk,
                quality_score=quality.score,
                is_confirmed=confirmed,
                log=log_buffer,
                hip_height_m=depth_features.hip_height_m,
                floor_quality=depth_features.floor_quality,
                depth_mode=depth_features.depth_mode
            )
    finally:
        source.stop()


# ============================================================================
# FASTAPI
# ============================================================================

app = FastAPI(title="Fall Detection")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "web" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


@app.get("/video")
async def video_feed():
    def gen():
        while True:
            jpg = STATE.get_rgb_jpeg()
            if jpg:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
            time.sleep(0.033)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/depth")
async def depth_feed():
    def gen():
        while True:
            jpg = STATE.get_depth_jpeg()
            if jpg:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
            time.sleep(0.033)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(STATE.get_json())
            await asyncio.sleep(0.066)
    except WebSocketDisconnect:
        pass


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-realsense", action="store_true")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    threading.Thread(target=detection_loop, args=(not args.no_realsense, args.camera), daemon=True).start()
    
    print(f"\n{'='*50}")
    print("Fall Detection Web Dashboard")
    print(f"{'='*50}")
    print(f"Dashboard: http://localhost:{args.port}")
    print(f"RGB:       http://localhost:{args.port}/video")
    print(f"Depth:     http://localhost:{args.port}/depth")
    print(f"{'='*50}\n")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
