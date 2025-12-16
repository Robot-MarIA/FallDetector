"""
Explainable logging for analysis and reproducibility.

This module provides detailed logging to CSV and JSON formats for:
- TFG analysis and defense
- System behavior analysis
- Reproducibility of results

All decisions are logged with:
- Timestamps
- State and scores
- Feature values (torso angle, height ratio, etc.)
- Scheduler mode
- Reason strings

This is MANDATORY for academic work and debugging.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import threading


@dataclass
class LogEntry:
    """
    A single log entry with all explainable data.
    
    This captures everything needed to understand
    and reproduce a classification decision.
    """
    # Timing
    timestamp: float
    datetime_str: str
    frame_number: int
    
    # Classification
    risk_state: str
    confirmed_state: str
    internal_pose: str
    risk_score: float
    quality_score: float
    is_confirmed: bool
    
    # Features
    torso_angle: Optional[float]
    hip_height_ratio: Optional[float]
    shoulder_height_ratio: Optional[float]
    bbox_aspect_ratio: float
    compactness: Optional[float]
    bbox_bottom_ratio: float
    n_valid_keypoints: int
    
    # Scheduler
    scheduler_mode: str
    target_fps: float
    
    # Explainability
    reason: str
    temporal_reason: str
    
    # Detection info
    n_persons_detected: int
    selection_reason: str


class ExplainableLogger:
    """
    Logger that writes detailed, explainable entries to CSV and JSON.
    
    Features:
    - CSV for easy spreadsheet analysis
    - JSON for programmatic analysis
    - Thread-safe writing
    - Automatic file rotation (optional)
    
    Args:
        output_dir: Directory for log files
        session_name: Name for this logging session
        write_csv: Enable CSV output
        write_json: Enable JSON output
        buffer_size: Number of entries to buffer before writing
    """
    
    def __init__(
        self,
        output_dir: str = "logs",
        session_name: Optional[str] = None,
        write_csv: bool = True,
        write_json: bool = True,
        buffer_size: int = 10
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate session name from timestamp if not provided
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name
        
        self.write_csv = write_csv
        self.write_json = write_json
        self.buffer_size = buffer_size
        
        # File paths
        self.csv_path = self.output_dir / f"{session_name}.csv"
        self.json_path = self.output_dir / f"{session_name}.json"
        
        # Buffers
        self.buffer: List[LogEntry] = []
        self.all_entries: List[Dict] = []  # For JSON (keeps all)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # CSV header written flag
        self._csv_header_written = False
        
        print(f"Logging to: {self.output_dir}")
        if write_csv:
            print(f"  CSV: {self.csv_path}")
        if write_json:
            print(f"  JSON: {self.json_path}")
    
    def log(
        self,
        timestamp: float,
        frame_number: int,
        risk_state: str,
        confirmed_state: str,
        internal_pose: str,
        risk_score: float,
        quality_score: float,
        is_confirmed: bool,
        torso_angle: Optional[float],
        hip_height_ratio: Optional[float],
        shoulder_height_ratio: Optional[float],
        bbox_aspect_ratio: float,
        compactness: Optional[float],
        bbox_bottom_ratio: float,
        n_valid_keypoints: int,
        scheduler_mode: str,
        target_fps: float,
        reason: str,
        temporal_reason: str,
        n_persons_detected: int,
        selection_reason: str
    ):
        """
        Log a single frame's analysis results.
        
        All arguments are explicit to ensure complete logging.
        """
        entry = LogEntry(
            timestamp=timestamp,
            datetime_str=datetime.fromtimestamp(timestamp).isoformat(),
            frame_number=frame_number,
            risk_state=risk_state,
            confirmed_state=confirmed_state,
            internal_pose=internal_pose,
            risk_score=round(risk_score, 4),
            quality_score=round(quality_score, 4),
            is_confirmed=is_confirmed,
            torso_angle=round(torso_angle, 2) if torso_angle is not None else None,
            hip_height_ratio=round(hip_height_ratio, 4) if hip_height_ratio is not None else None,
            shoulder_height_ratio=round(shoulder_height_ratio, 4) if shoulder_height_ratio is not None else None,
            bbox_aspect_ratio=round(bbox_aspect_ratio, 4),
            compactness=round(compactness, 4) if compactness is not None else None,
            bbox_bottom_ratio=round(bbox_bottom_ratio, 4),
            n_valid_keypoints=n_valid_keypoints,
            scheduler_mode=scheduler_mode,
            target_fps=round(target_fps, 1),
            reason=reason,
            temporal_reason=temporal_reason,
            n_persons_detected=n_persons_detected,
            selection_reason=selection_reason,
        )
        
        with self.lock:
            self.buffer.append(entry)
            self.all_entries.append(asdict(entry))
            
            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()
    
    def _flush_buffer(self):
        """Write buffered entries to files."""
        if not self.buffer:
            return
        
        if self.write_csv:
            self._write_csv()
        
        # JSON is written on close (full file)
        
        self.buffer.clear()
    
    def _write_csv(self):
        """Append buffer to CSV file."""
        mode = 'a' if self._csv_header_written else 'w'
        
        with open(self.csv_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if not self._csv_header_written:
                # Write header
                if self.buffer:
                    header = list(asdict(self.buffer[0]).keys())
                    writer.writerow(header)
                    self._csv_header_written = True
            
            for entry in self.buffer:
                writer.writerow(list(asdict(entry).values()))
    
    def _write_json(self):
        """Write all entries to JSON file."""
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'session': self.session_name,
                'created': datetime.now().isoformat(),
                'n_entries': len(self.all_entries),
                'entries': self.all_entries
            }, f, indent=2)
    
    def close(self):
        """Flush remaining buffer and close files."""
        with self.lock:
            self._flush_buffer()
            
            if self.write_json:
                self._write_json()
        
        print(f"Logged {len(self.all_entries)} entries")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of logged data."""
        if not self.all_entries:
            return {'n_entries': 0}
        
        states = [e['confirmed_state'] for e in self.all_entries]
        
        return {
            'n_entries': len(self.all_entries),
            'first_timestamp': self.all_entries[0]['datetime_str'],
            'last_timestamp': self.all_entries[-1]['datetime_str'],
            'state_counts': {
                'OK': states.count('OK'),
                'RISK': states.count('RISK'),
                'NEEDS_HELP': states.count('NEEDS_HELP'),
                'UNKNOWN': states.count('UNKNOWN'),
            },
            'avg_risk_score': sum(e['risk_score'] for e in self.all_entries) / len(self.all_entries),
            'avg_quality_score': sum(e['quality_score'] for e in self.all_entries) / len(self.all_entries),
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def create_logger(**kwargs) -> ExplainableLogger:
    """Factory function to create a logger."""
    return ExplainableLogger(**kwargs)
