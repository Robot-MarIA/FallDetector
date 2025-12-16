"""
Dual Sidebar Professional UI (HTML -> Python)

- Layout: left sidebar + video panel + right sidebar
- Video is rendered inside a rounded panel (16:10)
- Skeleton drawn inside the video panel with correct transform
- Text rendered with PIL (single pass)
- Log panel with cards, system stats, clock

Color rules (STRICT for state):
- GREEN: OK
- ORANGE: ANALYZING / Uncertain
- RED: ONLY confirmed fall (risk > 0.8 and is_confirmed True)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from datetime import datetime
import time

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.classifier import RiskState
from core.scheduler import SchedulerMode


# =========================
# THEME (matches your HTML)
# =========================

THEME = {
    "bg": (0, 0, 0),

    # panels
    "panel_fill_rgba": (255, 255, 255, int(255 * 0.05)),
    "panel_hover_rgba": (255, 255, 255, int(255 * 0.08)),
    "panel_border_rgba": (255, 255, 255, int(255 * 0.10)),

    "text": (255, 255, 255, 255),
    "muted": (255, 255, 255, int(255 * 0.60)),
    "dim": (255, 255, 255, int(255 * 0.40)),

    # state colors (RGB)
    "accent": (10, 132, 255),
    "warn": (255, 159, 10),
    "danger": (255, 69, 58),
    "success": (50, 215, 75),
}

# Skeleton palette (RGB)
SKEL = {
    "ankle_knee": (255, 130, 200),     # pink-ish
    "knee_hip": (100, 160, 230),       # blue-ish
    "hip_shoulder": (180, 140, 200),   # purple-ish
    "shoulder_elbow": (100, 200, 220), # cyan-ish
    "elbow_wrist": (240, 220, 100),    # gold-ish
    "connector": (210, 190, 210),
    "head": (200, 200, 200),
}

SKELETON = [
    ((5, 6), "connector"),
    ((5, 7), "shoulder_elbow"), ((7, 9), "elbow_wrist"),
    ((6, 8), "shoulder_elbow"), ((8, 10), "elbow_wrist"),
    ((5, 11), "hip_shoulder"), ((6, 12), "hip_shoulder"),
    ((11, 12), "connector"),
    ((11, 13), "knee_hip"), ((13, 15), "ankle_knee"),
    ((12, 14), "knee_hip"), ((14, 16), "ankle_knee"),
]

POSITIONS = {
    "lying": "Tumbado",
    "sitting_floor": "En el suelo",
    "all_fours": "A cuatro patas",
    "kneeling": "Arrodillado",
    "normal": "De pie",
    "standing": "De pie",
    "unknown": "Desconocido",
}


@dataclass
class LogItem:
    text: str
    ts: float
    level: str  # "ok" | "warn" | "danger" | "info"


# =========================
# Helpers
# =========================

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    # Try to match your HTML fonts
    candidates = [
        "C:/Windows/Fonts/SegoeUI.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()

def _clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))

def _rgb_to_bgr(c: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (c[2], c[1], c[0])

def _now_hms() -> str:
    return datetime.now().strftime("%H:%M:%S")


# =========================
# Main UI
# =========================

class DashboardVisualizer:
    """
    Dual Sidebar UI.

    Public API must remain compatible with your pipeline:
    - draw(frame, keypoints, state, risk_score, quality_score, is_confirmed, internal_pose, scheduler_mode)
    """

    def __init__(
        self,
        target_width: int = 1280,
        target_height: int = 720,
    ):
        self.W = int(target_width)
        self.H = int(target_height)

        # Layout (tuned for 1280x720)
        self.pad = 16
        self.gap = 16
        self.sidebar_w = 340
        self.radius = 20

        # Video panel aspect ratio (your HTML uses 16/10)
        self.video_aspect = 16 / 10

        # Fonts
        self.f_header = _load_font(13)
        self.f_status = _load_font(17)
        self.f_metric_label = _load_font(13)
        self.f_metric_big = _load_font(48)
        self.f_metric_sub = _load_font(17)
        self.f_small = _load_font(11)
        self.f_stat_label = _load_font(11)
        self.f_stat_val = _load_font(28)
        self.f_clock = _load_font(32)

        # FPS
        self._fps_buf: List[float] = []
        self._t_last = time.time()

        # Log
        self._log: List[LogItem] = []
        self._seed_log()

    def _seed_log(self):
        now = time.time()
        self._log = [
            LogItem("Postura correcta detectada", now - 120, "ok"),
            LogItem("Sistema estable", now - 300, "info"),
            LogItem("Monitoreo activo", now - 480, "info"),
            LogItem("Calibración completada", now - 900, "info"),
        ]

    def _fps(self) -> float:
        t = time.time()
        dt = t - self._t_last
        self._t_last = t
        if dt > 0:
            self._fps_buf.append(1.0 / dt)
            if len(self._fps_buf) > 20:
                self._fps_buf.pop(0)
        return sum(self._fps_buf) / max(1, len(self._fps_buf))

    def _state_style(self, state: RiskState, is_confirmed: bool, risk_score: float):
        # returns (label_es, color_rgb)
        if state == RiskState.OK:
            return "Normal", THEME["success"]
        if state == RiskState.NEEDS_HELP and is_confirmed and risk_score > 0.8:
            return "Caída", THEME["danger"]
        return "Analizando", THEME["warn"]

    def _push_log(self, text: str, level: str = "info"):
        now = time.time()
        if self._log and self._log[0].text == text:
            return
        self._log.insert(0, LogItem(text, now, level))
        self._log = self._log[:8]

    def draw(
        self,
        frame: np.ndarray,
        keypoints: Optional[List[Optional[Tuple[float, float, float]]]] = None,
        state: RiskState = RiskState.UNKNOWN,
        risk_score: float = 0.0,
        quality_score: float = 0.0,
        is_confirmed: bool = False,
        internal_pose: str = "unknown",
        scheduler_mode: SchedulerMode = SchedulerMode.CHECKING,
    ) -> np.ndarray:

        # Basic state strings
        pos = POSITIONS.get(str(internal_pose).lower(), "Desconocido")
        status_label, status_color = self._state_style(state, is_confirmed, risk_score)
        fps = self._fps()
        clock = _now_hms()

        # Update log (simple, non-spam)
        if status_label == "Caída":
            self._push_log("Caída confirmada", "danger")
        elif status_label == "Analizando":
            self._push_log("Analizando postura…", "warn")
        else:
            self._push_log("Postura correcta detectada", "ok")

        # Create output canvas
        out = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        out[:] = THEME["bg"]

        # Compute layout rects
        left_x = self.pad
        top_y = self.pad
        left_w = self.sidebar_w
        left_h = self.H - 2 * self.pad

        right_w = self.sidebar_w
        right_x = self.W - self.pad - right_w
        right_y = self.pad
        right_h = left_h

        mid_x = left_x + left_w + self.gap
        mid_w = right_x - self.gap - mid_x
        mid_y = self.pad
        mid_h = left_h

        # Video panel rect (centered in mid)
        # Keep 16:10 inside mid rect
        vp_w = mid_w
        vp_h = int(vp_w / self.video_aspect)
        if vp_h > mid_h:
            vp_h = mid_h
            vp_w = int(vp_h * self.video_aspect)
        vp_x = mid_x + (mid_w - vp_w) // 2
        vp_y = mid_y + (mid_h - vp_h) // 2
        video_rect = (vp_x, vp_y, vp_x + vp_w, vp_y + vp_h)

        # Render the camera frame into video_rect (cover or contain)
        video_bgr, kp_transform = self._render_video_into_rect(frame, video_rect)

        # Draw skeleton onto video (in-place) using kp_transform
        if keypoints:
            self._draw_skeleton_on_video(video_bgr, keypoints, kp_transform)

        # Paste video into out with rounded mask
        self._paste_rounded(out, video_bgr, video_rect, self.radius)

        # Draw the translucent panels + all text in one PIL pass
        out = self._draw_panels_and_text(
            out=out,
            left_rect=(left_x, top_y, left_x + left_w, top_y + left_h),
            right_rect=(right_x, right_y, right_x + right_w, right_y + right_h),
            video_rect=video_rect,
            status_label=status_label,
            status_color=status_color,
            pos=pos,
            quality_score=quality_score,
            fps=fps,
            clock=clock,
        )

        return out

    # ---------- VIDEO RENDER ----------
    def _render_video_into_rect(self, frame_bgr: np.ndarray, rect: Tuple[int,int,int,int]):
        x1, y1, x2, y2 = rect
        W = x2 - x1
        H = y2 - y1

        fh, fw = frame_bgr.shape[:2]

        # Fit (contain) with letterbox inside the video panel (like a "screen")
        s = min(W / fw, H / fh)
        nw, nh = int(fw * s), int(fh * s)
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((H, W, 3), np.uint8)
        # subtle background tint for empty space
        canvas[:] = (0, 0, 0)

        xo = (W - nw) // 2
        yo = (H - nh) // 2
        canvas[yo:yo+nh, xo:xo+nw] = resized

        # Transform for keypoints (orig frame coords -> video_canvas coords)
        transform = {
            "scale": s,
            "x_off": xo,
            "y_off": yo,
        }
        return canvas, transform

    def _draw_skeleton_on_video(self, video_bgr: np.ndarray, keypoints, tr: Dict[str, float]):
        min_conf = 0.30
        # thickness tuned for 16:10 panel
        thick = 2
        rad = 4

        def tp(x, y):
            return int(x * tr["scale"] + tr["x_off"]), int(y * tr["scale"] + tr["y_off"])

        # connections
        for (i, j), seg in SKELETON:
            if i < len(keypoints) and j < len(keypoints):
                a, b = keypoints[i], keypoints[j]
                if a and b and a[2] >= min_conf and b[2] >= min_conf:
                    p1 = tp(a[0], a[1])
                    p2 = tp(b[0], b[1])
                    rgb = SKEL.get(seg, (10,132,255))
                    cv2.line(video_bgr, p1, p2, _rgb_to_bgr(rgb), thick, cv2.LINE_AA)

        # keypoints (minimal, glow-like ring)
        for idx, kp in enumerate(keypoints):
            if kp and kp[2] >= min_conf:
                p = tp(kp[0], kp[1])
                cv2.circle(video_bgr, p, rad + 2, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(video_bgr, p, rad, _rgb_to_bgr(THEME["accent"]), -1, cv2.LINE_AA)

    # ---------- COMPOSITING ----------
    def _paste_rounded(self, out_bgr: np.ndarray, src_bgr: np.ndarray, rect, r: int):
        x1, y1, x2, y2 = rect
        H = y2 - y1
        W = x2 - x1

        src_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
        src_img = Image.fromarray(src_rgb).convert("RGBA")

        mask = Image.new("L", (W, H), 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle((0, 0, W, H), radius=r, fill=255)

        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        base = Image.fromarray(out_rgb).convert("RGBA")
        base.paste(src_img, (x1, y1), mask)

        out_bgr[:] = cv2.cvtColor(np.array(base.convert("RGB")), cv2.COLOR_RGB2BGR)

    # ---------- UI DRAW ----------
    def _draw_panels_and_text(
        self,
        out: np.ndarray,
        left_rect,
        right_rect,
        video_rect,
        status_label: str,
        status_color: Tuple[int,int,int],
        pos: str,
        quality_score: float,
        fps: float,
        clock: str,
    ) -> np.ndarray:

        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).convert("RGBA")
        d = ImageDraw.Draw(img)

        # panels
        self._panel(d, left_rect, self.radius)
        self._panel(d, right_rect, self.radius)
        self._panel(d, video_rect, self.radius)

        # --- Left panel content ---
        lx1, ly1, lx2, ly2 = left_rect
        cx = lx1 + 24
        y = ly1 + 24

        d.text((cx, y), "Estado", font=self.f_header, fill=THEME["muted"])
        y += 16 + 8

        # status card
        card = (lx1 + 24, y, lx2 - 24, y + 148)
        self._card(d, card, 16)
        cy = y + 18

        # dot + text
        dot_x = card[0] + 18
        dot_y = cy + 6
        self._status_dot(d, (dot_x, dot_y), status_color)
        d.text((dot_x + 18, cy), status_label, font=self.f_status, fill=THEME["text"])
        cy += 36

        d.text((card[0] + 18, cy), "Posición", font=self.f_metric_label, fill=THEME["muted"])
        cy += 18
        d.text((card[0] + 18, cy), pos, font=self.f_metric_sub, fill=THEME["muted"])

        y = card[3] + 20

        # confidence big
        d.text((cx, y), "Confianza", font=self.f_metric_label, fill=THEME["muted"])
        y += 18
        conf = int(_clamp(quality_score, 0.0, 1.0) * 100)
        d.text((cx, y), f"{conf}%", font=self.f_metric_big, fill=THEME["text"])

        # footer note
        y = ly2 - 24 - 44
        d.text((cx, y), "Sistema activo · Monitoreo continuo", font=self.f_small, fill=THEME["dim"])
        d.text((cx, y + 16), "Alerta solo en caída confirmada", font=self.f_small, fill=THEME["danger"] + (255,))

        # --- Video overlay badges (top of video) ---
        vx1, vy1, vx2, vy2 = video_rect
        badge_y = vy1 + 16
        badge1 = (vx1 + 16, badge_y, vx1 + 86, badge_y + 28)
        badge2 = (vx2 - 96, badge_y, vx2 - 16, badge_y + 28)
        self._badge(d, badge1, "En vivo", status_color)
        self._badge(d, badge2, f"{int(fps):d} fps", THEME["muted"][:3])

        # --- Right panel content ---
        rx1, ry1, rx2, ry2 = right_rect
        x = rx1 + 24
        y = ry1 + 24
        d.text((x, y), "Registro", font=self.f_header, fill=THEME["muted"])
        y += 16 + 8

        # alert list (cards)
        list_top = y
        max_cards = 5
        card_h = 58
        gap = 10
        for i, item in enumerate(self._log[:max_cards]):
            cy1 = list_top + i * (card_h + gap)
            cy2 = cy1 + card_h
            if cy2 > ry2 - 24 - 140:
                break
            rect = (rx1 + 24, cy1, rx2 - 24, cy2)
            self._card(d, rect, 12)
            self._alert_row(d, rect, item)

        # divider + system
        sys_y = ry2 - 24 - 120
        self._divider(d, (rx1 + 24, sys_y, rx2 - 24, sys_y + 1))

        sys_y += 14
        d.text((x, sys_y), "Sistema", font=self.f_header, fill=THEME["muted"])

        # stat grid (2 cards)
        grid_y = sys_y + 24
        grid_gap = 12
        grid_w = (right_rect[2] - right_rect[0]) - 48
        card_w = (grid_w - grid_gap) // 2

        c1 = (rx1 + 24, grid_y, rx1 + 24 + card_w, grid_y + 74)
        c2 = (rx1 + 24 + card_w + grid_gap, grid_y, rx2 - 24, grid_y + 74)
        self._card(d, c1, 12)
        self._card(d, c2, 12)

        d.text((c1[0] + 12, c1[1] + 10), "FPS", font=self.f_stat_label, fill=THEME["muted"])
        d.text((c1[0] + 12, c1[1] + 28), f"{int(fps):d}", font=self.f_stat_val, fill=THEME["text"])

        d.text((c2[0] + 12, c2[1] + 10), "Cámara", font=self.f_stat_label, fill=THEME["muted"])
        d.text((c2[0] + 12, c2[1] + 28), "●", font=self.f_stat_val, fill=THEME["success"] + (255,))

        # clock
        d.text((rx1 + 24, ry2 - 24 - 36), clock, font=self.f_clock, fill=THEME["text"])

        # back to BGR
        out[:] = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        return out

    # ---------- Drawing primitives ----------
    def _panel(self, d: ImageDraw.ImageDraw, rect, r: int):
        x1, y1, x2, y2 = rect
        d.rounded_rectangle(rect, radius=r, fill=THEME["panel_fill_rgba"], outline=THEME["panel_border_rgba"], width=1)

    def _card(self, d: ImageDraw.ImageDraw, rect, r: int):
        fill = (255, 255, 255, int(255 * 0.03))
        outline = (255, 255, 255, int(255 * 0.08))
        d.rounded_rectangle(rect, radius=r, fill=fill, outline=outline, width=1)

    def _divider(self, d: ImageDraw.ImageDraw, rect):
        x1, y1, x2, y2 = rect
        d.rectangle(rect, fill=(255, 255, 255, int(255 * 0.10)))

    def _status_dot(self, d: ImageDraw.ImageDraw, pos: Tuple[int,int], color_rgb: Tuple[int,int,int]):
        x, y = pos
        r = 6
        glow = (color_rgb[0], color_rgb[1], color_rgb[2], int(255 * 0.30))
        main = (color_rgb[0], color_rgb[1], color_rgb[2], 255)
        d.ellipse((x - r*2, y - r*2, x + r*2, y + r*2), fill=glow)
        d.ellipse((x - r, y - r, x + r, y + r), fill=main)

    def _badge(self, d: ImageDraw.ImageDraw, rect, text: str, dot_color_rgb: Tuple[int,int,int]):
        # badge background
        x1, y1, x2, y2 = rect
        fill = (0, 0, 0, int(255 * 0.70))
        outline = (255, 255, 255, int(255 * 0.15))
        d.rounded_rectangle(rect, radius=14, fill=fill, outline=outline, width=1)
        # dot
        dx = x1 + 10
        dy = y1 + (y2 - y1)//2
        dot = (dot_color_rgb[0], dot_color_rgb[1], dot_color_rgb[2], 255)
        d.ellipse((dx-3, dy-3, dx+3, dy+3), fill=dot)
        # text
        d.text((dx + 8, y1 + 6), text, font=self.f_small, fill=THEME["text"])

    def _alert_row(self, d: ImageDraw.ImageDraw, rect, item: LogItem):
        x1, y1, x2, y2 = rect
        icon_x = x1 + 14
        icon_y = y1 + 18

        if item.level == "danger":
            c = THEME["danger"]
        elif item.level == "warn":
            c = THEME["warn"]
        elif item.level == "ok":
            c = THEME["success"]
        else:
            c = (200, 200, 200)

        dot = (c[0], c[1], c[2], 255)
        d.ellipse((icon_x, icon_y, icon_x + 6, icon_y + 6), fill=dot)

        text_x = icon_x + 16
        d.text((text_x, y1 + 12), item.text, font=self.f_metric_label, fill=THEME["text"])

        # relative time
        dt = max(0, int(time.time() - item.ts))
        if dt < 60:
            ttxt = f"Hace {dt}s"
        elif dt < 3600:
            ttxt = f"Hace {dt//60} min"
        else:
            ttxt = f"Hace {dt//3600} h"

        d.text((text_x, y1 + 32), ttxt, font=self.f_small, fill=THEME["dim"])


# Compatibility aliases expected by your project
ProfessionalDashboard = DashboardVisualizer

def create_dashboard_visualizer(**kwargs) -> DashboardVisualizer:
    return DashboardVisualizer(**kwargs)
