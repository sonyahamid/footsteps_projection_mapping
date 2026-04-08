"""
test_demo.py — 3-stage pipeline visualiser with realistic projection mapping perspective
Camera/projector ~7 feet high, angled 45° down.

  Window 1: Floor canvas (pre-warp)      — flat top-down view + corner markers
  Window 2: Projector output (post-warp) — perspective-corrected projection
  Window 3: Camera screen space          — what the camera actually sees (trapezoid)

Press Q to quit, R to randomise trail.
"""

import cv2
import numpy as np
import random
from projection_mapping import ProjectionMapper, calibrate_from_known_points

# ─────────────────────────────────────────────────────────────────────────────
# Realistic perspective simulation
# ─────────────────────────────────────────────────────────────────────────────
#
# Setup: camera at height ~7ft, angled 45° downward, offset slightly to one side.
# At 45°, the far edge of the floor quad is farther from the camera than the near
# edge, so the trapezoid is wider at the bottom (near) and narrower at the top (far).
#
# Camera image: 640x480
#   - near edge (bottom of image) is ~3ft from camera base on the floor
#   - far edge  (top of image)    is ~10ft away
#   - lateral spread ~8ft wide
#
# These numbers are eyeballed to feel like a real install:

CAM_W, CAM_H = 640, 480

cam_pts = [
    (110,  90),   # TL — far-left  (compressed by perspective)
    (530,  90),   # TR — far-right
    (590, 430),   # BR — near-right (wider, lower in frame)
    ( 50, 430),   # BL — near-left
]

# Projector output: 1920x1080
# The projector is similarly offset — it throws a keystone-corrected quad.
# H_proj will warp floor-space back into this shape so it lands flat on the floor.

PROJ_W, PROJ_H = 1920, 1080

proj_pts = [
    (160,  80),   # TL
    (1760,  80),  # TR
    (1840, 1000), # BR
    ( 80, 1000),  # BL
]

calibrate_from_known_points(
    cam_pts  = cam_pts,
    proj_pts = proj_pts,
    floor_w  = 1000,
    floor_h  = 1000,
    proj_w   = PROJ_W,
    proj_h   = PROJ_H,
    out_path = "calibration_test.json",
)

mapper = ProjectionMapper("calibration_test.json")

# ─────────────────────────────────────────────────────────────────────────────
# Floor-space corner markers (TL TR BR BL) — what we tape on the actual floor
# ─────────────────────────────────────────────────────────────────────────────

FLOOR_CORNERS = [
    (  0,   0),   # TL
    (1000,  0),   # TR
    (1000, 1000), # BR
    (  0, 1000),  # BL
]
CORNER_LABELS  = ["TL", "TR", "BR", "BL"]
CORNER_COLORS  = [(0,0,255), (0,255,0), (255,0,255), (0,200,255)]

def draw_floor_corners(canvas):
    """Draw the 4 calibration corners onto a floor-space canvas."""
    for (fx, fy), label, color in zip(FLOOR_CORNERS, CORNER_LABELS, CORNER_COLORS):
        x, y = int(fx), int(fy)
        # Clamp so circles don't get clipped at exact edges
        xd = max(12, min(canvas.shape[1]-12, x))
        yd = max(12, min(canvas.shape[0]-12, y))
        cv2.circle(canvas, (xd, yd), 14, color, -1)
        cv2.circle(canvas, (xd, yd), 16, (255,255,255), 1)
        offset = (xd+18, yd+5) if xd < canvas.shape[1]//2 else (xd-40, yd+5)
        cv2.putText(canvas, label, offset,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    # Draw bounding quad
    pts = np.array([[max(1,min(999,x)), max(1,min(999,y))] for x,y in FLOOR_CORNERS],
                   np.int32).reshape((-1,1,2))
    cv2.polylines(canvas, [pts], True, (80,80,80), 1)
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Random trail
# ─────────────────────────────────────────────────────────────────────────────

def make_random_trail(n=12):
    x, y = random.randint(300, 700), random.randint(300, 700)
    pts = [(x, y)]
    dx, dy = random.randint(-40, 40), random.randint(-40, 40)  # initial direction
    for _ in range(n - 1):
        # Gradually steer rather than random jumps — feels more like walking
        dx = dx + random.randint(-30, 30) * 0.3
        dy = dy + random.randint(-30, 30) * 0.3
        x = max(80, min(920, int(x + dx)))
        y = max(80, min(920, int(y + dy)))
        pts.append((x, y))
    return pts


trail = make_random_trail()

# ─────────────────────────────────────────────────────────────────────────────
# Camera view renderer (inverse warp from floor -> camera space)
# ─────────────────────────────────────────────────────────────────────────────

def render_camera_view(mapper, floor_pts, trail_pts, cam_w=CAM_W, cam_h=CAM_H):
    H_cam_inv = np.linalg.inv(mapper.H_cam)
    canvas = mapper.make_floor_canvas()
    canvas = mapper.do_stuff(canvas, floor_pts, trail_pts)

    # Draw the floor corners on the floor canvas before warping into cam space
    draw_floor_corners(canvas)

    # Draw the camera quad directly in camera space (overlay after warp)
    result = cv2.warpPerspective(canvas, H_cam_inv, (cam_w, cam_h))

    # Overlay the raw cam_pts as corner markers in camera space
    for (cx, cy), label, color in zip(cam_pts, CORNER_LABELS, CORNER_COLORS):
        cv2.circle(result, (cx, cy), 8, color, -1)
        cv2.circle(result, (cx, cy), 10, (255,255,255), 1)
        ox = cx + 12 if cx < cam_w // 2 else cx - 30
        cv2.putText(result, label, (ox, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Draw the trapezoid quad to show calibrated region
    pts = np.array(cam_pts, np.int32).reshape((-1,1,2))
    cv2.polylines(result, [pts], True, (200,200,200), 1)
    return result


def label(img, text, sub=None):
    out = img.copy()
    cv2.putText(out, text, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220,220,220), 2)
    if sub:
        cv2.putText(out, sub, (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

print("Q = quit   R = new trail")

while True:

    # ── Stage 1: floor canvas (pre-warp, flat top-down) ───────────────────
    floor_canvas = mapper.make_floor_canvas()
    floor_canvas = mapper.do_stuff(floor_canvas, [trail[-1]], trail)
    draw_floor_corners(floor_canvas)
    view_floor = label(floor_canvas,
                       "1  floor space (pre-warp)",
                       "flat 1000x1000, corners = tape marks on floor")

    # ── Stage 2: projector output (post-warp) ─────────────────────────────
    proj_full  = mapper.render_projector_frame(floor_pts=[trail[-1]], trail_pts=trail)

    # Draw proj corner markers directly in projector space
    for (px, py), lbl, color in zip(proj_pts, CORNER_LABELS, CORNER_COLORS):
        cv2.circle(proj_full, (px, py), 18, color, -1)
        cv2.circle(proj_full, (px, py), 20, (255,255,255), 1)
        ox = px + 24 if px < PROJ_W // 2 else px - 56
        cv2.putText(proj_full, lbl, (ox, py + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    proj_small = cv2.resize(proj_full, (960, 540))
    view_proj  = label(proj_small,
                       "2  projector output (post-warp)",
                       "1920x1080 shown at 50% — keystone corrects for angle")

    # ── Stage 3: camera screen space ──────────────────────────────────────
    cam_view  = render_camera_view(mapper, [trail[-1]], trail)
    view_cam  = label(cam_view,
                      "3  camera screen space",
                      "640x480 — trapezoid = what camera sees from 7ft / 45 deg")

    cv2.imshow("1 | floor canvas (pre-warp)",      view_floor)
    cv2.imshow("2 | projector output (post-warp)",  view_proj)
    cv2.imshow("3 | camera screen space",           view_cam)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        trail = make_random_trail()
        mapper.tick = 0  # reset so steps reveal from the beginning
        print("New trail:", trail)

cv2.destroyAllWindows()