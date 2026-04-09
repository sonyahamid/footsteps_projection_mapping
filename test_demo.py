"""
Footsteps projection-mapping frontend.

The runtime now uses two windows:
  1. Debug window: webcam preview with draggable calibration points.
  2. Projector window: the final warped frame that gets sent to the projector.

Press G to toggle a verification grid on the projector output.
The grid is drawn in floor space then warped through H_proj, so you can
lay tape on the floor in a matching grid and check alignment visually.
"""

from __future__ import annotations

import os
import socket
import time
import random
import json

import cv2
import numpy as np

from projection_mapping import ProjectionMapper, calibrate_from_known_points, compute_homography, warp_frame

CAM_W, CAM_H = 640, 480
PROJ_W, PROJ_H = 1920, 1080
FLOOR_W, FLOOR_H = 1000, 1000
CALIBRATION_PATH = "calibration.json"
DEBUG_WINDOW = "Footsteps Debug"
PROD_WINDOW = "Footsteps Projector"
WEBCAM_INDEX = 0

cam_pts = [
    [110, 90],
    [530, 90],
    [590, 430],
    [50, 430],
]

proj_pts = [
    (160, 80),
    (1760, 80),
    (1840, 1000),
    (80, 1000),
]

FLOOR_CORNERS = [
    (0, 0),
    (FLOOR_W, 0),
    (FLOOR_W, FLOOR_H),
    (0, FLOOR_H),
]
CORNER_LABELS = ["TL", "TR", "BR", "BL"]
CORNER_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 200, 255)]

# Try to load existing calibration, or create a default one in-memory
if os.path.exists(CALIBRATION_PATH):
    mapper = ProjectionMapper(CALIBRATION_PATH)
    if mapper.cam_pts:
        cam_pts = [list(pt) for pt in mapper.cam_pts]
    if mapper.proj_pts:
        proj_pts = [list(pt) for pt in mapper.proj_pts]
else:
    # Just initialize a mapper manually based on default points since no file exists yet
    H_cam = compute_homography(cam_pts, FLOOR_CORNERS)
    H_proj = compute_homography(FLOOR_CORNERS, proj_pts)
    
    # We must mock enough to make ProjectionMapper work without a file
    class MockMapper(ProjectionMapper):
        def __init__(self):
            self.floor_w = FLOOR_W
            self.floor_h = FLOOR_H
            self.proj_w = PROJ_W
            self.proj_h = PROJ_H
            self.H_cam = H_cam
            self.H_proj = H_proj
            self.cam_pts = cam_pts
            self.proj_pts = proj_pts
            from projection_mapping import GifAnimator
            self.animators = [
                GifAnimator("white_foot.gif", size=(27,54)),
                GifAnimator("white_foot2.gif", size=(27,54)),
                GifAnimator("white_foot3.gif", size=(27,54))
            ]
            self.tick = 0
            
    mapper = MockMapper()


def draw_floor_corners(canvas):
    """Draw the four floor-space corners onto a canvas."""
    for (fx, fy), label, color in zip(FLOOR_CORNERS, CORNER_LABELS, CORNER_COLORS):
        x, y = int(fx), int(fy)
        xd = max(12, min(canvas.shape[1] - 12, x))
        yd = max(12, min(canvas.shape[0] - 12, y))
        cv2.circle(canvas, (xd, yd), 14, color, -1)
        cv2.circle(canvas, (xd, yd), 16, (255, 255, 255), 1)
        offset = (xd + 18, yd + 5) if xd < canvas.shape[1] // 2 else (xd - 40, yd + 5)
        cv2.putText(canvas, label, offset, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    pts = np.array(
        [[max(1, min(FLOOR_W - 1, x)), max(1, min(FLOOR_H - 1, y))] for x, y in FLOOR_CORNERS],
        np.int32,
    ).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts], True, (80, 80, 80), 1)
    return canvas



class FootstepUDPReceiver:
    def __init__(self, port=7000):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.setblocking(False)
        self.people = {}
        self.matches = {}

    def poll(self):
        try:
            while True:
                data, _addr = self.sock.recvfrom(4096)
                if not data:
                    break

                now = time.time()
                text = data.decode("utf-8").strip()
                for line in text.split("\n"):
                    if not line:
                        continue

                    parts = line.split()
                    if parts[0] == "MATCH":
                        if len(parts) < 4:
                            continue

                        person_id = int(parts[1])
                        history_length = int(parts[2])
                        age_secs = float(parts[3])
                        history = []
                        idx = 4
                        for _ in range(history_length):
                            if idx + 3 < len(parts):
                                history.append(
                                    (
                                        float(parts[idx]),
                                        float(parts[idx + 1]),
                                        float(parts[idx + 2]),
                                        float(parts[idx + 3]),
                                    )
                                )
                                idx += 4

                        self.matches[person_id] = {
                            "age": age_secs,
                            "history": history,
                            "last_seen": now,
                        }
                    else:
                        if len(parts) < 6:
                            continue

                        x = float(parts[0])
                        y = float(parts[1])
                        dx = float(parts[2])
                        dy = float(parts[3])
                        person_id = int(parts[4])
                        history_length = int(parts[5])

                        history = []
                        idx = 6
                        for _ in range(history_length):
                            if idx + 3 < len(parts):
                                history.append(
                                    (
                                        float(parts[idx]),
                                        float(parts[idx + 1]),
                                        float(parts[idx + 2]),
                                        float(parts[idx + 3]),
                                    )
                                )
                                idx += 4

                        self.people[person_id] = {
                            "pos": (x, y, dx, dy),
                            "history": history,
                            "last_seen": now,
                        }
        except BlockingIOError:
            pass
        except Exception:
            pass

        now = time.time()
        dead_ids = [pid for pid, data in self.people.items() if now - data["last_seen"] > 1.0]
        for pid in dead_ids:
            del self.people[pid]

        dead_matches = [pid for pid, data in self.matches.items() if now - data["last_seen"] > 1.0]
        for pid in dead_matches:
            del self.matches[pid]


class DraggableCalibration:
    def __init__(self, points):
        self.points = [list(point) for point in points]
        self.labels = CORNER_LABELS
        self.colors = CORNER_COLORS
        self.active_index = None
        self.drag_radius = 24

    def _hit_test(self, x, y):
        best_index = None
        best_distance = None
        for index, (px, py) in enumerate(self.points):
            distance = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
            if distance <= self.drag_radius and (best_distance is None or distance < best_distance):
                best_index = index
                best_distance = distance
        return best_index

    def mouse_callback(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.active_index = self._hit_test(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.active_index is not None:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                self.points[self.active_index][0] = int(x)
                self.points[self.active_index][1] = int(y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.active_index = None

    def as_tuples(self):
        return [(point[0], point[1]) for point in self.points]


class WebcamPreview:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
        self.opened = self.cap.isOpened()

    def read(self):
        if not self.opened:
            return None

        ok, frame = self.cap.read()
        if not ok or frame is None or frame.size == 0:
            return None

        return cv2.resize(frame, (CAM_W, CAM_H))

    def release(self):
        if self.cap is not None:
            self.cap.release()


def render_debug_fallback(mapper, person_trails, matched_trails=None, cam_w=CAM_W, cam_h=CAM_H):
    H_cam_inv = np.linalg.inv(mapper.H_cam)
    canvas = mapper.make_floor_canvas()

    if person_trails:
        for trail in person_trails:
            if trail:
                canvas = mapper.do_stuff(canvas, [trail[-1]], trail[:-1])

    if matched_trails:
        for match in matched_trails:
            if match["trail"]:
                canvas = mapper.do_stuff(canvas, [match["trail"][-1]], match["trail"][:-1], age_label=match["age_str"])

    draw_floor_corners(canvas)
    return cv2.warpPerspective(canvas, H_cam_inv, (cam_w, cam_h))


def render_webcam_overlay(frame, editor, show_grid=False, show_camera=False, show_proj_editor=False):
    out = frame.copy()
    points = editor.as_tuples()

    for index, ((x, y), label, color) in enumerate(zip(points, CORNER_LABELS, CORNER_COLORS)):
        radius = 10 if editor.active_index == index else 8
        line_thickness = 3 if editor.active_index == index else 2
        cv2.circle(out, (x, y), radius + 3, (255, 255, 255), 1)
        cv2.circle(out, (x, y), radius, color, -1)
        offset_x = x + 12 if x < out.shape[1] // 2 else x - 32
        cv2.putText(out, label, (offset_x, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, line_thickness)

    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [pts], True, (220, 220, 220), 1)
    cv2.putText(
        out,
        "Drag TL/TR/BR/BL to align the floor outline",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (240, 240, 240),
        2,
    )

    grid_label = "[G] Grid: ON" if show_grid else "[G] Grid: off"
    grid_color = (0, 255, 200) if show_grid else (160, 160, 160)
    cv2.putText(out, grid_label, (12, out.shape[0] - 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, grid_color, 1)

    cam_label = "[C] Camera: ON" if show_camera else "[C] Camera: off"
    cam_color = (0, 200, 255) if show_camera else (160, 160, 160)
    cv2.putText(out, cam_label, (12, out.shape[0] - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cam_color, 1)
                
    proj_label = "[P] Proj Editor: ON" if show_proj_editor else "[P] Proj Editor: off"
    proj_color = (0, 255, 200) if show_proj_editor else (160, 160, 160)
    cv2.putText(out, proj_label, (12, out.shape[0] - 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, proj_color, 1)

    cv2.putText(
        out,
        "Q to quit | S to save",
        (12, out.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
    )
    return out


def render_projector_overlay(frame, editor):
    out = frame.copy()
    points = editor.as_tuples()

    for index, ((x, y), label, color) in enumerate(zip(points, CORNER_LABELS, CORNER_COLORS)):
        radius = 10 if editor.active_index == index else 8
        line_thickness = 3 if editor.active_index == index else 2
        cv2.circle(out, (x, y), radius + 3, (255, 255, 255), 1)
        cv2.circle(out, (x, y), radius, color, -1)
        offset_x = x + 12 if x < out.shape[1] // 2 else x - 32
        cv2.putText(out, label, (offset_x, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, line_thickness)

    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [pts], True, (220, 220, 220), 1)
    
    cv2.putText(
        out,
        "Drag TL/TR/BR/BL to align the projector output map",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (240, 240, 240),
        2,
    )
    return out

def _draw_floor_grid(canvas, floor_w=FLOOR_W, floor_h=FLOOR_H, divisions=10):
    """Draw a verification grid in floor space before H_proj warp."""
    step_x = floor_w // divisions
    step_y = floor_h // divisions

    for i in range(divisions + 1):
        bright = (i % 5 == 0)
        color = (0, 220, 220) if bright else (0, 80, 80)
        thickness = 2 if bright else 1
        x = i * step_x
        cv2.line(canvas, (x, 0), (x, floor_h), color, thickness)
        y = i * step_y
        cv2.line(canvas, (0, y), (floor_w, y), color, thickness)
        if bright:
            for j in range(0, divisions + 1, 5):
                y2 = j * step_y
                cv2.putText(canvas, f"{x},{y2}", (x + 4, y2 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (floor_w - 1, floor_h - 1), (0, 255, 180), 2)
    for cx, cy, lbl in [(6, 18, "TL"), (floor_w-38, 18, "TR"),
                        (floor_w-38, floor_h-6, "BR"), (6, floor_h-6, "BL")]:
        cv2.putText(canvas, lbl, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2, cv2.LINE_AA)
    return canvas



def main():
    udp_receiver = FootstepUDPReceiver(7000)
    webcam = WebcamPreview(WEBCAM_INDEX)
    editor = DraggableCalibration(cam_pts)
    proj_editor = DraggableCalibration(proj_pts)
    current_h_cam = mapper.H_cam.copy()
    current_h_proj = mapper.H_proj.copy()
    webcam_frame = None

    show_grid = False    # G 
    show_camera = False  # C 
    show_proj_editor = False # P

    cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(PROD_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(DEBUG_WINDOW, editor.mouse_callback)
    cv2.setMouseCallback(PROD_WINDOW, proj_editor.mouse_callback)

    print("Q = quit | G = toggle grid | C = toggle camera passthrough | P = toggle projector map editor | S = save calibration data")

    selected_pid = None

    try:
        while True:
            udp_receiver.poll()

            current_cam_pts = editor.as_tuples()
            candidate_h_cam = compute_homography(current_cam_pts, FLOOR_CORNERS)
            if candidate_h_cam is not None:
                current_h_cam = candidate_h_cam
            mapper.H_cam = current_h_cam

            current_proj_pts = proj_editor.as_tuples()
            candidate_h_proj = compute_homography(FLOOR_CORNERS, current_proj_pts)
            if candidate_h_proj is not None:
                current_h_proj = candidate_h_proj
            mapper.H_proj = current_h_proj

            person_trails = []
            for pid, data in udp_receiver.people.items():
                trail = []
                for hx, hy, hdx, hdy in data["history"]:
                    hcx, hcy = hx * CAM_W, hy * CAM_H
                    fx, fy = mapper.cam_to_floor((hcx, hcy))
                    hx2 = hcx + hdx * 50.0
                    hy2 = hcy + hdy * 50.0
                    fx2, fy2 = mapper.cam_to_floor((hx2, hy2))
                    trail.append((fx, fy, fx2 - fx, fy2 - fy))

                if trail:
                    person_trails.append(trail)

            matched_trails = []
            if udp_receiver.matches:
                if selected_pid not in udp_receiver.matches:
                    selected_pid = random.choice(list(udp_receiver.matches.keys()))
                
                data = udp_receiver.matches[selected_pid]
                trail = []
                for hx, hy, hdx, hdy in data["history"]:
                    hcx, hcy = hx * CAM_W, hy * CAM_H
                    fx, fy = mapper.cam_to_floor((hcx, hcy))
                    hx2 = hcx + hdx * 50.0
                    hy2 = hcy + hdy * 50.0
                    fx2, fy2 = mapper.cam_to_floor((hx2, hy2))
                    trail.append((fx, fy, fx2 - fx, fy2 - fy))

                if trail:
                    age_secs = data["age"]
                    if age_secs < 60:
                        age_str = f"({int(age_secs)} secs ago)"
                    elif age_secs < 3600:
                        age_str = f"({int(age_secs // 60)} mins ago)"
                    else:
                        age_str = f"({int(age_secs // 3600)} hours ago)"

                    matched_trails.append({"trail": trail, "age_str": age_str})
            else:
                selected_pid = None

            projected_frame = mapper.render_projector_frame(
                person_trails=[],  # Only show history paths
                matched_trails=matched_trails,
            )

            webcam_frame = webcam.read()
            if show_grid:
                grid_canvas = mapper.make_floor_canvas()
                _draw_floor_grid(grid_canvas)
                grid_warped = warp_frame(mapper.H_proj, grid_canvas, PROJ_W, PROJ_H)
                mask = cv2.cvtColor(grid_warped, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
                mask_3ch = cv2.merge([mask, mask, mask])
                projected_frame = np.where(mask_3ch > 0, grid_warped, projected_frame)

            if show_camera and webcam_frame is not None:
                floor_from_cam = warp_frame(mapper.H_cam, webcam_frame, FLOOR_W, FLOOR_H)
                projected_frame = warp_frame(mapper.H_proj, floor_from_cam, PROJ_W, PROJ_H)

            if show_proj_editor:
                projected_frame = render_projector_overlay(projected_frame, proj_editor)

            if webcam_frame is None:
                debug_base = render_debug_fallback(mapper, [], matched_trails)
            else:
                debug_base = webcam_frame

            debug_frame = render_webcam_overlay(debug_base, editor, show_grid, show_camera, show_proj_editor)

            cv2.imshow(DEBUG_WINDOW, debug_frame)
            cv2.imshow(PROD_WINDOW, projected_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("g"):
                show_grid = not show_grid
            elif key == ord("c"):
                show_camera = not show_camera
            elif key == ord("p"):
                show_proj_editor = not show_proj_editor
            elif key == ord("s"):
                # Save calibration
                data = {
                    "cam_pts": current_cam_pts,
                    "proj_pts": current_proj_pts,
                    "floor_corners": FLOOR_CORNERS,
                    "floor_w": FLOOR_W,
                    "floor_h": FLOOR_H,
                    "proj_w": PROJ_W,
                    "proj_h": PROJ_H,
                    "H_cam": current_h_cam.tolist(),
                    "H_proj": current_h_proj.tolist(),
                }
                with open(CALIBRATION_PATH, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"[{time.strftime('%H:%M:%S')}] Calibration saved to {CALIBRATION_PATH}")
                break
            elif key == ord("g"):
                show_grid = not show_grid
            elif key == ord("s"):
                from projection_mapping import calibrate_from_known_points
                calibrate_from_known_points(
                    cam_pts=editor.as_tuples(),
                    proj_pts=proj_pts,
                    floor_w=FLOOR_W,
                    floor_h=FLOOR_H,
                    proj_w=PROJ_W,
                    proj_h=PROJ_H,
                    out_path=CALIBRATION_PATH,
                )
                print(f"Calibration saved to {CALIBRATION_PATH}")
                print(f"Grid {'ON' if show_grid else 'off'}")
            elif key == ord("c"):
                show_camera = not show_camera
                print(f"Camera passthrough {'ON' if show_camera else 'off'}")
    finally:
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()