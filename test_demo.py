"""
Footsteps projection-mapping frontend.

The runtime now uses two windows:
  1. Debug window: webcam preview with draggable calibration points.
  2. Projector window: the final warped frame that gets sent to the projector.

The debug window lets you drag the four camera-space corner markers to align the
floor quadrilateral. The projector window intentionally stays clean: only the
footstep animation and path text are rendered there.
"""

from __future__ import annotations

import socket
import time

import cv2
import numpy as np

from projection_mapping import ProjectionMapper, calibrate_from_known_points, compute_homography

CAM_W, CAM_H = 640, 480
PROJ_W, PROJ_H = 1920, 1080
FLOOR_W, FLOOR_H = 1000, 1000
CALIBRATION_PATH = "calibration_test.json"
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

calibrate_from_known_points(
    cam_pts=cam_pts,
    proj_pts=proj_pts,
    floor_w=FLOOR_W,
    floor_h=FLOOR_H,
    proj_w=PROJ_W,
    proj_h=PROJ_H,
    out_path=CALIBRATION_PATH,
)

mapper = ProjectionMapper(CALIBRATION_PATH)


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


def render_webcam_overlay(frame, editor):
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
    cv2.putText(
        out,
        "Q to quit",
        (12, out.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
    )
    return out


def main():
    udp_receiver = FootstepUDPReceiver(7000)
    webcam = WebcamPreview(WEBCAM_INDEX)
    editor = DraggableCalibration(cam_pts)
    current_h_cam = mapper.H_cam.copy()

    cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(PROD_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(DEBUG_WINDOW, editor.mouse_callback)

    print("Q = quit")

    try:
        while True:
            udp_receiver.poll()

            current_cam_pts = editor.as_tuples()
            candidate_h_cam = compute_homography(current_cam_pts, FLOOR_CORNERS)
            if candidate_h_cam is not None:
                current_h_cam = candidate_h_cam
            mapper.H_cam = current_h_cam

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
            for pid, data in udp_receiver.matches.items():
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

            projected_frame = mapper.render_projector_frame(
                person_trails=[],  # Only show history paths
                matched_trails=matched_trails,
            )

            webcam_frame = webcam.read()
            if webcam_frame is None:
                debug_base = render_debug_fallback(mapper, [], matched_trails)
            else:
                debug_base = webcam_frame

            debug_frame = render_webcam_overlay(debug_base, editor)

            cv2.imshow(DEBUG_WINDOW, debug_frame)
            cv2.imshow(PROD_WINDOW, projected_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()