import cv2
import json
import os
import numpy as np

CALIBRATION_PATH = "calibration.json"
CAM_W, CAM_H = 640, 480
PROJ_W, PROJ_H = 1920, 1080
FLOOR_W, FLOOR_H = 1000, 1000
DEBUG_WINDOW = "Footsteps Calibration Editor"
WEBCAM_INDEX = 0

CORNER_LABELS = ["TL", "TR", "BR", "BL"]
CORNER_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 200, 255)]
FLOOR_CORNERS = [(0, 0), (FLOOR_W, 0), (FLOOR_W, FLOOR_H), (0, FLOOR_H)]

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

def compute_homography(src_pts, dst_pts):
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    H, status = cv2.findHomography(src, dst)
    if H is None:
        return np.eye(3, dtype=np.float32)
    return H

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
        "Drag TL/TR/BR/BL to align the floor outline. Press 'S' to save.",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (240, 240, 240),
        2,
    )
    return out

def main():
    if not os.path.exists(CALIBRATION_PATH):
        print(f"Waiting for {CALIBRATION_PATH} to be created by the server...")
        return
        
    with open(CALIBRATION_PATH) as f:
        calib = json.load(f)
        cam_pts = calib['cam_pts']
        proj_pts = calib['proj_pts']

    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_ANY)

    editor = DraggableCalibration(cam_pts)
    proj_editor = DraggableCalibration(proj_pts)

    cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow("Projector Output Config", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(DEBUG_WINDOW, editor.mouse_callback)
    cv2.setMouseCallback("Projector Output Config", proj_editor.mouse_callback)
    
    print("Running Calibration UI. Drag corners to live-update mapping, 'Q' to quit.")

    last_cam_pts = None
    last_proj_pts = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            # Fallback if no camera
            frame = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (CAM_W, CAM_H))
            
        debug_frame = render_webcam_overlay(frame, editor)
        
        # Simple projector UI
        proj_canvas = np.zeros((PROJ_H, PROJ_W, 3), dtype=np.uint8)
        proj_frame = render_webcam_overlay(proj_canvas, proj_editor)

        cv2.imshow(DEBUG_WINDOW, debug_frame)
        cv2.imshow("Projector Output Config", proj_frame)

        current_cam_pts = editor.as_tuples()
        current_proj_pts = proj_editor.as_tuples()

        if current_cam_pts != last_cam_pts or current_proj_pts != last_proj_pts:
            last_cam_pts = list(current_cam_pts)
            last_proj_pts = list(current_proj_pts)
            
            try:
                data = calib.copy()
                data["cam_pts"] = current_cam_pts
                data["proj_pts"] = current_proj_pts
                H_cam = compute_homography(current_cam_pts, FLOOR_CORNERS)
                H_proj = compute_homography(FLOOR_CORNERS, current_proj_pts)
                data["H_cam"] = H_cam.tolist()
                data["H_proj"] = H_proj.tolist()
                
                # Write to temp file then rename to avoid partial JSON reads by the server
                temp_path = CALIBRATION_PATH + ".tmp"
                with open(temp_path, "w") as f:
                    json.dump(data, f, indent=2)
                os.replace(temp_path, CALIBRATION_PATH)
            except Exception as e:
                pass

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("s"):
            print("Force saved calibration locally! Server will hot-reload.")
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
