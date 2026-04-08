"""
Projection Mapping Pipeline
============================
Two homographies:
  H_cam  : camera image space  -> flat floor space
  H_proj : flat floor space    -> projector output space

Usage
-----
STEP 1 – Calibrate (run once, saves calibration.json):
    python projection_mapping.py --calibrate --cam 0 --proj-w 1920 --proj-h 1080

STEP 2 – Runtime (import and call from your tracker):
    from projection_mapping import ProjectionMapper
    mapper = ProjectionMapper("calibration.json")

    # Convert a raw camera coordinate to floor space:
    floor_pt = mapper.cam_to_floor((cx, cy))

    # Get the full projector frame to display (call in your render loop):
    proj_frame = mapper.render_projector_frame(floor_pts=[(fx, fy), ...])

Dependencies: opencv-python, numpy
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from PIL import Image

class GifAnimator:
    """Pre-loads a GIF and vends OpenCV-compatible frames by index."""

    def __init__(self, gif_path, size=None):
        """
        gif_path : path to foot1.gif
        size     : (w, h) to resize each frame, or None to keep original
        """
        gif = Image.open(gif_path)
        self.frames = []
        self.durations = []  # ms per frame (honours GIF timing if you need it)

        try:
            while True:
                frame = gif.copy().convert("RGBA")
                if size:
                    frame = frame.resize(size, Image.LANCZOS)
                # OpenCV uses BGR(A), Pillow gives RGBA
                bgra = cv2.cvtColor(np.array(frame), cv2.COLOR_RGBA2BGRA)
                self.frames.append(bgra)
                self.durations.append(gif.info.get("duration", 100))
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass  # end of frames

        self.n = len(self.frames)

    def get_frame(self, t):
        """Return the BGRA frame for global tick t (wraps automatically)."""
        return self.frames[t % self.n]
    

# ---------------------------------------------------------------------------
# Homography helpers
# ---------------------------------------------------------------------------

def compute_homography(src_pts, dst_pts):
    """Compute homography from 4 src points to 4 dst points."""
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def warp_point(H, pt):
    """Apply homography H to a single (x, y) point."""
    p = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
    result = cv2.perspectiveTransform(p, H)
    return (float(result[0][0][0]), float(result[0][0][1]))


def warp_frame(H, frame, out_w, out_h):
    """Warp an entire image/frame using homography H."""
    return cv2.warpPerspective(frame, H, (out_w, out_h))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class Calibrator:
    """
    Interactive calibration.
    Click the 4 tape-marker corners in order: TL, TR, BR, BL.
    Do this twice: once in the camera view, once in the projector view.
    """

    def __init__(self, cam_index, proj_w, proj_h, floor_w=1000, floor_h=1000):
        self.cam_index = cam_index
        self.proj_w = proj_w
        self.proj_h = proj_h
        self.floor_w = floor_w
        self.floor_h = floor_h

        self.cam_pts = []
        self.proj_pts = []

    def _cam_mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.cam_pts) < 4:
            self.cam_pts.append((x, y))
            print(f"  cam pt {len(self.cam_pts)}/4: ({x}, {y})")

    def calibrate_camera(self):
        print("\n=== CAMERA CALIBRATION ===")
        print("Click the 4 tape corners in order: TL, TR, BR, BL")
        cap = cv2.VideoCapture(self.cam_index)
        cv2.namedWindow("camera calibration")
        cv2.setMouseCallback("camera calibration", self._cam_mouse_cb)

        colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]
        labels = ["TL","TR","BR","BL"]

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            display = frame.copy()

            for i, pt in enumerate(self.cam_pts):
                cv2.circle(display, pt, 8, colors[i], -1)
                cv2.putText(display, labels[i], (pt[0]+10, pt[1]+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

            if len(self.cam_pts) == 4:
                pts_arr = np.array(self.cam_pts, np.int32).reshape((-1,1,2))
                cv2.polylines(display, [pts_arr], True, (255,255,255), 1)
                cv2.putText(display, "Press ENTER to confirm, R to reset",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            else:
                cv2.putText(display, f"Click corner {len(self.cam_pts)+1}/4 ({labels[len(self.cam_pts)]})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            cv2.imshow("camera calibration", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and len(self.cam_pts) == 4:  
                break
            if key == ord('r'):
                self.cam_pts = []

        cap.release()
        cv2.destroyAllWindows()

    # -- projector calibration (claude) -----------------------------------------------

    def _proj_mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.proj_pts) < 4:
            self.proj_pts.append((x, y))
            print(f"  proj pt {len(self.proj_pts)}/4: ({x}, {y})")

    def calibrate_projector(self):
        print("\n=== PROJECTOR CALIBRATION ===")
        print("A pattern will be shown. Click the 4 target circles: TL, TR, BR, BL")
        print("(Send this window to your projector display)")

        # Build a reference pattern with target circles
        pattern = np.zeros((self.proj_h, self.proj_w, 3), dtype=np.uint8)
        margin = 80
        targets = [
            (margin,               margin),
            (self.proj_w - margin, margin),
            (self.proj_w - margin, self.proj_h - margin),
            (margin,               self.proj_h - margin),
        ]
        colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]
        labels = ["TL","TR","BR","BL"]

        for i, (tx, ty) in enumerate(targets):
            cv2.circle(pattern, (tx, ty), 20, colors[i], 2)
            cv2.circle(pattern, (tx, ty), 3, colors[i], -1)
            cv2.putText(pattern, labels[i], (tx+25, ty+6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)

        # Draw crosshair lines
        cv2.line(pattern, (self.proj_w//2, 0), (self.proj_w//2, self.proj_h), (40,40,40), 1)
        cv2.line(pattern, (0, self.proj_h//2), (self.proj_w, self.proj_h//2), (40,40,40), 1)

        cv2.namedWindow("projector output", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("projector output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("projector output", pattern)
        cv2.setMouseCallback("projector output", self._proj_mouse_cb)

        print("Click the 4 circles in order. Press ENTER when done, R to reset.")
        while True:
            display = pattern.copy()
            for i, pt in enumerate(self.proj_pts):
                cv2.circle(display, pt, 10, (255,255,255), 2)
            if len(self.proj_pts) == 4:
                cv2.putText(display, "Press ENTER to confirm",
                            (self.proj_w//2 - 150, self.proj_h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("projector output", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and len(self.proj_pts) == 4:
                break
            if key == ord('r'):
                self.proj_pts = []

        cv2.destroyAllWindows()

    # -- save (claude) ----------------------------------------------------------------

    def run_and_save(self, out_path="calibration.json"):
        self.calibrate_camera()
        self.calibrate_projector()

        # Floor space is a normalised square (floor_w x floor_h).
        # Camera homography maps cam_pts -> floor corners.
        # Projector homography maps floor corners -> proj_pts.
        floor_corners = [
            (0,               0),
            (self.floor_w,    0),
            (self.floor_w,    self.floor_h),
            (0,               self.floor_h),
        ]

        H_cam  = compute_homography(self.cam_pts, floor_corners)
        H_proj = compute_homography(floor_corners, self.proj_pts)

        data = {
            "cam_pts":       self.cam_pts,
            "proj_pts":      self.proj_pts,
            "floor_corners": floor_corners,
            "floor_w":       self.floor_w,
            "floor_h":       self.floor_h,
            "proj_w":        self.proj_w,
            "proj_h":        self.proj_h,
            "H_cam":         H_cam.tolist(),
            "H_proj":        H_proj.tolist(),
        }

        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nCalibration saved to {out_path}")
        return data


# ---------------------------------------------------------------------------
# Runtime mapper  
# ---------------------------------------------------------------------------

class ProjectionMapper:
    """
    Load calibration and use it at runtime.

    Typical integration with your tracker
    --------------------------------------
    from projection_mapping import ProjectionMapper

    mapper = ProjectionMapper("calibration.json")

    # Your tracker gives you a camera-space coordinate, e.g. (320, 240).
    floor_pt = mapper.cam_to_floor((cam_x, cam_y))

    # Build a list of floor-space points you want to animate (footstep trail, etc.)
    proj_frame = mapper.render_projector_frame(floor_pts=[floor_pt, ...])

    # Send proj_frame to your projector window / output stream.
    cv2.imshow("projector", proj_frame)
    """

    def __init__(self, calibration_path="calibration.json"):
        with open(calibration_path) as f:
            data = json.load(f)

        self.floor_w  = data["floor_w"]
        self.floor_h  = data["floor_h"]
        self.proj_w   = data["proj_w"]
        self.proj_h   = data["proj_h"]
        self.H_cam    = np.array(data["H_cam"],  dtype=np.float64)
        self.H_proj   = np.array(data["H_proj"], dtype=np.float64)

        self.animator = GifAnimator("white_foot.gif", size=(27,54))
        self.tick = 0  # global frame counter, increment each render call
    # -- coordinate transforms -----------------------------------------------

    def cam_to_floor(self, cam_pt):
        """Map a single camera-space (x,y) -> floor-space (x,y)."""
        return warp_point(self.H_cam, cam_pt)

    def floor_to_proj(self, floor_pt):
        """Map a single floor-space (x,y) -> projector-space (x,y)."""
        return warp_point(self.H_proj, floor_pt)

    def cam_to_proj(self, cam_pt):
        """Shortcut: camera space -> projector space directly."""
        return self.floor_to_proj(self.cam_to_floor(cam_pt))

    def cam_pts_to_floor(self, pts):
        """Map a list of camera-space points to floor space."""
        return [self.cam_to_floor(p) for p in pts]

    # -- rendering -----------------------------------------------------------

    def make_floor_canvas(self):
        """Return a blank black floor-space canvas."""
        return np.zeros((self.floor_h, self.floor_w, 3), dtype=np.uint8)

    # def rotate_image(self, img, angle_deg):
    #     """Rotate a BGRA image by angle_deg around its center, keeping full size."""
    #     h, w = img.shape[:2]
    #     cx, cy = w // 2, h // 2
    #     M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    #     return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
    #                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    def rotate_image(self, img, angle_deg):
        """Rotate a BGRA image without cropping."""
        h, w = img.shape[:2]

        cx, cy = w / 2, h / 2
        M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)

        cos = abs(M[0, 0])
        sin = abs(M[0, 1])

        # compute new bounding dimensions
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # adjust rotation matrix to center the image
        M[0, 2] += (new_w / 2) - cx
        M[1, 2] += (new_h / 2) - cy

        return cv2.warpAffine(
            img,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

    def do_stuff(self, floor_canvas, floor_pts, trail_pts=None):
        all_pts = (trail_pts if trail_pts else []) + list(floor_pts)
        n = len(all_pts)

        step_interval = 6
        steps_visible = min((self.tick // step_interval) + 1, n)
        visible_pts = all_pts[:steps_visible]

        fade_duration = 80

        for i, pt in enumerate(visible_pts):
            step_placed_at = i * step_interval
            ticks_since_placed = self.tick - step_placed_at

            if i == steps_visible - 1:
                fade = 1.0
            else:
                fade = max(0.0, 1.0 - (ticks_since_placed / fade_duration))

            if fade == 0.0:
                continue

            if i == steps_visible - 1:
                frame_idx = self.animator.n - 1
            else:
                frame_idx = self.tick

            bgra_frame = self.animator.get_frame(frame_idx)

            # --- Compute walking direction angle from neighbouring points ---
            if i < len(visible_pts) - 1:
                nx = visible_pts[i+1][0] - pt[0]
                ny = visible_pts[i+1][1] - pt[1]
            elif i > 0:
                nx = pt[0] - visible_pts[i-1][0]
                ny = pt[1] - visible_pts[i-1][1]
            else:
                nx, ny = 1, 0

            angle_deg = np.degrees(np.arctan2(ny, nx))

            # Rotate the GIF frame to align with walking direction
            bgra_rotated = self.rotate_image(bgra_frame, angle_deg+90)

            fh, fw = bgra_rotated.shape[:2]
            alpha = bgra_rotated[:, :, 3:4] / 255.0
            bgr   = bgra_rotated[:, :, :3].astype(np.float32)

            # Left/right alternating offset perpendicular to walking direction
            length = max((nx**2 + ny**2) ** 0.5, 1)
            px, py = -ny / length, nx / length
            side = 1 if i % 2 == 0 else -1
            offset = 18
            cx = int(pt[0]) + int(px * offset * side)
            cy = int(pt[1]) + int(py * offset * side)

            # Composite
            x0, y0 = cx - fw // 2, cy - fh // 2
            x1, y1 = x0 + fw, y0 + fh

            sx0 = max(0, -x0);  sx1 = sx0 + (min(x1, floor_canvas.shape[1]) - max(x0, 0))
            sy0 = max(0, -y0);  sy1 = sy0 + (min(y1, floor_canvas.shape[0]) - max(y0, 0))
            dx0 = max(x0, 0);   dx1 = dx0 + (sx1 - sx0)
            dy0 = max(y0, 0);   dy1 = dy0 + (sy1 - sy0)

            if sx1 <= sx0 or sy1 <= sy0:
                continue

            src = bgr  [sy0:sy1, sx0:sx1]
            a   = alpha[sy0:sy1, sx0:sx1] * fade
            dst = floor_canvas[dy0:dy1, dx0:dx1].astype(np.float32)

            floor_canvas[dy0:dy1, dx0:dx1] = (src * a + dst * (1 - a)).astype(np.uint8)

        return floor_canvas
    
    
    def render_projector_frame(self, floor_pts=None, trail_pts=None):
        """
        Full pipeline: blank canvas -> draw animations -> warp to projector space.
        Returns an (proj_h x proj_w x 3) numpy array ready to display/stream.
        """
        floor_canvas = self.make_floor_canvas()

        if floor_pts:
            floor_canvas = self.do_stuff(floor_canvas, floor_pts, trail_pts)

        self.tick += 1
        proj_frame = warp_frame(self.H_proj, floor_canvas, self.proj_w, self.proj_h)
        return proj_frame

    def rectify_cam_frame(self, cam_frame):
        """
        Warp a raw camera frame into floor (top-down) space.
        Useful for debugging — not needed for the output pipeline.
        """
        return warp_frame(self.H_cam, cam_frame, self.floor_w, self.floor_h)


# ---------------------------------------------------------------------------
# Manual calibration helper (no webcam needed — type your points directly)
# ---------------------------------------------------------------------------

def calibrate_from_known_points(
    cam_pts,        # 4x (x,y) in camera pixel space,  order: TL TR BR BL
    proj_pts,       # 4x (x,y) in projector pixel space, same order
    floor_w=1000,
    floor_h=1000,
    proj_w=1920,
    proj_h=1080,
    out_path="calibration.json",
):
    """
    Use this if you already know your corner coordinates
    (e.g. measured manually or from a previous calibration session).

    Example
    -------
    calibrate_from_known_points(
        cam_pts  = [(112, 89), (527, 95), (531, 380), (108, 374)],
        proj_pts = [(150, 120), (1770, 120), (1770, 960), (150, 960)],
    )
    """
    floor_corners = [(0,0),(floor_w,0),(floor_w,floor_h),(0,floor_h)]
    H_cam  = compute_homography(cam_pts, floor_corners)
    H_proj = compute_homography(floor_corners, proj_pts)

    data = {
        "cam_pts":       cam_pts,
        "proj_pts":      proj_pts,
        "floor_corners": floor_corners,
        "floor_w":       floor_w,
        "floor_h":       floor_h,
        "proj_w":        proj_w,
        "proj_h":        proj_h,
        "H_cam":         H_cam.tolist(),
        "H_proj":        H_proj.tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Calibration saved to {out_path}")
    return data


# ---------------------------------------------------------------------------
# Demo loop (shows the pipeline end-to-end with a live webcam)
# ---------------------------------------------------------------------------

def demo_loop(calibration_path="calibration.json", cam_index=0):
    mapper = ProjectionMapper(calibration_path)
    cap = cv2.VideoCapture(cam_index)

    # Simulated trail — in real use these come from your tracker
    fake_trail = [(200 + i*20, 300 + i*10) for i in range(15)]

    print("Demo running. Press Q to quit.")
    while True:
        ret, cam_frame = cap.read()
        if not ret:
            break

        # Show rectified floor view (debug)
        floor_view = mapper.rectify_cam_frame(cam_frame)

        # Render projector output with placeholder animation
        proj_out = mapper.render_projector_frame(
            floor_pts=[fake_trail[-1]],
            trail_pts=fake_trail,
        )

        cv2.imshow("floor (rectified)", floor_view)
        cv2.imshow("projector output",  proj_out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true", help="Run interactive calibration")
    parser.add_argument("--demo",      action="store_true", help="Run demo loop after calibration")
    parser.add_argument("--cam",       type=int, default=0,    help="Camera index")
    parser.add_argument("--proj-w",    type=int, default=1920, help="Projector width px")
    parser.add_argument("--proj-h",    type=int, default=1080, help="Projector height px")
    parser.add_argument("--floor-w",   type=int, default=1000, help="Floor canvas width")
    parser.add_argument("--floor-h",   type=int, default=1000, help="Floor canvas height")
    parser.add_argument("--out",       default="calibration.json", help="Output calibration file")
    args = parser.parse_args()

    print(args)
    if args.calibrate:
        cal = Calibrator(args.cam, args.proj_w, args.proj_h, args.floor_w, args.floor_h)
        cal.run_and_save(args.out)

    if args.demo:
        demo_loop(args.out, args.cam)