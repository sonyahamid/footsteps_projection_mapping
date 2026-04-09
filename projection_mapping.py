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
from PIL import Image
import random

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
    

# Homography helpers
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

        # cycle through different step animations
        self.animators = [
            GifAnimator("white_foot.gif", size=(27,54)),
            GifAnimator("white_foot2.gif", size=(27,54)),
            GifAnimator("white_foot3.gif", size=(27,54))
        ]

        self.tick = 0  # global frame counter
        
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

    def do_stuff(self, floor_canvas, floor_pts, trail_pts=None, age_label=None):
        all_pts = (trail_pts if trail_pts else []) + list(floor_pts)
        n = len(all_pts)

        for i, pt in enumerate(all_pts):
            if i == n - 1:
                fade = 1.0
            else:
                age_idx = n - 1 - i
                fade = max(0.2, 1.0 - (age_idx * 0.05))

            animator = self.animators[i % len(self.animators)]
            if i == n - 1:
                frame_idx = animator.n - 1
            else:
                frame_idx = self.tick

            bgra_frame = animator.get_frame(frame_idx)

            # compute walking direction angle from neighbouring coords 
            if i < n - 1:
                nx = all_pts[i+1][0] - pt[0]
                ny = all_pts[i+1][1] - pt[1]
            elif i > 0:
                nx = pt[0] - all_pts[i-1][0]
                ny = pt[1] - all_pts[i-1][1]
            else:
                nx, ny = 1, 0

            # normalize path vector
            plen = max((nx**2 + ny**2) ** 0.5, 1e-5)
            nx, ny = nx / plen, ny / plen

            if len(pt) >= 4:
                # we have a hardware direction vector (already in floor space from test_demo)
                dx, dy = pt[2], pt[3]
                dlen = max((dx**2 + dy**2) ** 0.5, 1e-5)
                if dlen > 1e-3:
                    dx, dy = dx / dlen, dy / dlen
                    # average vectors
                    nx, ny = (nx + dx) / 2.0, (ny + dy) / 2.0
            
            angle_deg = np.degrees(np.arctan2(ny, nx))

            # rotate the gif frame to align with walking direction
            bgra_rotated = self.rotate_image(bgra_frame, angle_deg+90)

            fh, fw = bgra_rotated.shape[:2]
            alpha = bgra_rotated[:, :, 3:4] / 255.0
            bgr   = bgra_rotated[:, :, :3].astype(np.float32)

            # left/right alternating offset perpendicular to walking direction
            length = max((nx**2 + ny**2) ** 0.5, 1)
            px, py = -ny / length, nx / length
            side = 1 if i % 2 == 0 else -1
            offset = 18
            cx = int(pt[0]) + int(px * offset * side)
            cy = int(pt[1]) + int(py * offset * side)

            # composite
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

        if age_label and n > 1:
            self._draw_text_along_path(floor_canvas, all_pts, age_label)

        return floor_canvas

    def _draw_text_along_path(self, canvas, pts, text, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.7, color=(200, 200, 200), thickness=2):
        if len(pts) < 2 or not text:
            return

        # Calculate cumulative distances
        dists = [0.0]
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i-1][0]
            dy = pts[i][1] - pts[i-1][1]
            dist = (dx**2 + dy**2)**0.5
            dists.append(dists[-1] + dist)

        total_dist = dists[-1]
        if total_dist == 0:
            return

        # Measure text characters
        char_sizes = [cv2.getTextSize(c, font, scale, thickness)[0] for c in text]
        total_text_width = sum(w for w, h in char_sizes) + len(text) * 4 # 4px padding between chars
        
        # Start distance near the end of the line
        start_dist = max(0, total_dist - total_text_width - 20)
        curr_dist = start_dist
        
        for i, char in enumerate(text):
            cw, ch = char_sizes[i]
            
            # Find the segment that contains curr_dist
            idx = 0
            while idx < len(dists) - 2 and dists[idx+1] < curr_dist:
                idx += 1
                
            p1, p2 = pts[idx], pts[idx+1]
            segment_len = dists[idx+1] - dists[idx]
            if segment_len > 0:
                t = (curr_dist - dists[idx]) / segment_len
            else:
                t = 0
            
            t = max(0, min(1, t))
            cx = p1[0] + t * (p2[0] - p1[0])
            cy = p1[1] + t * (p2[1] - p1[1])
            
            nx = p2[0] - p1[0]
            ny = p2[1] - p1[1]
            angle_deg = np.degrees(np.arctan2(ny, nx))
            
            # Draw char on a single channel patch
            patch_size = int(max(cw, ch) * 2.5) + 10
            patch = np.zeros((patch_size, patch_size), dtype=np.uint8)
            
            text_x = (patch_size - cw) // 2
            text_y = (patch_size + ch) // 2
            cv2.putText(patch, char, (text_x, text_y), font, scale, 255, thickness, cv2.LINE_AA)
            
            # Convert to BGRA
            bgra_patch = np.zeros((patch_size, patch_size, 4), dtype=np.uint8)
            bgra_patch[:, :, 0] = color[0]
            bgra_patch[:, :, 1] = color[1]
            bgra_patch[:, :, 2] = color[2]
            bgra_patch[:, :, 3] = patch
            
            rotated = self.rotate_image(bgra_patch, angle_deg)
            rh, rw = rotated.shape[:2]
            
            # Offset perpendicularly so it's beside the footsteps
            slen = max(segment_len, 1e-5)
            px, py = -ny / slen, nx / slen
            offset = 40
            text_cx = cx + px * offset
            text_cy = cy + py * offset
            
            x0 = int(text_cx - rw // 2)
            y0 = int(text_cy - rh // 2)
            x1 = x0 + rw
            y1 = y0 + rh
            
            canvas_h, canvas_w = canvas.shape[:2]
            sx0 = max(0, -x0); sx1 = sx0 + (min(x1, canvas_w) - max(x0, 0))
            sy0 = max(0, -y0); sy1 = sy0 + (min(y1, canvas_h) - max(y0, 0))
            dx0 = max(x0, 0); dx1 = dx0 + (sx1 - sx0)
            dy0 = max(y0, 0); dy1 = dy0 + (sy1 - sy0)
            
            if sx1 > sx0 and sy1 > sy0:
                src = rotated[sy0:sy1, sx0:sx1]
                alpha = src[:, :, 3:4] / 255.0
                bgr = src[:, :, :3].astype(np.float32)
                dst = canvas[dy0:dy1, dx0:dx1].astype(np.float32)
                canvas[dy0:dy1, dx0:dx1] = (bgr * alpha + dst * (1 - alpha)).astype(np.uint8)
                
            curr_dist += cw + 4

    def render_projector_frame(self, person_trails=None, matched_trails=None):
        """
        Full pipeline: blank canvas -> draw animations -> warp to projector space.
        person_trails: List of point arrays. Each array is a trail for one person.
                       The last point is the current position.
        matched_trails: List of dicts {"trail": [...], "age": "..."}
        Returns an (proj_h x proj_w x 3) numpy array ready to display/stream.
        """
        floor_canvas = self.make_floor_canvas()

        if person_trails:
            for trail in person_trails:
                if trail:
                    # current position is the last element
                    floor_canvas = self.do_stuff(floor_canvas, [trail[-1]], trail[:-1])
                    
        if matched_trails:
            for match in matched_trails:
                if match["trail"]:
                    floor_canvas = self.do_stuff(floor_canvas, [match["trail"][-1]], match["trail"][:-1], age_label=match["age_str"])

        self.tick += 1
        proj_frame = warp_frame(self.H_proj, floor_canvas, self.proj_w, self.proj_h)
        return proj_frame

# Manual calibration helper (no webcam needed — type your points directly)
def calibrate_from_known_points(
    cam_pts, # 4x (x,y) in camera pixel space,  order: TL TR BR BL
    proj_pts, # 4x (x,y) in projector pixel space, same order
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
