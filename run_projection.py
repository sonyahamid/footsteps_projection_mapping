import cv2
import time
import random
from projection_mapping import ProjectionMapper
from udp_receiver import FootstepReceiver

def main():
    mapper = ProjectionMapper("calibration.json") # Updated to existing calibration
    receiver = FootstepReceiver(port=7000)

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
    cv2.namedWindow("projector", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    selected_pid = None
    active_history_paths = {}  # pid -> {"path": [...], "fade": 1.0, "fade_start": None}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- DEBUG WINDOW ---
        debug = frame.copy()

        trails = receiver.get_trails()
        matched = receiver.get_matched_paths()

        # draw footsteps in camera space
        for pid, pts in trails.items():
            for (nx, ny) in pts:
                cx = int(nx * frame.shape[1])
                cy = int(ny * frame.shape[0])
                cv2.circle(debug, (cx, cy), 6, (0,255,0), -1)

        cv2.imshow("debug", debug)

        # --- PROJECTOR WINDOW ---
        # Convert normalized → floor coords
        person_trails_floor = []
        if trails:
            if selected_pid not in trails:
                selected_pid = random.choice(list(trails.keys()))
            
            pts = trails[selected_pid]
            # Convert normalized camera coordinates to pixel coordinates, then to floor space
            floor_pts = [mapper.cam_to_floor((p[0] * frame.shape[1], p[1] * frame.shape[0])) for p in pts]
            person_trails_floor.append(floor_pts)
        else:
            selected_pid = None

        # Update active history paths
        current_time = time.time()
        
        # 1. Update matched path for the selected_pid (if available)
        if selected_pid is not None and selected_pid in matched:
            active_history_paths[selected_pid] = {
                "path": matched[selected_pid],
                "fade": 1.0,
                "fade_start": None
            }
            
        # 2. Check all active history paths to see if their pid has left the frame
        pids_to_del = []
        for pid, data in active_history_paths.items():
            if pid not in trails:
                # Walker has left the frame, start fading
                if data["fade_start"] is None:
                    data["fade_start"] = current_time
                
                # Fade out completely over 3 seconds
                elapsed = current_time - data["fade_start"]
                fade_duration = 3.0
                data["fade"] = max(0.0, 1.0 - (elapsed / fade_duration))
                
                if data["fade"] <= 0.0:
                    pids_to_del.append(pid)
            else:
                # Walker is back in frame? Reset fade.
                data["fade_start"] = None
                data["fade"] = 1.0
                
        for pid in pids_to_del:
            del active_history_paths[pid]

        # Prepare matched trails for projection mapping
        matched_trails_floor = []
        for pid, data in active_history_paths.items():
            if 'pts' in data["path"]:
                path_pts = data["path"]["pts"]
                age_str = f"Seen {int(data['path']['age_secs'])}s ago"
            else:
                path_pts = data["path"]
                age_str = ""

            floor_pts = [mapper.cam_to_floor((p[0] * frame.shape[1], p[1] * frame.shape[0])) for p in path_pts]
            if floor_pts:
                matched_trails_floor.append({
                    "trail": floor_pts,
                    "fade": data["fade"],
                    "age_str": age_str
                })

        proj_frame = mapper.render_projector_frame(
            person_trails=person_trails_floor,
            matched_trails=matched_trails_floor
        )
        cv2.imshow("projector", proj_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
