import cv2
from projection_mapping import ProjectionMapper
from udp_receiver import FootstepReceiver

import random

def main():
    mapper = ProjectionMapper("calibration.json")
    receiver = FootstepReceiver(port=7000)

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
    cv2.namedWindow("projector", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    selected_pid = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- DEBUG WINDOW ---
        debug = frame.copy()

        trails = receiver.get_trails()

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
            floor_pts = [(p[0] * mapper.floor_w, p[1] * mapper.floor_h) for p in pts]
            person_trails_floor.append(floor_pts)
        else:
            selected_pid = None

        proj_frame = mapper.render_projector_frame(person_trails=person_trails_floor)
        cv2.imshow("projector", proj_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
