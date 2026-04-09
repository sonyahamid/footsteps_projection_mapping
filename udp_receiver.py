import socket
import threading
import time

class FootstepReceiver:
    def __init__(self, port=7000):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Allow reusing the port for UDP to prevent "Address already in use" crashes upon restart
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.sock.bind(("0.0.0.0", port))
        self.lock = threading.Lock()
        self.trails = {}  # person_id -> list of (x,y)
        self.matched_paths = {} # person_id -> list of (x,y)
        self.last_seen = {} # person_id -> timestamp

        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        while True:
            data, _ = self.sock.recvfrom(4096)
            parts = data.decode().strip().split()
            if not parts:
                continue

            if parts[0] == "MATCH":
                pid = int(parts[1])
                path_len = int(parts[2])
                age_secs = float(parts[3])
                pts = []
                idx = 4
                for _ in range(path_len):
                    if idx + 3 < len(parts):
                        x = float(parts[idx])
                        y = float(parts[idx+1])
                        pts.append((x, y))
                        idx += 4
                with self.lock:
                    self.matched_paths[pid] = {'pts': pts, 'age_secs': age_secs}
            else:
                x = float(parts[0])
                y = float(parts[1])
                pid = int(parts[4])

                with self.lock:
                    if pid not in self.trails:
                        self.trails[pid] = []
                    self.trails[pid].append((x, y))

                    # keep last N points
                    self.trails[pid] = self.trails[pid][-40:]
                    self.last_seen[pid] = time.time()

    def get_trails(self):
        with self.lock:
            # Clean up old trails (walker out of frame for 1s)
            current_time = time.time()
            active_pids = []
            for pid, last_t in list(self.last_seen.items()):
                if current_time - last_t > 1.0:
                    self.trails.pop(pid, None)
                    self.last_seen.pop(pid, None)
                    self.matched_paths.pop(pid, None)
                else:
                    active_pids.append(pid)
                    
            return {pid: pts[:] for pid, pts in self.trails.items() if pid in active_pids}

    def get_matched_paths(self):
        with self.lock:
            # Only return matches for paths that are still active
            return {pid: data for pid, data in self.matched_paths.items() if pid in self.trails}
