import socket
import threading

class FootstepReceiver:
    def __init__(self, port=7000):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.lock = threading.Lock()
        self.trails = {}  # person_id -> list of (x,y)

        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        while True:
            data, _ = self.sock.recvfrom(4096)
            parts = data.decode().strip().split()

            x = float(parts[0])
            y = float(parts[1])
            pid = int(parts[2])

            with self.lock:
                if pid not in self.trails:
                    self.trails[pid] = []
                self.trails[pid].append((x, y))

                # keep last N points
                self.trails[pid] = self.trails[pid][-40:]

    def get_trails(self):
        with self.lock:
            return {pid: pts[:] for pid, pts in self.trails.items()}
