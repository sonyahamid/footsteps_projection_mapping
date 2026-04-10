import asyncio
import json
import logging
import os
import threading
import http.server
import socketserver
import websockets
import time

from udp_receiver import FootstepReceiver
from projection_mapping import calibrate_from_known_points, ProjectionMapper

PORT_HTTP = 8000
PORT_WS = 8001
CALIBRATION_FILE = "calibration.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ProjectionServer")

HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/footstep_history.json")

def ensure_calibration_exists():
    if not os.path.exists(CALIBRATION_FILE):
        logger.info(f"{CALIBRATION_FILE} not found. Generating default calibration.")
        calibrate_from_known_points(
            cam_pts=[[110, 90], [530, 90], [590, 430], [50, 430]],
            proj_pts=[(160, 80), (1760, 80), (1840, 1000), (80, 1000)],
            out_path=CALIBRATION_FILE
        )

# --- HTTP Server ---
def run_http_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    Handler = http.server.SimpleHTTPRequestHandler
    # Disable caching for development
    class NoCacheHandler(Handler):
        def do_GET(self):
            if self.path == '/api/random_path':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
                self.end_headers()
                
                try:
                    with open(HISTORY_FILE, 'r') as f:
                        history = json.load(f)
                    if not history:
                        self.wfile.write(b"null")
                        return
                        
                    import random
                    path_data = random.choice(history)
                    self.wfile.write(json.dumps(path_data).encode())
                except FileNotFoundError:
                    self.wfile.write(b"null")
                return
            elif self.path == '/api/history':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
                self.end_headers()
                try:
                    with open(HISTORY_FILE, 'r') as f:
                        self.wfile.write(f.read().encode())
                except FileNotFoundError:
                    self.wfile.write(b"[]")
                return
            else:
                super().do_GET()

        def end_headers(self):
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
            super().end_headers()

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("", PORT_HTTP), NoCacheHandler) as httpd:
        logger.info(f"Serving HTTP on http://localhost:{PORT_HTTP}/web_ui")
        httpd.serve_forever()

CLIENTS = set()

async def ws_handler(websocket, path=None): # path parameter needed for older websockets API compat
    CLIENTS.add(websocket)
    try:
        # Load and send calibration data immediately (H_proj matrix, etc)
        with open(CALIBRATION_FILE) as f:
            calib = json.load(f)
            await websocket.send(json.dumps({
                "type": "calibration",
                "data": calib
            }))
            
        await websocket.wait_closed()
    finally:
        CLIENTS.remove(websocket)

async def broadcast_data(data):
    if not CLIENTS:
        return
    message = json.dumps(data)
    await asyncio.gather(
        *(client.send(message) for client in CLIENTS),
        return_exceptions=True
    )

async def process_and_save_trail(pid, data, mapper, CAM_W, CAM_H):
    """Processes a path to floor coordinates and appends to footstep_history.json."""
    if len(data["path"]) < 10:
        # Ignore very short trails
        return
        
    floor_pts = []
    for p in data["path"]:
        fx, fy = mapper.cam_to_floor((p[0] * CAM_W, p[1] * CAM_H))
        floor_pts.append({"x": fx, "y": fy})
        
    if not os.path.exists(os.path.dirname(HISTORY_FILE)):
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []
            
    history.append(floor_pts)
    # Keep last N trails
    history = history[-100:]
    
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)
    logger.info(f"Saved path for pid {pid} with {len(floor_pts)} points")

async def data_loop():
    """Reads UDP stream, transforms state, and broadcasts to WebSockets."""
    receiver = FootstepReceiver(port=7000)
    
    # Load basic configuration based on python mapper logic
    mapper = ProjectionMapper(CALIBRATION_FILE)
    CAM_W, CAM_H = 640, 480

    last_calib_mtime = os.path.getmtime(CALIBRATION_FILE)

    active_history_paths = {}  # pid -> {"path": [...], "fade": 1.0, "fade_start": None}

    while True:
        # Match python projection map 60fps ~ 0.016s sleep
        await asyncio.sleep(1/60.0)
        
        try:
            current_mtime = os.path.getmtime(CALIBRATION_FILE)
            if current_mtime != last_calib_mtime:
                last_calib_mtime = current_mtime
                mapper = ProjectionMapper(CALIBRATION_FILE)
                with open(CALIBRATION_FILE) as f:
                    calib = json.load(f)
                    logger.info("Calibration file updated, broadcasting new parameters...")
                    await broadcast_data({"type": "calibration", "data": calib})
        except Exception:
            pass
        
        trails = receiver.get_trails()
        matched = receiver.get_matched_paths()

        person_trails = {}
        if trails:
            # Map all active live trails into floor space
            for pid, pts in trails.items():
                floor_pts = []
                for p in pts:
                    fx, fy = mapper.cam_to_floor((p[0] * CAM_W, p[1] * CAM_H))
                    floor_pts.append({"x": fx, "y": fy})
                person_trails[pid] = floor_pts

        current_time = time.time()
        
        # Add any new matched walkers
        for pid in matched:
            if pid in trails:
                if pid not in active_history_paths:
                    active_history_paths[pid] = {
                        "path": matched[pid]['pts'],
                        "fade": 1.0,
                        "fade_start": None,
                        "age_str": f"{matched[pid]['age_secs']:.1f}s"
                    }
                else:
                    active_history_paths[pid]["path"] = matched[pid]['pts']
                    active_history_paths[pid]["fade"] = 1.0
                    active_history_paths[pid]["fade_start"] = None
                    active_history_paths[pid]["age_str"] = f"{matched[pid]['age_secs']:.1f}s"
            
        pids_to_del = []
        for pid, data in active_history_paths.items():
            if pid not in trails:
                # Walker left frame
                if data["fade_start"] is None:
                    data["fade_start"] = current_time
                    
                elapsed = current_time - data["fade_start"]
                fade_duration = 1.5
                data["fade"] = max(0.0, 1.0 - (elapsed / fade_duration))
                
                if data["fade"] <= 0.0:
                    pids_to_del.append(pid)
                    # When fading out is fully complete, save the path
                    await process_and_save_trail(pid, data, mapper, CAM_W, CAM_H)
            else:
                data["fade_start"] = None
                data["fade"] = 1.0
                
        for pid in pids_to_del:
            del active_history_paths[pid]

        matched_trails = {}
        # We NO LONGER broadcast paths to WS. Let frontend drive playback.
        
        payload = {
            "type": "frame",
            "person_trails": {},
            "matched_trails": {}
        }
        
        await broadcast_data(payload)

async def main():
    ensure_calibration_exists()
    
    # Start HTTP server background thread
    threading.Thread(target=run_http_server, daemon=True).start()
    
    # Optional: Automatically push the web UI open
    import webbrowser
    # Give the HTTP server a tiny bit to bind
    threading.Timer(1.0, lambda: webbrowser.open('http://localhost:8000/web_ui/')).start()
    
    # Start WS server and Data loop concurrently
    logger.info(f"Starting WebSocket server on port {PORT_WS}...")
    # Note: websockets serve returns a server object.
    async with websockets.serve(ws_handler, "localhost", PORT_WS, reuse_address=True, reuse_port=True):
        await data_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down servers...")
