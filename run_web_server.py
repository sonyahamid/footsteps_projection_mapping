import asyncio
import json
import websockets
import http.server
import socketserver
import threading
import os
import time

PORT_HTTP = 8000
PORT_WS = 8001

# --- HTTP Server for serving Web UI ---
def run_http_server():
    os.chdir(os.path.join(os.path.dirname(__file__), "web_ui"))
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT_HTTP), Handler) as httpd:
        print(f"Serving HTTP on port {PORT_HTTP}")
        httpd.serve_forever()

if __name__ == "__main__":
    t = threading.Thread(target=run_http_server, daemon=True)
    t.start()
    
    print("Test http server running.")

