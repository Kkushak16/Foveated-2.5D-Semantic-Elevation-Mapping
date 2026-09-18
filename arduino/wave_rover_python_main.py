import time
import json
import urllib.request
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from arduino.app_utils import App, Bridge

rover_ip = "192.168.4.1"

def get_rover_url():
    return f"http://{rover_ip}"

def send_rover_cmd(inputA, inputB, inputC):
    try:
        url = f"{get_rover_url()}/cmd?inputA={inputA}&inputB={inputB}&inputC={inputC}"
        req = urllib.request.Request(url, headers={'User-Agent': 'UnoQBridge'})
        with urllib.request.urlopen(req, timeout=0.35) as resp:
            return True
    except Exception:
        return False

def do_wiggle():
    # 1. Show Left Arrow & Rotate Left
    try:
        Bridge.call('show_arrow', 2)
    except Exception:
        pass
    send_rover_cmd(1, -0.85, 0.85)
    time.sleep(0.35)

    # 2. Show Right Arrow & Rotate Right
    try:
        Bridge.call('show_arrow', 3)
    except Exception:
        pass
    send_rover_cmd(1, 0.85, -0.85)
    time.sleep(0.35)

    # 3. Stop
    send_rover_cmd(1, 0, 0)
    time.sleep(0.12)

    # 4. Show bright OK on 13x8 Matrix
    try:
        Bridge.call('show_ok')
    except Exception:
        pass

class RoverHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        global rover_ip
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path == '/ok':
            try:
                Bridge.call('show_ok')
            except Exception:
                pass
            self._json({'status': 'ok', 'matrix': 'ok'})

        elif parsed.path == '/off':
            try:
                Bridge.call('turn_off')
            except Exception:
                pass
            self._json({'status': 'ok', 'matrix': 'off'})

        elif parsed.path == '/wiggle':
            threading.Thread(target=do_wiggle, daemon=True).start()
            self._json({'status': 'ok', 'action': 'wiggle_started'})

        elif parsed.path == '/set_rover_ip':
            new_ip = qs.get('ip', [rover_ip])[0].strip()
            if new_ip:
                rover_ip = new_ip
            self._json({'status': 'ok', 'rover_ip': rover_ip})

        elif parsed.path == '/arrow':
            dir_code = int(qs.get('dir', [0])[0])
            try:
                Bridge.call('show_arrow', dir_code)
            except Exception:
                pass
            self._json({'status': 'ok', 'dir': dir_code})

        elif parsed.path == '/drive':
            l = float(qs.get('left', [0.0])[0])
            r = float(qs.get('right', [0.0])[0])
            send_rover_cmd(1, l, r)
            if l > 0 and r > 0:
                d = 0 # UP
            elif l < 0 and r < 0:
                d = 1 # DOWN
            elif l < r:
                d = 2 # LEFT
            elif l > r:
                d = 3 # RIGHT
            else:
                d = -1
            if d >= 0:
                try:
                    Bridge.call('show_arrow', d)
                except Exception:
                    pass
            elif l == 0 and r == 0:
                try:
                    Bridge.call('show_ok')
                except Exception:
                    pass
            self._json({'status': 'ok', 'left': l, 'right': r})

        elif parsed.path == '/stop':
            send_rover_cmd(1, 0, 0)
            try:
                Bridge.call('show_ok')
            except Exception:
                pass
            self._json({'status': 'ok', 'action': 'stop'})

        elif parsed.path == '/deviceInfo':
            try:
                req = urllib.request.Request(f'{get_rover_url()}/deviceInfo', headers={'User-Agent': 'UnoQBridge'})
                with urllib.request.urlopen(req, timeout=0.8) as resp:
                    data = resp.read()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(data)
                    return
            except Exception as e:
                self._json({'V': 11.2, 'status': 'rover_offline', 'error': str(e)})

        else:
            self._json({'status': 'ok', 'service': 'unoq_rover_daemon', 'rover_ip': rover_ip})

    def _json(self, data, code=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

def start_http():
    srv = HTTPServer(('0.0.0.0', 7600), RoverHandler)
    srv.serve_forever()

threading.Thread(target=start_http, daemon=True).start()

# Initial OK on matrix upon boot
time.sleep(1.0)
try:
    Bridge.call('show_ok')
except Exception:
    pass

def loop():
    time.sleep(5)

App.run(user_loop=loop)
