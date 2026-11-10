"""Local viewer server with uncached assets and an explicit prediction-file list."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent


class ViewerHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_GET(self):
        if urlsplit(self.path).path == '/api/predictions':
            folder = Path(self.directory) / 'data'
            names = sorted(p.name for p in folder.iterdir()
                           if p.is_file() and p.suffix.lower() == '.csv') if folder.is_dir() else []
            body = json.dumps(names).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8081)
    args = parser.parse_args()
    handler = partial(ViewerHandler, directory=str(ROOT))
    with ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
        print(f'Possession viewer: http://127.0.0.1:{args.port}/', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
