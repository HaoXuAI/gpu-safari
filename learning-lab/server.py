#!/usr/bin/env python3
"""Serve the guided lab and its local-only GPU execution API."""

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from runtime.api import handle_api_request


LAB_ROOT = Path(__file__).parent


def is_trusted_origin(host: str, origin: str | None) -> bool:
    hostname = host.split(":", 1)[0]
    return hostname in {"127.0.0.1", "localhost"} and origin == f"http://{host}"


class LearningLabHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(LAB_ROOT), **kwargs)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path.startswith("/api/"):
            status, payload = handle_api_request("GET", self.path, None)
            self._send_json(status, payload)
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self._send_json(404, {"error": "not found"})
            return
        if not is_trusted_origin(self.headers.get("Host", ""), self.headers.get("Origin")):
            self._send_json(403, {"error": "untrusted request origin"})
            return
        if self.headers.get_content_type() != "application/json":
            self._send_json(415, {"error": "Content-Type must be application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 8192:
                raise ValueError("invalid body length")
            body = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "invalid JSON body"})
            return
        status, payload = handle_api_request("POST", self.path, body)
        self._send_json(status, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    address = ("127.0.0.1", args.port)
    print(f"GPU Safari learning lab: http://{address[0]}:{address[1]}")
    ThreadingHTTPServer(address, LearningLabHandler).serve_forever()


if __name__ == "__main__":
    main()
