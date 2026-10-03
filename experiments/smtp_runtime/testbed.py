"""A local SMTP service with a durable, independently read capture store.

This is a TCP SMTP submission/capture endpoint, not a recipient delivery service.
It intentionally has no relay, upstream connection, authentication or TLS.
"""
from __future__ import annotations

from email import policy
from email.parser import BytesParser
import json
import os
from pathlib import Path
import re
import socketserver
import threading
import time
import uuid

from .adapter import COMMIT_BINDING_SCHEMA, decoded, sha


def durable_write(path: Path, data: bytes):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class SMTPHandler(socketserver.StreamRequestHandler):
    def reply(self, line: bytes):
        self.wfile.write(line + b"\r\n")
        self.wfile.flush()

    def handle(self):
        self.request.settimeout(10)
        self.reply(b"220 obligate.test capture SMTP")
        sender = None
        targets = []
        while True:
            line = self.rfile.readline(65537)
            if not line:
                return
            if len(line) > 65536:
                self.reply(b"500 line too long")
                return
            text = line.decode("utf-8", errors="replace").strip()
            command = text.split(" ", 1)[0].upper()
            if command in ("EHLO", "HELO"):
                self.reply(b"250 obligate.test")
            elif command == "MAIL":
                match = re.fullmatch(r"MAIL FROM:\s*<([^<>]+)>(?: .*)?", text, re.I)
                if not match or not match[1].endswith("@obligate.test"):
                    self.reply(b"550 capture sender domain required")
                    continue
                sender, targets = match[1], []
                self.reply(b"250 sender accepted")
            elif command == "RCPT":
                match = re.fullmatch(r"RCPT TO:\s*<([^<>]+)>(?: .*)?", text, re.I)
                if sender is None or not match or not match[1].endswith("@obligate.test"):
                    self.reply(b"550 capture recipient domain required")
                    continue
                targets.append(match[1])
                self.reply(b"250 recipient accepted")
            elif command == "DATA":
                if not sender or not targets:
                    self.reply(b"503 envelope required")
                    continue
                self.reply(b"354 end with dot")
                chunks = []
                total = 0
                while True:
                    chunk = self.rfile.readline(65537)
                    if not chunk:
                        return
                    if chunk in (b".\r\n", b".\n"):
                        break
                    if chunk.startswith(b".."):
                        chunk = chunk[1:]
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > 8 * 1024 * 1024:
                        self.reply(b"552 capture limit exceeded")
                        return
                raw = b"".join(chunks)
                parsed = BytesParser(policy=policy.default).parsebytes(raw)
                trace = str(parsed.get("X-ObliGate-Trace", ""))
                if not re.fullmatch(r"[0-9a-f]{32}", trace):
                    self.reply(b"554 invalid capture trace")
                    continue
                capture_id = uuid.uuid4().hex
                stem = trace + "-" + capture_id
                spool = self.server.spool
                durable_write(spool / (stem + ".eml"), raw)
                record = {"trace": trace, "capture_id": capture_id,
                          "envelope_sender": sender, "envelope_recipients": sorted(set(targets)),
                          "raw_filename": stem + ".eml", "raw_sha256": sha(raw),
                          "captured_ns": time.perf_counter_ns()}
                durable_write(spool / (stem + ".json"), json.dumps(record).encode())
                self.reply(b"250 persisted " + capture_id.encode())
                sender, targets = None, []
            elif command == "RSET":
                sender, targets = None, []
                self.reply(b"250 reset")
            elif command == "NOOP":
                self.reply(b"250 ok")
            elif command == "QUIT":
                self.reply(b"221 bye")
                return
            else:
                self.reply(b"502 command unsupported")


class CaptureServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, spool: Path):
        self.spool = Path(spool)
        self.spool.mkdir(parents=True, exist_ok=True)
        # Ephemeral loopback port permits parallel invocations without fixed-port collisions.
        super().__init__(("127.0.0.1", 0), SMTPHandler)
        self.thread = None

    def __enter__(self):
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()
        return self

    @property
    def port(self):
        return self.server_address[1]

    def __exit__(self, *exc):
        self.shutdown()
        self.server_close()
        self.thread.join(timeout=5)

    def manifest(self):
        return {"service": "ObliGate portable SMTP capture service", "version": "2",
                "commit_binding_schema": COMMIT_BINDING_SCHEMA,
                "listen": "127.0.0.1", "port": self.port,
                "relay_configured": False, "recipient_domain": "obligate.test",
                "mailbox_semantics": "durable SMTP capture, not downstream mailbox delivery",
                "network_sandbox": False}


def observe(spool: Path, trace: str) -> dict:
    # Observation reads service-owned files and MIME bytes, never gate/adapter results.
    if not re.fullmatch(r"[0-9a-f]{32}", trace):
        raise ValueError("invalid observer trace")
    messages = []
    for path in sorted(Path(spool).glob(trace + "-*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        raw = (Path(spool) / item["raw_filename"]).read_bytes()
        if sha(raw) != item["raw_sha256"]:
            raise RuntimeError("capture store hash mismatch")
        item["decoded"] = decoded(raw)
        messages.append(item)
    return {"messages": messages, "observed_ns": time.perf_counter_ns(),
            "status": "complete", "scope": "local durable capture store"}
