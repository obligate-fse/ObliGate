"""Last-hop SMTP binding and explicit redaction controls.

This module never imports fault conditions or expected experiment outcomes.
The transport is deliberately restricted to a loopback capture service.
"""
from __future__ import annotations

from copy import deepcopy
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import smtplib
import time

COMMIT_BINDING_SCHEMA = "obligate.smtp-semantic-message-binding/v2"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def message(request: dict, trace: str) -> bytes:
    msg = EmailMessage(policy=policy.SMTP)
    msg["From"] = request["sender"]
    msg["To"] = ", ".join(request["recipients"])
    if request["cc"]:
        msg["Cc"] = ", ".join(request["cc"])
    msg["Subject"] = request["subject"]
    msg["X-ObliGate-Trace"] = trace
    msg["Message-ID"] = "<" + trace + "@obligate.test>"
    msg.set_content(request["body"])
    for attachment in request["attachments"]:
        msg.add_attachment(bytes.fromhex(attachment["bytes_hex"]),
                           maintype="application", subtype="json",
                           filename=attachment["name"])
    return msg.as_bytes()


def _semantic_part(part) -> dict:
    """Bind every MIME container, leaf, role and non-serialization header.

    MIME boundaries, transfer encodings and text charsets may differ if they
    decode to the same content. Extra parts/headers/parameters cannot disappear
    merely because a client's preferred-body selector ignores them.
    """
    for name in {name.casefold() for name in part.keys()}:
        if len(part.get_all(name, [])) != 1:
            raise ValueError("duplicate_mime_or_message_header:" + name)
    encoding = str(part.get("Content-Transfer-Encoding", "7bit")).casefold()
    if encoding not in {"7bit", "8bit", "binary", "base64", "quoted-printable"}:
        raise ValueError("unsupported_content_transfer_encoding")
    content_parameters = sorted(
        (str(name).casefold(), str(value))
        for name, value in (part.get_params(header="Content-Type", failobj=[]) or [])[1:]
        if str(name).casefold() not in {"boundary", "charset"})
    disposition_parameters = sorted(
        (str(name).casefold(), str(value))
        for name, value in (part.get_params(header="Content-Disposition", failobj=[]) or [])[1:]
        if str(name).casefold() != "filename")
    signature = {"content_type": part.get_content_type(),
                 "content_parameters": content_parameters,
                 "disposition": part.get_content_disposition(),
                 "disposition_parameters": disposition_parameters,
                 "filename": part.get_filename(),
                 "headers": sorted(
                     (name.casefold(), str(value).replace("\r\n", "\n"))
                     for name, value in part.items()
                     if name.casefold() not in {"content-type", "content-transfer-encoding", "content-disposition"})}
    if part.is_multipart():
        signature["preamble"] = part.preamble.replace("\r\n", "\n") if part.preamble is not None else None
        signature["epilogue"] = part.epilogue.replace("\r\n", "\n") if part.epilogue is not None else None
        signature["parts"] = [_semantic_part(child) for child in part.iter_parts()]
    elif part.get_content_maintype() == "text" and part.get_content_disposition() != "attachment":
        signature["text"] = part.get_content().replace("\r\n", "\n")
    else:
        payload = part.get_payload(decode=True)
        if payload is None:
            raise ValueError("undecodable_mime_leaf")
        signature["payload_sha256"] = sha(payload)
    if part.defects:
        raise ValueError("malformed_mime_message")
    return signature


def decoded(raw: bytes) -> dict:
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    body = msg.get_body(preferencelist=("plain",))
    return {"subject": str(msg["Subject"]),
            "body": body.get_content().replace("\r\n", "\n").rstrip("\n") if body else "",
            "attachments": [{"name": item.get_filename(),
                             "sha256": sha(item.get_payload(decode=True))}
                            for item in msg.iter_attachments()],
            "trace": str(msg["X-ObliGate-Trace"]),
            "sender": str(msg["From"]),
            "recipients": sorted(address.addr_spec for address in msg["To"].addresses)
                          if msg["To"] else [],
            "cc": sorted(address.addr_spec for address in msg["Cc"].addresses)
                  if msg["Cc"] else [],
            "mime_signature": _semantic_part(msg)}


def payload_bytes(raw: bytes) -> bytes:
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    leaves = []
    for part in msg.walk():
        leaves.extend(str(value).replace("\r\n", "\n").encode("utf-8") for _, value in part.items())
        for text in (part.preamble, part.epilogue):
            if text is not None:
                leaves.append(text.replace("\r\n", "\n").encode("utf-8"))
        if part.is_multipart():
            continue
        if part.get_content_maintype() == "text":
            leaves.append(part.get_content().replace("\r\n", "\n").encode("utf-8"))
        else:
            leaves.append(part.get_payload(decode=True) or b"")
    return b"".join(leaves)


def send_raw(port: int, sender: str, targets: list[str], raw: bytes,
             smtp_class=None) -> dict:
    if not targets or any(not target.endswith("@obligate.test") for target in targets):
        raise ValueError("transport accepts only explicit .test capture recipients")
    client = (smtp_class or smtplib.SMTP)("127.0.0.1", port, timeout=10)
    try:
        client.ehlo()
        code, _ = client.mail(sender)
        if code != 250:
            raise RuntimeError("MAIL FROM rejected")
        for target in targets:
            code, _ = client.rcpt(target)
            if code not in (250, 251):
                raise RuntimeError("RCPT TO rejected")
        code, response = client.data(raw)
        if code != 250:
            raise RuntimeError("DATA rejected")
        return {"data_code": code, "response": response.decode(errors="replace")}
    finally:
        client.close()


class BoundSMTP:
    def __init__(self, port: int, audit: Path, *, smtp_class=None):
        self.port = port
        self.audit = Path(audit)
        self.smtp_class = smtp_class
        self.key = secrets.token_bytes(32)
        self.used: set[str] = set()
        self.events: list[dict] = []
        self.smtp_attempts = 0

    def prepare(self, request: dict, trace: str) -> tuple[dict, str]:
        prepared = deepcopy(request)
        ticket = hmac.digest(self.key, canonical({"request": prepared, "trace": trace}),
                             "sha256").hex()
        return prepared, ticket

    def submit(self, request: dict, ticket: str, trace: str, *, wire=None, envelope=None):
        start = time.perf_counter_ns()
        wanted = hmac.digest(self.key, canonical({"request": request, "trace": trace}),
                             "sha256").hex()
        if not hmac.compare_digest(ticket, wanted):
            raise ValueError("request_binding_mismatch")
        targets = sorted(set(envelope if envelope is not None else request["envelope"]))
        if targets != request["envelope"]:
            raise ValueError("envelope_binding_mismatch")
        raw = wire if wire is not None else message(request, trace)
        expected = decoded(message(request, trace))
        if decoded(raw) != expected:
            raise ValueError("serialized_content_binding_mismatch")
        if trace in self.used:
            raise ValueError("duplicate_application_intent")
        if any(not target.endswith("@obligate.test") for target in targets):
            raise ValueError("non_test_recipient")
        self.audit.parent.mkdir(parents=True, exist_ok=True)
        event = {"trace": trace, "request_digest": sha(canonical(request)),
                 "recipients_digest": sha(canonical(targets)), "stage": "before_smtp",
                 "monotonic_ns": time.perf_counter_ns()}
        with self.audit.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        audited = time.perf_counter_ns()
        # Claim before attempting a send. An unknown remote outcome is never retried here.
        self.used.add(trace)
        self.smtp_attempts += 1
        try:
            result = send_raw(self.port, request["sender"], targets, raw, self.smtp_class)
        except Exception:
            self.events.append({"trace": trace, "start_ns": start,
                                "audit_committed_ns": audited, "state": "uncertain"})
            raise
        self.events.append({"trace": trace, "start_ns": start,
                            "audit_committed_ns": audited,
                            "accepted_ns": time.perf_counter_ns(), "state": "accepted"})
        return result


def redact(request: dict, fields: list[str]) -> dict:
    request = deepcopy(request)
    for field in fields:
        request["body"] = request["body"].replace(field, "[REDACTED]")
        for attachment in request["attachments"]:
            data = bytes.fromhex(attachment["bytes_hex"]).replace(field.encode(), b"[REDACTED]")
            attachment["bytes_hex"] = data.hex()
            attachment["sha256"] = sha(data)
    return request


class ControlledSMTP(BoundSMTP):
    def __init__(self, port, audit, control_policy, redactor, **kwargs):
        super().__init__(port, audit, **kwargs)
        self.control_policy = deepcopy(control_policy)
        self.redactor = redactor

    def apply_controls(self, request: dict) -> dict:
        request = deepcopy(request)
        if not set(request["envelope"]) <= set(self.control_policy["allowed_recipients"]):
            raise ValueError("recipient_policy_mismatch")
        if self.redactor is None:
            raise RuntimeError("required_redaction_capability_missing")
        request = self.redactor(request, self.control_policy["forbidden_fields"])
        data = request["body"].encode() + b"".join(
            bytes.fromhex(attachment["bytes_hex"]) for attachment in request["attachments"])
        if any(field.encode() in data for field in self.control_policy["forbidden_fields"]):
            raise RuntimeError("redaction_incomplete")
        request["control_plan"] = {"audit": "sync_metadata", "data": "redacted",
                                   "recipients": "explicit_allowset",
                                   "redaction_policy_digest": sha(canonical(self.control_policy))}
        return request
