"""Offline binding checks and end-to-end tests across a real loopback SMTP socket."""
from copy import deepcopy
from email import policy
from email.encoders import encode_base64
from email.parser import BytesParser
import json
from pathlib import Path
import tempfile
import unittest
import uuid

from experiments.smtp_runtime.adapter import BoundSMTP, ControlledSMTP, decoded, message, payload_bytes, redact
from experiments.smtp_runtime.run import ROOT, run_trial
from experiments.smtp_runtime.testbed import CaptureServer, observe


class SMTPRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seeds = json.loads((ROOT / "configs/smtp_mail_seeds.json").read_text(encoding="utf-8"))

    def test_request_mutation_rejected_before_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = BoundSMTP(1, Path(directory) / "audit.jsonl")
            trace = uuid.uuid4().hex
            request, ticket = adapter.prepare(self.seeds[0]["request"], trace)
            request["recipients"].append("attacker@obligate.test")
            with self.assertRaisesRegex(ValueError, "request_binding"):
                adapter.submit(request, ticket, trace)
            self.assertEqual(adapter.smtp_attempts, 0)
            self.assertFalse(adapter.audit.exists())

    def test_envelope_and_serialized_payload_rejected_before_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = BoundSMTP(1, Path(directory) / "audit.jsonl")
            trace = uuid.uuid4().hex
            request, ticket = adapter.prepare(self.seeds[0]["request"], trace)
            with self.assertRaisesRegex(ValueError, "envelope_binding"):
                adapter.submit(request, ticket, trace,
                               envelope=request["envelope"] + ["attacker@obligate.test"])
            altered = deepcopy(request)
            altered["attachments"][0]["bytes_hex"] += "20"
            with self.assertRaisesRegex(ValueError, "serialized_content"):
                adapter.submit(request, ticket, trace, wire=message(altered, trace))
            self.assertEqual(adapter.smtp_attempts, 0)
            self.assertFalse(adapter.audit.exists())

    def test_ticket_binds_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = BoundSMTP(1, Path(directory) / "audit.jsonl")
            request, ticket = adapter.prepare(self.seeds[0]["request"], uuid.uuid4().hex)
            with self.assertRaisesRegex(ValueError, "request_binding"):
                adapter.submit(request, ticket, uuid.uuid4().hex)
            self.assertEqual(adapter.smtp_attempts, 0)

    def test_undeclared_mime_alternatives_inline_parts_and_types_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = BoundSMTP(1, Path(directory) / "audit.jsonl")
            trace = uuid.uuid4().hex
            request, ticket = adapter.prepare(self.seeds[0]["request"], trace)
            original = message(request, trace)
            alternative = BytesParser(policy=policy.SMTP).parsebytes(original)
            alternative.get_body(preferencelist=("plain",)).add_alternative(
                "<p>UNAUTHORIZED ADDITIONAL HTML PAYLOAD</p>", subtype="html")
            inline = BytesParser(policy=policy.SMTP).parsebytes(original)
            inline.get_body(preferencelist=("plain",)).add_related(
                b"UNAUTHORIZED INLINE PAYLOAD", maintype="application", subtype="octet-stream")
            changed_type = BytesParser(policy=policy.SMTP).parsebytes(original)
            next(changed_type.iter_attachments()).replace_header("Content-Type", "text/html")
            extra_header = BytesParser(policy=policy.SMTP).parsebytes(original)
            extra_header["X-Undeclared-Payload"] = "UNAUTHORIZED HEADER PAYLOAD"
            preamble = BytesParser(policy=policy.SMTP).parsebytes(original)
            preamble.preamble = "UNAUTHORIZED PREAMBLE PAYLOAD"
            epilogue = BytesParser(policy=policy.SMTP).parsebytes(original)
            epilogue.epilogue = "UNAUTHORIZED EPILOGUE PAYLOAD"
            duplicate = original.replace(b"Subject: ", b"Subject: Duplicate\r\nSubject: ", 1)
            for raw in [alternative.as_bytes(), inline.as_bytes(), changed_type.as_bytes(),
                        extra_header.as_bytes(), preamble.as_bytes(), epilogue.as_bytes(), duplicate]:
                with self.assertRaises(ValueError):
                    adapter.submit(request, ticket, trace, wire=raw)
            self.assertIn(b"UNAUTHORIZED ADDITIONAL HTML PAYLOAD", payload_bytes(alternative.as_bytes()))
            self.assertIn(b"UNAUTHORIZED INLINE PAYLOAD", payload_bytes(inline.as_bytes()))
            self.assertIn(b"UNAUTHORIZED PREAMBLE PAYLOAD", payload_bytes(preamble.as_bytes()))
            self.assertIn(b"UNAUTHORIZED EPILOGUE PAYLOAD", payload_bytes(epilogue.as_bytes()))
            self.assertEqual(adapter.smtp_attempts, 0)
            self.assertFalse(adapter.audit.exists())

    def test_synchronous_audit_failure_precedes_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = BoundSMTP(1, Path(directory))
            trace = uuid.uuid4().hex
            request, ticket = adapter.prepare(self.seeds[0]["request"], trace)
            with self.assertRaises((IsADirectoryError, PermissionError)):
                adapter.submit(request, ticket, trace)
            self.assertEqual(adapter.smtp_attempts, 0)

    def test_equivalent_mime_and_real_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with CaptureServer(out / "captures") as server:
                row = run_trial(self.seeds[0], "equivalent_MIME", "Full", server, out)
                self.assertTrue(row["expectation_passed"])
                self.assertEqual(row["captures"], 1)
                observation = observe(server.spool, row["trace_id"])
                self.assertEqual(observation["messages"][0]["decoded"]["attachments"][0]["sha256"],
                                 self.seeds[0]["request"]["attachments"][0]["sha256"])

    def test_equivalent_base64_body_encoding_real_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with CaptureServer(out / "captures") as server:
                adapter = BoundSMTP(server.port, out / "audit.jsonl")
                trace = uuid.uuid4().hex
                request, ticket = adapter.prepare(self.seeds[0]["request"], trace)
                msg = BytesParser(policy=policy.SMTP).parsebytes(message(request, trace))
                body = msg.get_body(preferencelist=("plain",))
                data = body.get_payload(decode=True)
                del body["Content-Transfer-Encoding"]
                body.set_payload(data)
                encode_base64(body)
                adapter.submit(request, ticket, trace, wire=msg.as_bytes())
                self.assertEqual(len(observe(server.spool, trace)["messages"]), 1)

    def test_real_confirmation_requires_fresh_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with CaptureServer(out / "captures") as server:
                row = run_trial(self.seeds[0], "valid_confirmation", "Full", server, out)
                self.assertTrue(row["expectation_passed"])
                self.assertFalse(row["confirmation"]["initial_dispatched"])
                self.assertFalse(row["confirmation"]["confirmation_dispatched"])
                self.assertNotEqual(row["initial_certificate"], row["fresh_certificate"])
                self.assertEqual(row["gate_state"], "done")

    def test_wrong_actor_and_expiry_never_cross_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with CaptureServer(out / "captures") as server:
                for condition in ("wrong_actor", "expired_confirmation"):
                    row = run_trial(self.seeds[0], condition, "Full", server, out)
                    self.assertTrue(row["expectation_passed"])
                    self.assertEqual(row["captures"], 0)
                    self.assertEqual(row["smtp_attempts"], 0)
                    self.assertEqual(row["confirmation"]["state"], "blocked")

    def test_lost_real_data_reply_has_one_effect_and_no_redispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with CaptureServer(out / "captures") as server:
                row = run_trial(self.seeds[0], "lost_DATA_reply", "Full", server, out)
                self.assertTrue(row["expectation_passed"])
                self.assertEqual(row["gate_state"], "uncertain")
                self.assertEqual(row["captures"], 1)
                self.assertEqual(row["smtp_attempts"], 1)
                self.assertFalse(row["retry_dispatched"])
                baseline = run_trial(self.seeds[0], "lost_DATA_reply", "NoGate", server, out)
                self.assertTrue(baseline["expectation_passed"])
                self.assertEqual(baseline["captures"], 2)

    def test_redaction_and_failure_controls_real_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with CaptureServer(out / "captures") as server:
                for condition in ("redaction_success", "missing_redactor", "incomplete_redaction", "redactor_exception"):
                    row = run_trial(self.seeds[4], condition, "Full", server, out)
                    self.assertTrue(row["expectation_passed"], condition)
                    self.assertFalse(row["restricted_field_leakage"])
                    self.assertEqual(row["captures"], int(condition == "redaction_success"))
                    baseline = run_trial(self.seeds[4], condition, "NoGate", server, out)
                    self.assertTrue(baseline["expectation_passed"], condition)
                    self.assertTrue(baseline["restricted_field_leakage"])

    def test_independent_observer_detects_reachable_forbidden_recipient(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with CaptureServer(out / "captures") as server:
                row = run_trial(self.seeds[0], "envelope_mutation", "NoGate", server, out)
                self.assertTrue(row["expectation_passed"])
                self.assertIn("attacker@obligate.test", row["observed_recipients"])


if __name__ == "__main__":
    unittest.main()
