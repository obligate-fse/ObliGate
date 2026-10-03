"""Trusted issuance proof for theory ActionCertificates."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field, replace

from .model import ActionCertificate, canonical_json


@dataclass(frozen=True, slots=True)
class CertificateIssuer:
    """HMAC-backed in-process issuer; production may inject a durable TCB key."""

    key: bytes = field(repr=False)
    issuer_id: str = ""

    def __post_init__(self) -> None:
        if len(self.key) < 32:
            raise ValueError("certificate issuer key must contain at least 32 bytes")
        if not self.issuer_id:
            object.__setattr__(
                self,
                "issuer_id",
                f"issuer:{hashlib.sha256(self.key).hexdigest()[:20]}",
            )

    def issue(self, certificate: ActionCertificate) -> ActionCertificate:
        unsigned = replace(certificate, issuer_id=self.issuer_id, issuer_proof="")
        proof = hmac.new(
            self.key,
            canonical_json(unsigned.signing_dict()).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return replace(unsigned, issuer_proof=proof)

    def verify(self, certificate: ActionCertificate) -> bool:
        if certificate.issuer_id != self.issuer_id or not certificate.issuer_proof:
            return False
        expected = self.issue(
            replace(certificate, issuer_proof="")
        ).issuer_proof
        return hmac.compare_digest(certificate.issuer_proof, expected)


_PROCESS_ISSUER = CertificateIssuer(secrets.token_bytes(32))


def default_certificate_issuer() -> CertificateIssuer:
    return _PROCESS_ISSUER


__all__ = ["CertificateIssuer", "default_certificate_issuer"]
