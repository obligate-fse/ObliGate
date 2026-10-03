"""Linearizable state and outbox for the theory-level ToolGate.

The store deliberately owns the check-and-transition operation.  Callers may do
an early validation for a nicer error, but a confirmed action is not dispatchable
until the store has revalidated every binding while holding its lock and has
atomically written the immutable outbox entry.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable, Literal

from .model import (
    ActionCandidate,
    ActionCertificate,
    ConfirmationGrant,
    PolicyAcceptanceRecord,
    Realization,
    canonical_json,
    sha256_json,
)

ConfirmationStatus = Literal[
    "pending",
    "fresh",
    "claimed",
    "sending",
    "done",
    "uncertain",
    "closed",
]
OutboxStatus = Literal["claimed", "sending", "done", "uncertain", "closed"]


class GateStateError(RuntimeError):
    """Base class for fail-closed state-machine errors."""


class DuplicateNonceError(GateStateError):
    """The nonce or replay key is already present."""


class InvalidTransitionError(GateStateError):
    """The requested transition is not legal from the current state."""


class BindingMismatchError(GateStateError):
    """A dispatch-time value differs from the value bound to the token."""


class ExpiredConfirmationError(GateStateError):
    """The confirmation token is no longer usable."""


class UnknownNonceError(GateStateError):
    """No confirmation or outbox entry exists for the nonce."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise BindingMismatchError(f"{name} binding mismatch")


def action_operation_digest(action: ActionCandidate) -> str:
    """Stable identity used to suppress unresolved delivery retries.

    It deliberately excludes facts, evidence, plan, realization, certificate,
    and classifier flags: re-adjudicating or reclassifying the same tool call
    must not turn an unknown delivery outcome into a fresh send attempt.
    """

    return sha256_json(
        {
            "actor": action.actor,
            "tenant": action.tenant,
            "session": action.session,
            "tool_name": action.tool_name,
            "arguments": dict(action.arguments),
            "payload_digest": action.payload_digest,
            "asset_digests": list(action.asset_digests),
        }
    )


def validate_dispatch_bindings(
    *,
    action: ActionCandidate,
    certificate: ActionCertificate,
    policy: PolicyAcceptanceRecord,
    realization: Realization,
    fact_digest: str,
    evidence_digest: str,
    plan_digest: str,
) -> None:
    """Validate the certificate against values recomputed immediately pre-send."""

    if not policy.accepted:
        raise BindingMismatchError("policy acceptance record is not accepted")
    if not realization.available:
        raise BindingMismatchError("realization is unavailable")
    if certificate.outcome not in ("allow", "execute_with_constraints"):
        raise InvalidTransitionError(
            f"certificate outcome {certificate.outcome!r} is not dispatchable"
        )
    nonbaseline_controls = any(
        root != "baseline" and not root.startswith("baseline:")
        for _, root in realization.control_roots
    )
    if realization.plan.block_like:
        raise InvalidTransitionError("a block-like realization is never dispatchable")
    if realization.plan.confirmation:
        if (
            certificate.nonce is None
            or certificate.prior_certificate_digest is None
            or certificate.confirmation_binding_digest is None
        ):
            raise InvalidTransitionError(
                "an approval-required realization needs a fresh bound confirmation grant"
            )
        expected_outcome = "execute_with_constraints"
    elif realization.plan.default_controls and not nonbaseline_controls:
        expected_outcome = "allow"
    else:
        expected_outcome = "execute_with_constraints"
    if certificate.outcome != expected_outcome:
        raise BindingMismatchError(
            f"outcome/realization projection mismatch: expected {expected_outcome!r}"
        )

    _require_equal("policy", certificate.policy_acceptance_digest, policy.digest)
    _require_equal("action", certificate.action_digest, action.digest)
    _require_equal("fact", certificate.fact_digest, fact_digest)
    _require_equal("evidence", certificate.evidence_digest, evidence_digest)
    _require_equal("plan", certificate.plan_digest, plan_digest)
    _require_equal("realization plan", realization.plan.digest, plan_digest)
    _require_equal("realization", certificate.realization_digest, realization.digest)


@dataclass(frozen=True, slots=True)
class ConfirmationBinding:
    """Immutable data covered by a single user-confirmation token."""

    nonce: str
    actor: str
    tenant: str
    session: str
    action_digest: str
    action_json: str
    payload_digest: str
    asset_digests: tuple[str, ...]
    fact_digest: str
    evidence_digest: str
    policy_acceptance_digest: str
    plan_digest: str
    initial_certificate_digest: str
    expires_at: datetime

    def __post_init__(self) -> None:
        if not self.nonce:
            raise ValueError("nonce must be non-empty")
        object.__setattr__(self, "asset_digests", tuple(sorted(self.asset_digests)))
        object.__setattr__(self, "expires_at", _utc(self.expires_at))

    @classmethod
    def from_request(
        cls,
        *,
        nonce: str,
        action: ActionCandidate,
        certificate: ActionCertificate,
        expires_at: datetime,
    ) -> "ConfirmationBinding":
        return cls(
            nonce=nonce,
            actor=action.actor,
            tenant=action.tenant,
            session=action.session,
            action_digest=action.digest,
            action_json=canonical_json(action.normalized_dict()),
            payload_digest=action.payload_digest,
            asset_digests=action.asset_digests,
            fact_digest=certificate.fact_digest,
            evidence_digest=certificate.evidence_digest,
            policy_acceptance_digest=certificate.policy_acceptance_digest,
            plan_digest=certificate.plan_digest,
            initial_certificate_digest=certificate.digest,
            expires_at=expires_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nonce": self.nonce,
            "actor": self.actor,
            "tenant": self.tenant,
            "session": self.session,
            "action_digest": self.action_digest,
            "action_json": self.action_json,
            "payload_digest": self.payload_digest,
            "asset_digests": list(self.asset_digests),
            "fact_digest": self.fact_digest,
            "evidence_digest": self.evidence_digest,
            "policy_acceptance_digest": self.policy_acceptance_digest,
            "plan_digest": self.plan_digest,
            "initial_certificate_digest": self.initial_certificate_digest,
            "expires_at": self.expires_at.isoformat(),
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class ConfirmationRecord:
    binding: ConfirmationBinding
    status: ConfirmationStatus
    created_at: datetime
    confirmed_at: datetime | None = None
    claimed_at: datetime | None = None
    sending_at: datetime | None = None
    completed_at: datetime | None = None
    close_reason: str = ""


@dataclass(frozen=True, slots=True)
class OutboxEntry:
    """Append-only dispatch identity plus monotone delivery state.

    Status changes replace the snapshot; the nonce, action, certificate and all
    policy/evidence bindings never change.
    """

    nonce: str
    operation_digest: str
    action_digest: str
    action_json: str
    payload_digest: str
    asset_digests: tuple[str, ...]
    fact_digest: str
    evidence_digest: str
    policy_acceptance_digest: str
    plan_digest: str
    certificate_digest: str
    realization_digest: str
    idempotency_key: str | None
    status: OutboxStatus
    claimed_at: datetime
    sending_at: datetime | None = None
    completed_at: datetime | None = None
    result_digest: str = ""
    error: str = ""


class LinearizableGateStateStore:
    """Thread-safe in-memory state store with linearizable transitions."""

    def __init__(self, *, clock: Callable[[], datetime] = utc_now) -> None:
        self._clock = clock
        self._lock = RLock()
        self._confirmations: dict[str, ConfirmationRecord] = {}
        self._outbox: dict[str, OutboxEntry] = {}
        # Includes claimed/sending/uncertain operations.  A known completion
        # releases the key; an uncertain outcome requires explicit reconcile.
        self._active_operations: dict[str, str] = {}

    def _now(self, supplied: datetime | None = None) -> datetime:
        return _utc(supplied if supplied is not None else self._clock())

    def issue_pending(
        self,
        binding: ConfirmationBinding,
        *,
        now: datetime | None = None,
    ) -> ConfirmationRecord:
        timestamp = self._now(now)
        with self._lock:
            if binding.nonce in self._confirmations or binding.nonce in self._outbox:
                raise DuplicateNonceError(f"nonce {binding.nonce!r} already exists")
            if timestamp >= binding.expires_at:
                raise ExpiredConfirmationError("confirmation expires before issuance")
            record = ConfirmationRecord(
                binding=binding,
                status="pending",
                created_at=timestamp,
            )
            self._confirmations[binding.nonce] = record
            return record

    def confirm(
        self,
        nonce: str,
        *,
        actor: str,
        tenant: str,
        session: str,
        now: datetime | None = None,
    ) -> ConfirmationRecord:
        timestamp = self._now(now)
        with self._lock:
            record = self._confirmation_locked(nonce)
            if record.status != "pending":
                raise InvalidTransitionError(
                    f"confirmation {nonce!r} is {record.status}, not pending"
                )
            self._require_not_expired(record, timestamp)
            _require_equal("actor", actor, record.binding.actor)
            _require_equal("tenant", tenant, record.binding.tenant)
            _require_equal("session", session, record.binding.session)
            fresh = replace(record, status="fresh", confirmed_at=timestamp)
            self._confirmations[nonce] = fresh
            return fresh

    def confirmation_grant(self, nonce: str) -> ConfirmationGrant:
        with self._lock:
            record = self._confirmation_locked(nonce)
            if record.status != "fresh":
                raise InvalidTransitionError(
                    f"confirmation {nonce!r} is {record.status}, not fresh"
                )
            return ConfirmationGrant(
                nonce=nonce,
                confirmation_binding_digest=record.binding.digest,
                initial_certificate_digest=record.binding.initial_certificate_digest,
            )

    def claim_confirmed(
        self,
        nonce: str,
        *,
        action: ActionCandidate,
        certificate: ActionCertificate,
        policy: PolicyAcceptanceRecord,
        realization: Realization,
        fact_digest: str,
        evidence_digest: str,
        plan_digest: str,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> OutboxEntry:
        """Atomically consume ``fresh`` and create the immutable outbox item."""

        timestamp = self._now(now)
        with self._lock:
            record = self._confirmation_locked(nonce)
            if record.status != "fresh":
                raise InvalidTransitionError(
                    f"confirmation {nonce!r} is {record.status}, not fresh"
                )
            self._require_not_expired(record, timestamp)
            binding = record.binding

            # This is intentionally repeated under the transition lock.  It is
            # the linearization point for authorization and durable intent.
            validate_dispatch_bindings(
                action=action,
                certificate=certificate,
                policy=policy,
                realization=realization,
                fact_digest=fact_digest,
                evidence_digest=evidence_digest,
                plan_digest=plan_digest,
            )
            _require_equal("nonce", certificate.nonce, nonce)
            if certificate.digest == binding.initial_certificate_digest:
                raise BindingMismatchError(
                    "confirmed dispatch requires a newly adjudicated certificate"
                )
            _require_equal(
                "prior certificate",
                certificate.prior_certificate_digest,
                binding.initial_certificate_digest,
            )
            _require_equal(
                "confirmation binding",
                certificate.confirmation_binding_digest,
                binding.digest,
            )
            _require_equal("actor", action.actor, binding.actor)
            _require_equal("tenant", action.tenant, binding.tenant)
            _require_equal("session", action.session, binding.session)
            _require_equal("action", action.digest, binding.action_digest)
            _require_equal(
                "normalized action",
                canonical_json(action.normalized_dict()),
                binding.action_json,
            )
            _require_equal("payload", action.payload_digest, binding.payload_digest)
            _require_equal("assets", action.asset_digests, binding.asset_digests)
            if fact_digest == binding.fact_digest or evidence_digest == binding.evidence_digest:
                raise BindingMismatchError(
                    "confirmed dispatch must bind newly adjudicated fact and evidence snapshots"
                )
            _require_equal("policy", policy.digest, binding.policy_acceptance_digest)
            _require_equal("plan", plan_digest, binding.plan_digest)

            if nonce in self._outbox:
                raise DuplicateNonceError(f"nonce {nonce!r} already has an outbox item")
            self._claim_operation_locked(action, nonce)
            outbox = self._new_outbox(
                nonce=nonce,
                action=action,
                certificate=certificate,
                realization=realization,
                fact_digest=fact_digest,
                evidence_digest=evidence_digest,
                plan_digest=plan_digest,
                idempotency_key=idempotency_key,
                timestamp=timestamp,
            )
            self._outbox[nonce] = outbox
            self._confirmations[nonce] = replace(
                record,
                status="claimed",
                claimed_at=timestamp,
            )
            return outbox

    def claim_direct(
        self,
        replay_key: str,
        *,
        action: ActionCandidate,
        certificate: ActionCertificate,
        policy: PolicyAcceptanceRecord,
        realization: Realization,
        fact_digest: str,
        evidence_digest: str,
        plan_digest: str,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> OutboxEntry:
        """Atomically create an outbox item for an allow that needs no prompt."""

        timestamp = self._now(now)
        with self._lock:
            if certificate.nonce is not None:
                raise InvalidTransitionError(
                    "a nonce-bound certificate must use the confirmed claim path"
                )
            validate_dispatch_bindings(
                action=action,
                certificate=certificate,
                policy=policy,
                realization=realization,
                fact_digest=fact_digest,
                evidence_digest=evidence_digest,
                plan_digest=plan_digest,
            )
            if replay_key in self._outbox or replay_key in self._confirmations:
                raise DuplicateNonceError(
                    f"replay key {replay_key!r} already has a state entry"
                )
            self._claim_operation_locked(action, replay_key)
            outbox = self._new_outbox(
                nonce=replay_key,
                action=action,
                certificate=certificate,
                realization=realization,
                fact_digest=fact_digest,
                evidence_digest=evidence_digest,
                plan_digest=plan_digest,
                idempotency_key=idempotency_key,
                timestamp=timestamp,
            )
            self._outbox[replay_key] = outbox
            return outbox

    def mark_sending(
        self, nonce: str, *, now: datetime | None = None
    ) -> OutboxEntry:
        timestamp = self._now(now)
        with self._lock:
            outbox = self._outbox_locked(nonce)
            if outbox.status != "claimed":
                raise InvalidTransitionError(
                    f"outbox {nonce!r} is {outbox.status}, not claimed"
                )
            updated = replace(outbox, status="sending", sending_at=timestamp)
            self._outbox[nonce] = updated
            self._advance_confirmation_locked(nonce, "claimed", "sending", timestamp)
            return updated

    def mark_done(
        self,
        nonce: str,
        *,
        result_digest: str,
        now: datetime | None = None,
    ) -> OutboxEntry:
        timestamp = self._now(now)
        with self._lock:
            outbox = self._outbox_locked(nonce)
            if outbox.status != "sending":
                raise InvalidTransitionError(
                    f"outbox {nonce!r} is {outbox.status}, not sending"
                )
            updated = replace(
                outbox,
                status="done",
                completed_at=timestamp,
                result_digest=result_digest,
            )
            self._outbox[nonce] = updated
            self._advance_confirmation_locked(nonce, "sending", "done", timestamp)
            self._release_operation_locked(outbox)
            return updated

    def mark_uncertain(
        self,
        nonce: str,
        *,
        error: str,
        now: datetime | None = None,
    ) -> OutboxEntry:
        timestamp = self._now(now)
        with self._lock:
            outbox = self._outbox_locked(nonce)
            if outbox.status not in ("claimed", "sending"):
                raise InvalidTransitionError(
                    f"outbox {nonce!r} is {outbox.status}, not in flight"
                )
            previous = outbox.status
            updated = replace(
                outbox,
                status="uncertain",
                completed_at=timestamp,
                error=error,
            )
            self._outbox[nonce] = updated
            self._advance_confirmation_locked(
                nonce, previous, "uncertain", timestamp
            )
            return updated

    def reconcile_uncertain(
        self,
        nonce: str,
        *,
        delivered: bool,
        result_digest: str = "",
        reason: str = "",
        now: datetime | None = None,
    ) -> OutboxEntry:
        """Explicitly resolve an unknown delivery before any same-action retry."""

        timestamp = self._now(now)
        with self._lock:
            outbox = self._outbox_locked(nonce)
            if outbox.status != "uncertain":
                raise InvalidTransitionError(
                    f"outbox {nonce!r} is {outbox.status}, not uncertain"
                )
            if delivered and not result_digest:
                raise ValueError("a delivered reconciliation requires result_digest")
            updated = replace(
                outbox,
                status="done" if delivered else "closed",
                completed_at=timestamp,
                result_digest=result_digest if delivered else "",
                error=reason,
            )
            self._outbox[nonce] = updated
            self._release_operation_locked(outbox)
            record = self._confirmations.get(nonce)
            if record is not None and record.status == "uncertain":
                self._confirmations[nonce] = replace(
                    record,
                    status="done" if delivered else "closed",
                    completed_at=timestamp,
                    close_reason=reason,
                )
            return updated

    def close_confirmation(
        self,
        nonce: str,
        *,
        reason: str,
        now: datetime | None = None,
    ) -> ConfirmationRecord:
        timestamp = self._now(now)
        with self._lock:
            record = self._confirmation_locked(nonce)
            if record.status not in ("pending", "fresh"):
                raise InvalidTransitionError(
                    f"confirmation {nonce!r} cannot close from {record.status}"
                )
            closed = replace(
                record,
                status="closed",
                completed_at=timestamp,
                close_reason=reason,
            )
            self._confirmations[nonce] = closed
            return closed

    def get_confirmation(self, nonce: str) -> ConfirmationRecord:
        with self._lock:
            return self._confirmation_locked(nonce)

    def get_outbox(self, nonce: str) -> OutboxEntry:
        with self._lock:
            return self._outbox_locked(nonce)

    def _new_outbox(
        self,
        *,
        nonce: str,
        action: ActionCandidate,
        certificate: ActionCertificate,
        realization: Realization,
        fact_digest: str,
        evidence_digest: str,
        plan_digest: str,
        idempotency_key: str | None,
        timestamp: datetime,
    ) -> OutboxEntry:
        return OutboxEntry(
            nonce=nonce,
            operation_digest=action_operation_digest(action),
            action_digest=action.digest,
            action_json=canonical_json(action.normalized_dict()),
            payload_digest=action.payload_digest,
            asset_digests=action.asset_digests,
            fact_digest=fact_digest,
            evidence_digest=evidence_digest,
            policy_acceptance_digest=certificate.policy_acceptance_digest,
            plan_digest=plan_digest,
            certificate_digest=certificate.digest,
            realization_digest=realization.digest,
            idempotency_key=idempotency_key,
            status="claimed",
            claimed_at=timestamp,
        )

    def _claim_operation_locked(self, action: ActionCandidate, nonce: str) -> None:
        operation_digest = action_operation_digest(action)
        previous = self._active_operations.get(operation_digest)
        if previous is not None:
            previous_entry = self._outbox.get(previous)
            status = previous_entry.status if previous_entry is not None else "active"
            raise DuplicateNonceError(
                "same canonical operation already has an unresolved "
                f"{status} delivery at {previous!r}"
            )
        self._active_operations[operation_digest] = nonce

    def _release_operation_locked(self, outbox: OutboxEntry) -> None:
        if self._active_operations.get(outbox.operation_digest) == outbox.nonce:
            del self._active_operations[outbox.operation_digest]

    def _confirmation_locked(self, nonce: str) -> ConfirmationRecord:
        try:
            return self._confirmations[nonce]
        except KeyError as exc:
            raise UnknownNonceError(f"unknown confirmation nonce {nonce!r}") from exc

    def _outbox_locked(self, nonce: str) -> OutboxEntry:
        try:
            return self._outbox[nonce]
        except KeyError as exc:
            raise UnknownNonceError(f"unknown outbox nonce {nonce!r}") from exc

    @staticmethod
    def _require_not_expired(
        record: ConfirmationRecord, timestamp: datetime
    ) -> None:
        if timestamp >= record.binding.expires_at:
            raise ExpiredConfirmationError(
                f"confirmation {record.binding.nonce!r} has expired"
            )

    def _advance_confirmation_locked(
        self,
        nonce: str,
        expected: ConfirmationStatus,
        target: ConfirmationStatus,
        timestamp: datetime,
    ) -> None:
        record = self._confirmations.get(nonce)
        if record is None:
            return
        if record.status != expected:
            raise InvalidTransitionError(
                f"confirmation {nonce!r} is {record.status}, expected {expected}"
            )
        changes: dict[str, Any] = {"status": target}
        if target == "sending":
            changes["sending_at"] = timestamp
        if target in ("done", "uncertain"):
            changes["completed_at"] = timestamp
        self._confirmations[nonce] = replace(record, **changes)


# Descriptive alias for callers that prefer the storage implementation in its
# name; both names refer to the same linearizable in-memory implementation.
InMemoryLinearizableStateStore = LinearizableGateStateStore


__all__ = [
    "BindingMismatchError",
    "ConfirmationBinding",
    "ConfirmationRecord",
    "DuplicateNonceError",
    "ExpiredConfirmationError",
    "GateStateError",
    "InMemoryLinearizableStateStore",
    "InvalidTransitionError",
    "LinearizableGateStateStore",
    "OutboxEntry",
    "UnknownNonceError",
    "action_operation_digest",
    "validate_dispatch_bindings",
]
