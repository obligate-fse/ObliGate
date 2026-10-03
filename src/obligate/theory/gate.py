"""Theory-compliant ToolGate and confirmation protocol.

The gate is the only component in this module that invokes a remote dispatcher.
It first recomputes certificate bindings, atomically claims authorization together
with an outbox write, marks the send in progress, and only then crosses the remote
boundary.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

from .issuer import CertificateIssuer, default_certificate_issuer
from .model import (
    ActionCandidate,
    ActionCertificate,
    PolicyAcceptanceRecord,
    PublicDecision,
    Realization,
    sha256_json,
)
from .policy import PolicySpec, build_default_policy
from .realizer import Realizer
from .runtime import _replay_checked_policy
from .state import (
    BindingMismatchError,
    ConfirmationBinding,
    GateStateError,
    LinearizableGateStateStore,
    OutboxEntry,
    utc_now,
)

Dispatcher = Callable[[ActionCandidate, Realization, str | None], Any]


@dataclass(frozen=True, slots=True)
class GateResult:
    decision: PublicDecision
    state: str
    dispatched: bool
    nonce: str | None = None
    reason: str = ""
    result: Any = None
    outbox: OutboxEntry | None = None


class ToolGate:
    """Fail-closed execution boundary for certified actions."""

    def __init__(
        self,
        state: LinearizableGateStateStore | None = None,
        *,
        confirmation_ttl: timedelta = timedelta(minutes=5),
        nonce_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] = utc_now,
        policy: PolicySpec | None = None,
        acceptance: PolicyAcceptanceRecord | None = None,
        realizer: Realizer | None = None,
        certificate_issuer: CertificateIssuer | None = None,
    ) -> None:
        if confirmation_ttl <= timedelta(0):
            raise ValueError("confirmation_ttl must be positive")
        self.state = state if state is not None else LinearizableGateStateStore(clock=clock)
        self._confirmation_ttl = confirmation_ttl
        self._nonce_factory = nonce_factory or (lambda: secrets.token_urlsafe(24))
        self._clock = clock
        self._certificate_issuer = certificate_issuer or default_certificate_issuer()
        self.policy_spec = policy or build_default_policy()
        replayed = _replay_checked_policy(self.policy_spec)
        if acceptance is not None and acceptance.digest != replayed.digest:
            raise ValueError("ToolGate acceptance does not bind the replay-checked policy")
        self.policy_acceptance = replayed
        self.realizer = realizer

    def process(
        self,
        *,
        action: ActionCandidate,
        certificate: ActionCertificate,
        policy: PolicyAcceptanceRecord,
        realization: Realization,
        fact_digest: str,
        evidence_digest: str,
        plan_digest: str,
        dispatcher: Dispatcher | None = None,
        idempotency_key: str | None = None,
        nonce: str | None = None,
        expires_at: datetime | None = None,
        now: datetime | None = None,
    ) -> GateResult:
        """Route a certificate without ever dispatching a confirmation decision."""

        try:
            self._validate_certificate_authenticity(certificate)
            self._validate_loaded_policy(policy, realization)
        except BindingMismatchError as exc:
            return self._fail(str(exc), nonce=certificate.nonce)

        if certificate.outcome in ("block", "block_with_compliance_error"):
            return self._fail(
                f"certificate outcome {certificate.outcome!r} is non-dispatchable",
                nonce=certificate.nonce,
                decision=certificate.outcome,
            )
        if certificate.outcome == "require_confirmation":
            # The dispatcher is deliberately ignored on this branch.  A later
            # call must carry a newly adjudicated allow/constraint certificate.
            return self.request_confirmation(
                action=action,
                certificate=certificate,
                policy=policy,
                realization=realization,
                fact_digest=fact_digest,
                evidence_digest=evidence_digest,
                plan_digest=plan_digest,
                nonce=nonce,
                expires_at=expires_at,
                now=now,
            )
        if dispatcher is None:
            return self._fail("dispatchable certificate requires a dispatcher")
        return self.dispatch(
            action=action,
            certificate=certificate,
            policy=policy,
            realization=realization,
            fact_digest=fact_digest,
            evidence_digest=evidence_digest,
            plan_digest=plan_digest,
            dispatcher=dispatcher,
            idempotency_key=idempotency_key,
            now=now,
        )

    def request_confirmation(
        self,
        *,
        action: ActionCandidate,
        certificate: ActionCertificate,
        policy: PolicyAcceptanceRecord,
        realization: Realization,
        fact_digest: str,
        evidence_digest: str,
        plan_digest: str,
        nonce: str | None = None,
        expires_at: datetime | None = None,
        now: datetime | None = None,
    ) -> GateResult:
        timestamp = now if now is not None else self._clock()
        chosen_nonce = nonce or certificate.nonce or self._nonce_factory()
        expiry = expires_at or (timestamp + self._confirmation_ttl)
        try:
            self._validate_certificate_authenticity(certificate)
            self._validate_loaded_policy(policy, realization)
            self._validate_confirmation_request(
                action=action,
                certificate=certificate,
                policy=policy,
                realization=realization,
                fact_digest=fact_digest,
                evidence_digest=evidence_digest,
                plan_digest=plan_digest,
                nonce=chosen_nonce,
            )
            binding = ConfirmationBinding.from_request(
                nonce=chosen_nonce,
                action=action,
                certificate=certificate,
                expires_at=expiry,
            )
            self.state.issue_pending(binding, now=timestamp)
        except (GateStateError, ValueError) as exc:
            return self._fail(str(exc), nonce=chosen_nonce)
        return GateResult(
            decision="require_confirmation",
            state="pending",
            dispatched=False,
            nonce=chosen_nonce,
        )

    def confirm(
        self,
        nonce: str,
        *,
        actor: str,
        tenant: str,
        session: str,
        now: datetime | None = None,
    ) -> GateResult:
        try:
            record = self.state.confirm(
                nonce,
                actor=actor,
                tenant=tenant,
                session=session,
                now=now,
            )
        except GateStateError as exc:
            return self._fail(str(exc), nonce=nonce)
        grant = self.state.confirmation_grant(nonce)
        return GateResult(
            decision="require_confirmation",
            state=record.status,
            dispatched=False,
            nonce=nonce,
            result=grant,
        )

    def dispatch(
        self,
        *,
        action: ActionCandidate,
        certificate: ActionCertificate,
        policy: PolicyAcceptanceRecord,
        realization: Realization,
        fact_digest: str,
        evidence_digest: str,
        plan_digest: str,
        dispatcher: Dispatcher,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> GateResult:
        """Claim once, create outbox intent, and perform one remote send attempt."""

        replay_key = certificate.nonce or f"direct:{certificate.digest}"
        try:
            self._validate_certificate_authenticity(certificate)
            self._validate_loaded_policy(policy, realization)
            if certificate.nonce is not None:
                self.state.claim_confirmed(
                    certificate.nonce,
                    action=action,
                    certificate=certificate,
                    policy=policy,
                    realization=realization,
                    fact_digest=fact_digest,
                    evidence_digest=evidence_digest,
                    plan_digest=plan_digest,
                    idempotency_key=idempotency_key,
                    now=now,
                )
            else:
                self.state.claim_direct(
                    replay_key,
                    action=action,
                    certificate=certificate,
                    policy=policy,
                    realization=realization,
                    fact_digest=fact_digest,
                    evidence_digest=evidence_digest,
                    plan_digest=plan_digest,
                    idempotency_key=idempotency_key,
                    now=now,
                )
            self.state.mark_sending(replay_key, now=now)
        except GateStateError as exc:
            return self._fail(str(exc), nonce=certificate.nonce)

        try:
            result = dispatcher(action, realization, idempotency_key)
        except Exception as exc:  # the send may have reached the remote boundary
            error = f"{type(exc).__name__}: {exc}"
            uncertain = self.state.mark_uncertain(replay_key, error=error, now=now)
            suffix = (
                "no idempotency key; automatic retry is forbidden"
                if idempotency_key is None
                else "delivery outcome is unknown; no automatic retry was attempted"
            )
            return GateResult(
                decision=certificate.outcome,
                state="uncertain",
                dispatched=True,
                nonce=certificate.nonce,
                reason=f"{error}; {suffix}",
                outbox=uncertain,
            )

        done = self.state.mark_done(
            replay_key,
            result_digest=self._result_digest(result),
            now=now,
        )
        return GateResult(
            decision=certificate.outcome,
            state="done",
            dispatched=True,
            nonce=certificate.nonce,
            result=result,
            outbox=done,
        )

    @staticmethod
    def _validate_confirmation_request(
        *,
        action: ActionCandidate,
        certificate: ActionCertificate,
        policy: PolicyAcceptanceRecord,
        realization: Realization,
        fact_digest: str,
        evidence_digest: str,
        plan_digest: str,
        nonce: str,
    ) -> None:
        if certificate.outcome != "require_confirmation":
            raise BindingMismatchError(
                "only require_confirmation can issue a confirmation token"
            )
        if (
            certificate.prior_certificate_digest is not None
            or certificate.confirmation_binding_digest is not None
        ):
            raise BindingMismatchError(
                "an initial confirmation request cannot carry a prior grant binding"
            )
        if not policy.accepted:
            raise BindingMismatchError("policy acceptance record is not accepted")
        if not realization.available:
            raise BindingMismatchError("realization is unavailable")
        if certificate.nonce is not None and certificate.nonce != nonce:
            raise BindingMismatchError("nonce binding mismatch")
        expected = {
            "policy": (certificate.policy_acceptance_digest, policy.digest),
            "action": (certificate.action_digest, action.digest),
            "fact": (certificate.fact_digest, fact_digest),
            "evidence": (certificate.evidence_digest, evidence_digest),
            "plan": (certificate.plan_digest, plan_digest),
            "realization plan": (realization.plan.digest, plan_digest),
            "realization": (certificate.realization_digest, realization.digest),
        }
        for name, (actual, wanted) in expected.items():
            if actual != wanted:
                raise BindingMismatchError(f"{name} binding mismatch")

    @staticmethod
    def _result_digest(result: Any) -> str:
        try:
            return sha256_json(result)
        except (TypeError, ValueError):
            return sha256_json(
                {"type": f"{type(result).__module__}.{type(result).__qualname__}", "repr": repr(result)}
            )

    def _validate_certificate_authenticity(
        self,
        certificate: ActionCertificate,
    ) -> None:
        if not self._certificate_issuer.verify(certificate):
            raise BindingMismatchError(
                "certificate lacks a valid trusted adjudicator issuance proof"
            )

    def _validate_loaded_policy(
        self,
        policy: PolicyAcceptanceRecord,
        realization: Realization,
    ) -> None:
        if policy.digest != self.policy_acceptance.digest:
            raise BindingMismatchError(
                "policy acceptance is not the ToolGate replay-accepted artifact"
            )
        matching = []
        for template in self.policy_spec.realizations:
            candidate = self.realizer.materialize(template) if self.realizer is not None else template
            if (
                candidate.plan == realization.plan
                and candidate.capabilities == realization.capabilities
                and candidate.actual_audit == realization.actual_audit
                and candidate.control_roots == realization.control_roots
                and candidate.utility_vector == realization.utility_vector
                and candidate.available == realization.available
            ):
                matching.append(template)
        if not matching:
            raise BindingMismatchError(
                "realization is not declared by the replay-checked policy"
            )
        allowed_ids = {
            value
            for item in matching
            for value in (
                item.realization_id,
                f"runtime:{item.realization_id}",
                self.realizer.materialize(item).realization_id if self.realizer is not None else item.realization_id,
            )
        }
        if realization.realization_id not in allowed_ids:
            raise BindingMismatchError(
                "realization identity is not declared by the replay-checked policy"
            )

    @staticmethod
    def _fail(
        reason: str,
        *,
        nonce: str | None = None,
        decision: PublicDecision = "block",
    ) -> GateResult:
        return GateResult(
            decision=decision,
            state="blocked",
            dispatched=False,
            nonce=nonce,
            reason=reason,
        )


__all__ = ["Dispatcher", "GateResult", "ToolGate"]
