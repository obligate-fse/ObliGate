"""Trusted-task compilation for the canonical theory runtime.

The contract compiler is deliberately small.  It does not infer authority from
tool output, benchmark labels, task identifiers, or a policy decision.  A
benchmark adapter first derives candidate claims from the trusted user task and
observable provenance, then this module makes the three-way contract total for
the action's authorization requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .model import SignedAtom, TaskContract, is_authorization_predicate, sha256_json


@dataclass(frozen=True, slots=True)
class TrustedTaskSource:
    """The only authority-bearing source accepted by ``TaskContractCompiler``."""

    instruction: str
    channel: str = "user"

    def __post_init__(self) -> None:
        if self.channel != "user":
            raise ValueError("TaskContract authority must originate from the trusted user channel")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "contract_schema": "obligate.task-contract/v1",
                "channel": self.channel,
                "instruction": self.instruction,
            }
        )


@dataclass(frozen=True, slots=True)
class TaskContractBuild:
    contract: TaskContract
    required_claims: tuple[str, ...]

    def to_trace(self) -> dict[str, object]:
        return {
            "schema": "obligate.task-contract/v1",
            "source_digest": self.contract.source_digest,
            "required_claim_count": len(self.required_claims),
            "permit_count": len(self.contract.permit),
            "deny_count": len(self.contract.deny),
            "unresolved_count": len(self.contract.unresolved),
        }


class TaskContractCompiler:
    """Compile ``C_U=(Permit, Deny, Unresolved)`` over one action.

    ``permit`` and ``deny`` are claims already derived from the trusted task
    parser.  Everything else is unresolved.  Unknown or overlapping claims are
    rejected instead of being silently accepted.
    """

    def compile(
        self,
        *,
        source: TrustedTaskSource,
        requirements: Iterable[SignedAtom],
        permit: Iterable[str] = (),
        deny: Iterable[str] = (),
    ) -> TaskContractBuild:
        required = frozenset(
            atom.unsigned_key
            for atom in requirements
            if is_authorization_predicate(atom.predicate)
        )
        permitted = frozenset(str(item) for item in permit)
        denied = frozenset(str(item) for item in deny)
        unknown = (permitted | denied) - required
        if unknown:
            raise ValueError(f"TaskContract contains claims outside the candidate requirements: {sorted(unknown)}")
        if permitted & denied:
            raise ValueError("TaskContract permit and deny claims overlap")
        unresolved = required - permitted - denied
        contract = TaskContract(
            permit=permitted,
            deny=denied,
            unresolved=unresolved,
            source_digest=source.digest,
        )
        return TaskContractBuild(contract=contract, required_claims=tuple(sorted(required)))


__all__ = ["TaskContractBuild", "TaskContractCompiler", "TrustedTaskSource"]
