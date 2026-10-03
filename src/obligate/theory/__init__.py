"""Theory-aligned ObliGate v1 kernel.

The legacy policy engine remains available as a compatibility facade.  New
integrations should use this package's finite checker, witness engine, runtime,
and ToolGate.
"""

from .checker import CheckReport, FinitePolicyChecker, check_default_policy, check_policy
from .fact_extraction import ObservableFactExtractor, oracle_free_tool_context
from .gate import GateResult, ToolGate
from .issuer import CertificateIssuer
from .lattice import EnforceableControlPlanner
from .model import (
    THEORY_SEMANTIC_VERSION,
    ActionCandidate,
    ActionCertificate,
    ActionGraph,
    BehaviorPlan,
    PolicyAcceptanceRecord,
    TheoryDecision,
)
from .policy import PolicySpec, build_default_policy
from .realizer import BackendBinding, BoundRealizer, Realizer
from .runtime import RuntimeEvidence, TheoryRuntime
from .task_contract import TaskContractCompiler, TrustedTaskSource
from .witness import BipolarWitnessEngine, WitnessClosure

__all__ = [
    "THEORY_SEMANTIC_VERSION",
    "ActionCandidate",
    "ActionCertificate",
    "ActionGraph",
    "BehaviorPlan",
    "BackendBinding",
    "BipolarWitnessEngine",
    "BoundRealizer",
    "CheckReport",
    "CertificateIssuer",
    "EnforceableControlPlanner",
    "FinitePolicyChecker",
    "GateResult",
    "ObservableFactExtractor",
    "PolicyAcceptanceRecord",
    "PolicySpec",
    "Realizer",
    "RuntimeEvidence",
    "TaskContractCompiler",
    "TheoryDecision",
    "TheoryRuntime",
    "ToolGate",
    "TrustedTaskSource",
    "WitnessClosure",
    "build_default_policy",
    "check_default_policy",
    "check_policy",
    "oracle_free_tool_context",
]
