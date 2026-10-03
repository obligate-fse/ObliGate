from __future__ import annotations

from .banking import BankingPolicyEngine
from .slack import SlackPolicyEngine
from .travel import TravelPolicyEngine
from .workspace import WorkspacePolicyEngine

DEFAULT_POLICY_ENGINES = [
    BankingPolicyEngine(),
    TravelPolicyEngine(),
    SlackPolicyEngine(),
    WorkspacePolicyEngine(),
]

__all__ = ["DEFAULT_POLICY_ENGINES", "BankingPolicyEngine", "TravelPolicyEngine", "SlackPolicyEngine", "WorkspacePolicyEngine"]
