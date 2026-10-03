"""Concrete backend bindings for abstract policy realizations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Protocol

from .model import Realization


@dataclass(frozen=True, slots=True)
class BackendBinding:
    """One deployer-supplied implementation of a policy capability."""

    capability: str
    backend_id: str
    available: bool = True
    production_verified: bool = False

    def __post_init__(self) -> None:
        if not self.capability or not self.backend_id:
            raise ValueError("realizer backend bindings require capability and backend_id")
        if self.backend_id.casefold().startswith(("oracle:", "scorer:")):
            raise ValueError("oracle/scorer components cannot be realizer backends")


class Realizer(Protocol):
    def materialize(self, template: Realization) -> Realization: ...

    def manifest(self) -> Mapping[str, object]: ...


class BoundRealizer:
    """Materialize plans only when every named backend is bound and available."""

    def __init__(
        self,
        bindings: Iterable[BackendBinding],
        *,
        bind_control_roots: bool = True,
    ) -> None:
        values = tuple(bindings)
        by_capability = {item.capability: item for item in values}
        if len(by_capability) != len(values):
            raise ValueError("realizer capabilities must have exactly one backend binding")
        self._bindings = by_capability
        self._bind_control_roots = bind_control_roots

    @classmethod
    def from_capabilities(cls, capabilities: Iterable[str]) -> "BoundRealizer":
        """Compatibility constructor for tests and single-process benchmarks.

        These bindings are explicit but not production verified; the manifest
        keeps ``RealizerRefines`` in the external TCB.
        """

        return cls(
            (
                BackendBinding(
                    capability=str(capability),
                    backend_id=f"declared/{capability}",
                    production_verified=False,
                )
                for capability in capabilities
            ),
            bind_control_roots=False,
        )

    def materialize(self, template: Realization) -> Realization:
        required = set(template.capabilities)
        bound = {
            capability
            for capability in required
            if (binding := self._bindings.get(capability)) is not None and binding.available
        }
        available = template.available and required <= bound
        actual_audit = template.actual_audit if "audit" in bound else frozenset()
        roots = (
            tuple(
                (
                    control,
                    self._bound_root(root),
                )
                for control, root in template.control_roots
            )
            if self._bind_control_roots
            else template.control_roots
        )
        return replace(
            template,
            realization_id=f"runtime:{template.realization_id}",
            capabilities=frozenset(bound),
            actual_audit=actual_audit,
            control_roots=roots,
            available=available,
        )

    def manifest(self) -> Mapping[str, object]:
        bindings = [
            {
                "capability": capability,
                "backend_id": binding.backend_id,
                "available": binding.available,
                "production_verified": binding.production_verified,
            }
            for capability, binding in sorted(self._bindings.items())
        ]
        return {
            "schema": "obligate.realizer-bindings/v1",
            "bindings": bindings,
            "control_roots_bound": self._bind_control_roots,
            "production_verified": bool(bindings) and all(
                binding.production_verified and binding.available
                for binding in self._bindings.values()
            ),
        }

    def _bound_root(self, root: str) -> str:
        if not root.startswith("backend:"):
            return root
        capability = root.removeprefix("backend:")
        binding = self._bindings.get(capability)
        if binding is None:
            return f"backend:unbound/{capability}"
        return f"backend:{binding.backend_id}"


__all__ = ["BackendBinding", "BoundRealizer", "Realizer"]
