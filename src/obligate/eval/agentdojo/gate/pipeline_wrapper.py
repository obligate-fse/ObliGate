from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from ..compat.types import ToolCallContext
from .tool_firewall import AgentDojoToolFirewall


def wrap_tool_function(
    *,
    firewall: AgentDojoToolFirewall,
    suite: str,
    func: Callable[..., Any],
    tool_name: str | None = None,
    task_context: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    name = tool_name or getattr(func, "__name__", "unknown_tool")
    base = dict(task_context or {})

    @wraps(func)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        tool_args = dict(kwargs)
        if args:
            tool_args["_positional_args"] = args
        context = ToolCallContext(
            suite=suite,
            tool_name=name,
            tool_args=tool_args,
            user_task=str(base.get("user_task", "")),
            user_task_id=base.get("user_task_id"),
            injection_task_id=base.get("injection_task_id"),
            allowed_tools=set(base.get("allowed_tools", [])),
            allowed_groups=set(base.get("allowed_groups", [])),
            attack_goal_signatures=list(base.get("attack_goal_signatures", [])),
            run_id=str(base.get("run_id", "agentdojo_run")),
            sample_id=base.get("sample_id"),
        )
        decision = firewall.guard_before_tool(context)
        if not decision.execute:
            return decision.safe_result
        positional = tool_args.pop("_positional_args", ())
        raw = func(*positional, **tool_args)
        return firewall.observe_after_tool(context, raw)

    return guarded


class RuntimeToolCallAdapter:
    def __init__(self, runtime: Any, firewall: AgentDojoToolFirewall, *, suite: str, task_context: dict[str, Any] | None = None) -> None:
        self.runtime = runtime
        self.firewall = firewall
        self.suite = suite
        self.task_context = dict(task_context or {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runtime, name)

    def call_function(self, tool_name: str, *args: Any, **kwargs: Any) -> Any:
        return self._guard_and_call("call_function", tool_name, *args, **kwargs)

    def run_function(self, tool_name: str, *args: Any, **kwargs: Any) -> Any:
        return self._guard_and_call("run_function", tool_name, *args, **kwargs)

    def execute(self, tool_name: str, *args: Any, **kwargs: Any) -> Any:
        return self._guard_and_call("execute", tool_name, *args, **kwargs)

    def _guard_and_call(self, method_name: str, tool_name: str, *args: Any, **kwargs: Any) -> Any:
        tool_args = dict(kwargs)
        if args:
            tool_args["_positional_args"] = args
        context = ToolCallContext(
            suite=self.suite,
            tool_name=str(tool_name),
            tool_args=tool_args,
            user_task=str(self.task_context.get("user_task", "")),
            allowed_tools=set(self.task_context.get("allowed_tools", [])),
            allowed_groups=set(self.task_context.get("allowed_groups", [])),
            attack_goal_signatures=list(self.task_context.get("attack_goal_signatures", [])),
            run_id=str(self.task_context.get("run_id", "agentdojo_run")),
            sample_id=self.task_context.get("sample_id"),
        )
        decision = self.firewall.guard_before_tool(context)
        if not decision.execute:
            return decision.safe_result
        method = getattr(self.runtime, method_name)
        raw = method(tool_name, *args, **kwargs)
        return self.firewall.observe_after_tool(context, raw)

