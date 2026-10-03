"""Strict OpenAI-compatible victim transport."""
from __future__ import annotations

import os


def create_transport(config):
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.agent_pipeline.llms import openai_llm as adapter
    from openai import OpenAI
    from obligate.eval.model_accounting import instrument_client

    class OpenAICompatibleVictim(BasePipelineElement):
        def __init__(self):
            self.name = self.model = config["model"]
            key = os.environ.get(config["api_key_env"])
            if not key:
                raise RuntimeError(f"Set {config['api_key_env']} before running model episodes")
            options = {"api_key": key, "timeout": config["timeout"], "max_retries": 0}
            if config.get("base_url"):
                options["base_url"] = config["base_url"]
            self.client = instrument_client(OpenAI(**options))

        def query(self, query, runtime, env=None, messages=(), extra_args=None):
            converted = [adapter._message_to_openai(message, self.model) for message in messages]
            if config["developer_role"] == "system":
                for message in converted:
                    if message["role"] == "developer":
                        message["role"] = "system"
            options = {"model": self.model, "messages": converted}
            tools = [adapter._function_to_openai(tool) for tool in runtime.functions.values()]
            if tools:
                options.update(tools=tools, tool_choice="auto")
            if config.get("temperature") is not None:
                options["temperature"] = config["temperature"]
            if config.get("extra_body"):
                options["extra_body"] = config["extra_body"]
            completion = self.client.chat.completions.create(**options)
            if len(completion.choices) != 1:
                raise ValueError("Victim completion must contain exactly one choice")
            finish_reason = completion.choices[0].finish_reason
            if finish_reason not in {"stop", "tool_calls", "function_call"}:
                raise ValueError(f"Victim completion terminated with {finish_reason}")
            raw = completion.choices[0].message
            if not raw.tool_calls and not raw.content:
                raise ValueError("Victim completion has no content or tool calls")
            calls = None if raw.tool_calls is None else [adapter._openai_to_tool_call(call) for call in raw.tool_calls]
            if any(not isinstance(call.args, dict) for call in calls or []):
                raise ValueError("Victim tool arguments must be a JSON object")
            output = {"role": "assistant", "content": adapter._assistant_message_to_content(raw), "tool_calls": calls}
            return query, runtime, env, [*messages, output], extra_args or {}

        def close(self):
            self.client.close()

    return OpenAICompatibleVictim()
