"""Compatibility layer for the ``deepagents`` package.

AGENTS.md prescribes this usage pattern::

    from deepagents import Agent
    agent = Agent(name="...", llm=get_llm(), system_prompt="...", tools=[...])
    raw = agent.run(prompt)

Recent ``deepagents`` releases (>= 0.6) expose ``create_deep_agent`` instead
of an ``Agent`` class. This module re-exports ``Agent`` directly when the
installed package provides it, and otherwise defines an adapter with the
exact same constructor and synchronous ``run()`` interface.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)

try:  # Older / alternative deepagents releases that ship an Agent class.
    from deepagents import Agent  # type: ignore[attr-defined]
except ImportError:
    from deepagents import create_deep_agent

    class Agent:  # type: ignore[no-redef]
        """Adapter exposing the AGENTS.md ``Agent`` interface over ``create_deep_agent``."""

        def __init__(
            self,
            name: str,
            llm: Any,
            system_prompt: str,
            tools: Sequence[Any] | None = None,
        ) -> None:
            self.name = name
            self._graph = create_deep_agent(
                model=llm,
                tools=list(tools or []),
                system_prompt=system_prompt,
                name=name,
            )

        def run(self, prompt: str) -> str:
            """Run the agent synchronously and return the final message text."""
            result = self._graph.invoke(
                {"messages": [{"role": "user", "content": prompt}]}
            )
            messages = result.get("messages", [])
            if not messages:
                logger.warning("Agent '%s' returned no messages", self.name)
                return ""
            content = messages[-1].content
            if isinstance(content, list):
                # Content-block format: join all text blocks.
                content = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            return str(content)


__all__ = ["Agent"]
