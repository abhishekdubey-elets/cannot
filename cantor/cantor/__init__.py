"""CANTOR — Cognitive Architecture for Neural Thought, Orchestration, and Reasoning.

Reference implementation of the architecture designed in
``human-reasoning-to-llm-architecture.md`` (Part 10). Each submodule maps to a
report section; see README.md for the docs-to-code table.
"""

from .agent import CantorAgent
from .llm import MockLLM

__all__ = ["CantorAgent", "MockLLM"]
