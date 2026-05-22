"""
LitReviewAgents — Multi-agent literature search and synthesis for academic manuscripts.

Public API:
    from litreviewagents import Pipeline, LiteratureAgent, Config, llm_call

Typical usage:
    config = Config.from_yaml("my_paper.yaml")
    pipeline = Pipeline(config)
    pipeline.run()

Custom agents:
    from litreviewagents import LiteratureAgent

    class MyCustomAgent(LiteratureAgent):
        name = "MyCustomAgent"
        def build_prompt(self, papers):
            return f"Custom prompt with {papers}"
"""

__version__ = "0.1.0"

from .agents import Agent, LiteratureAgent, RelevanceAgent, SynthesisAgent, ValidatorAgent
from .config import Config
from .llm import LLMClient, llm_call
from .memory import PaperMemory
from .pipeline import Pipeline
from .report import Report

__all__ = [
    "Agent",
    "LiteratureAgent",
    "SynthesisAgent",
    "RelevanceAgent",
    "ValidatorAgent",
    "Config",
    "LLMClient",
    "llm_call",
    "Pipeline",
    "PaperMemory",
    "Report",
]
