"""
Configuration loader.

LitReviewAgents is YAML-driven. A single config file defines:
- The manuscript context (paper title, claim, weaknesses)
- LLM backend (any OpenAI-compatible endpoint)
- Paper sources (arxiv, crossref, semantic_scholar, pubmed)
- Agents — each with queries, system prompt template, and user prompt template

See `examples/iot_compliance.yaml` for a full annotated example.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "PyYAML is required. Install with: pip install pyyaml"
    ) from e


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5:7b"
    api_key: str = "ollama"
    timeout: int = 600
    max_retries: int = 1
    retry_wait: int = 20
    temperature: float = 0.2
    extra_options: dict = field(default_factory=dict)
    strip_preamble: bool = True


@dataclass
class AgentConfig:
    """
    Configuration for a single LiteratureAgent.

    Example YAML:
        - name: RelatedWorkAgent
          description: "Searches for comparable middleware frameworks"
          queries:
            - "FIWARE IoT platform security middleware"
            - "Eclipse Ditto IoT device management"
          system_prompt: "You are a rigorous academic reviewer..."
          user_prompt: |
            These papers describe IoT middleware comparable to ours.
            {papers}
            For each relevant paper: ...
          max_tokens: 1000
    """
    name: str
    description: str = ""
    queries: list = field(default_factory=list)
    system_prompt: str = "You are a rigorous academic reviewer."
    user_prompt: str = "Analyze these papers:\n\n{papers}"
    max_tokens: int = 800
    artifact_type: Optional[str] = None  # defaults to name_papers / name_analysis


@dataclass
class SynthesisConfig:
    """Configuration for the final synthesis step (optional)."""
    enabled: bool = True
    sections: list = field(default_factory=list)
    research_questions_prompt: Optional[str] = None
    discussion_prompt: Optional[str] = None
    max_tokens: int = 500


@dataclass
class Config:
    """Top-level project configuration."""

    # Project identity
    project_name: str
    paper_context: str          # full context for synthesis
    short_context: str          # trimmed context for per-call prompts

    # Backends
    llm: LLMConfig
    sources: dict               # {source_name: bool}

    # Agents
    agents: list                # list[AgentConfig]

    # Output paths
    output_dir: Path
    memory_dir: Path

    # Synthesis & validation
    synthesis: SynthesisConfig
    enable_relevance_agent: bool = True
    enable_validator: bool = True

    # Search behavior
    max_results_per_query: int = 4
    relevance_cap: int = 20

    @classmethod
    def from_yaml(cls, path) -> "Config":
        """Load configuration from a YAML file."""
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f)

        # Resolve relative paths against the config file's parent directory
        base_dir = path.parent

        def resolve_path(p):
            p = Path(p).expanduser()
            return p if p.is_absolute() else (base_dir / p).resolve()

        # LLM section — environment variable substitution
        llm_raw = raw.get("llm", {})
        llm = LLMConfig(
            base_url=os.environ.get("LITREVIEW_BASE_URL", llm_raw.get("base_url", "http://localhost:11434/v1")),
            model=os.environ.get("LITREVIEW_MODEL", llm_raw.get("model", "qwen2.5:7b")),
            api_key=os.environ.get("LITREVIEW_API_KEY", llm_raw.get("api_key", "ollama")),
            timeout=llm_raw.get("timeout", 600),
            max_retries=llm_raw.get("max_retries", 1),
            retry_wait=llm_raw.get("retry_wait", 20),
            temperature=llm_raw.get("temperature", 0.2),
            extra_options=llm_raw.get("extra_options", {}),
            strip_preamble=llm_raw.get("strip_preamble", True),
        )

        # Sources — default to arxiv + crossref enabled
        sources_default = {"arxiv": True, "crossref": True,
                           "semantic_scholar": False, "pubmed": False}
        sources = {**sources_default, **raw.get("sources", {})}

        # Agents
        agents = []
        for a in raw.get("agents", []):
            agents.append(AgentConfig(
                name=a["name"],
                description=a.get("description", ""),
                queries=a.get("queries", []),
                system_prompt=a.get("system_prompt", "You are a rigorous academic reviewer."),
                user_prompt=a.get("user_prompt", "Analyze these papers:\n\n{papers}"),
                max_tokens=a.get("max_tokens", 800),
                artifact_type=a.get("artifact_type"),
            ))

        # Synthesis
        syn_raw = raw.get("synthesis", {})
        synthesis = SynthesisConfig(
            enabled=syn_raw.get("enabled", True),
            sections=syn_raw.get("sections", []),
            research_questions_prompt=syn_raw.get("research_questions_prompt"),
            discussion_prompt=syn_raw.get("discussion_prompt"),
            max_tokens=syn_raw.get("max_tokens", 500),
        )

        return cls(
            project_name=raw["project_name"],
            paper_context=raw.get("paper_context", ""),
            short_context=raw.get("short_context", ""),
            llm=llm,
            sources=sources,
            agents=agents,
            output_dir=resolve_path(raw.get("output_dir", "./output")),
            memory_dir=resolve_path(raw.get("memory_dir", "./memory")),
            synthesis=synthesis,
            enable_relevance_agent=raw.get("enable_relevance_agent", True),
            enable_validator=raw.get("enable_validator", True),
            max_results_per_query=raw.get("max_results_per_query", 4),
            relevance_cap=raw.get("relevance_cap", 20),
        )

    def validate(self):
        """Raise ValueError if configuration is invalid."""
        errors = []
        if not self.project_name:
            errors.append("project_name is required")
        if not self.agents:
            errors.append("at least one agent must be defined")
        if not any(self.sources.values()):
            errors.append("at least one source must be enabled")
        for a in self.agents:
            if not a.name:
                errors.append("every agent needs a name")
            if not a.queries:
                errors.append(f"agent '{a.name}' has no queries")
            if "{papers}" not in a.user_prompt:
                errors.append(f"agent '{a.name}' user_prompt must contain {{papers}}")
        if errors:
            raise ValueError("Configuration errors:\n  - " + "\n  - ".join(errors))
