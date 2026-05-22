"""
Command-line interface for LitReviewAgents.

    litreviewagents run --config my_paper.yaml
    litreviewagents init my_paper.yaml          # create a starter config
    litreviewagents accept --config my_paper.yaml --url ... --title ...
    litreviewagents reject --config my_paper.yaml --url ... --title ...
    litreviewagents list --config my_paper.yaml
    litreviewagents list-rejected --config my_paper.yaml
    litreviewagents clear-seen --config my_paper.yaml
"""

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import Config
from .memory import PaperMemory
from .pipeline import Pipeline

STARTER_CONFIG = """# LitReviewAgents — Project Configuration
# ----------------------------------------
# Fill in your manuscript context and define agents below.
# Run with:  litreviewagents run --config <this-file>.yaml

project_name: "My Paper Project"

# Full context — used in the final synthesis prompts.
paper_context: |
  Paper: "<your paper title>"
  Authors: <authors>
  Journal: <target journal>, under revision

  CORE CLAIM: <one-sentence core claim>

  CRITICAL WEAKNESSES IDENTIFIED FOR REVISION:
  1. <reviewer point 1>
  2. <reviewer point 2>

# Trimmed context — used per LLM call (keeps prompts cheap).
short_context: |
  Paper on <topic>. Core claim: <claim>. Weaknesses: <key gaps>.

# ── LLM backend ──────────────────────────────────────────────────────────────
# Any OpenAI-compatible endpoint works:
#   Ollama:    http://localhost:11434/v1   model: qwen2.5:7b
#   vLLM:      http://localhost:8000/v1
#   LM Studio: http://localhost:1234/v1
#   OpenAI:    https://api.openai.com/v1   model: gpt-4o-mini   api_key: $OPENAI_API_KEY
#   Groq:      https://api.groq.com/openai/v1
#
# Env overrides: LITREVIEW_BASE_URL, LITREVIEW_MODEL, LITREVIEW_API_KEY
llm:
  base_url: "http://localhost:11434/v1"
  model: "qwen2.5:7b"
  api_key: "ollama"
  timeout: 600
  max_retries: 1
  temperature: 0.2
  # For Ollama qwen3 only — disables visible chain-of-thought:
  # extra_options: {think: false}

# ── Paper sources ────────────────────────────────────────────────────────────
sources:
  arxiv: true
  crossref: true
  semantic_scholar: false   # set true and export SEMANTIC_SCHOLAR_API_KEY for high-volume runs
  pubmed: false             # biomedical focus

# ── Output ───────────────────────────────────────────────────────────────────
output_dir: "./output"
memory_dir: "./memory"

# ── Search behavior ──────────────────────────────────────────────────────────
max_results_per_query: 4
relevance_cap: 20

# ── Agents ───────────────────────────────────────────────────────────────────
# Define one agent per literature gap your paper has.
# Each agent gets its own set of queries and prompt templates.
# The {papers} placeholder is required in user_prompt; {context} optional in system_prompt.

agents:
  - name: RelatedWorkAgent
    description: "Searches for comparable prior work"
    queries:
      - "your topic prior art query 1"
      - "your topic prior art query 2"
    system_prompt: "You are a rigorous academic reviewer. Context: {context}"
    user_prompt: |
      These papers are candidates for the Related Work section.

      {papers}

      For each relevant paper:
      1. Summarize its contribution in one sentence.
      2. Explain how our paper differs.
      3. Draft a single citation-ready sentence.
    max_tokens: 1000

  - name: MethodsAgent
    description: "Finds methodology justifications"
    queries:
      - "<your methodology> empirical justification"
    system_prompt: "You are a methodology expert. Context: {context}"
    user_prompt: |
      We need to justify our methodology. Use these papers:

      {papers}

      Draft a 150-word Methods justification paragraph.
    max_tokens: 800

# ── Synthesis ────────────────────────────────────────────────────────────────
synthesis:
  enabled: true
  research_questions_prompt: |
    Respond immediately with the following — no preamble:

    **RQ1**: [Specific question your gap analysis answers]
    **RQ2**: [Specific question your method tests]
    **RQ3**: [Specific question your results address]

    Each RQ must be answerable by your experiments. Keep them sharp.
  sections:
    - title: "Related Work Comparison"
      source_agent: RelatedWorkAgent
    - title: "Methods Justification"
      source_agent: MethodsAgent

enable_relevance_agent: true
enable_validator: true
"""


def cmd_init(args):
    path = Path(args.config).expanduser().resolve()
    if path.exists() and not args.force:
        print(f"  Refusing to overwrite existing config: {path}")
        print("  Use --force to overwrite.")
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(STARTER_CONFIG)
    print(f"  Starter config written to: {path}")
    print(f"  Edit it, then run:  litreviewagents run --config {path.name}")
    return 0


def cmd_run(args):
    config = Config.from_yaml(args.config)
    pipeline = Pipeline(config)
    pipeline.run()
    return 0


def _memory_from_config(config_path):
    config = Config.from_yaml(config_path)
    return PaperMemory(config.memory_dir)


def cmd_accept(args):
    mem = _memory_from_config(args.config)
    mem.add_accepted(
        url=args.url,
        title=args.title,
        authors=[a.strip() for a in args.authors.split(",")],
        date=args.date,
        section=args.section,
        notes=args.notes,
    )
    return 0


def cmd_reject(args):
    mem = _memory_from_config(args.config)
    mem.add_rejected(url=args.url, title=args.title, reason=args.reason)
    return 0


def cmd_list(args):
    _memory_from_config(args.config).list_accepted()
    return 0


def cmd_list_rejected(args):
    _memory_from_config(args.config).list_rejected()
    return 0


def cmd_clear_seen(args):
    _memory_from_config(args.config).clear_seen()
    return 0


def cmd_show_config(args):
    config = Config.from_yaml(args.config)
    print(f"\n  Project: {config.project_name}")
    print(f"  Output : {config.output_dir}")
    print(f"  Memory : {config.memory_dir}")
    print(f"  Model  : {config.llm.model} @ {config.llm.base_url}")
    print(f"  Sources: {[s for s, on in config.sources.items() if on]}")
    print(f"  Agents : {len(config.agents)}")
    for a in config.agents:
        print(f"    - {a.name} ({len(a.queries)} queries)")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="litreviewagents",
        description="Multi-agent literature search and synthesis for academic manuscripts.",
    )
    p.add_argument("--version", action="version", version=f"litreviewagents {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create a starter config YAML")
    p_init.add_argument("config", help="Path to write the new config file")
    p_init.add_argument("--force", action="store_true", help="Overwrite if file exists")
    p_init.set_defaults(func=cmd_init)

    p_run = sub.add_parser("run", help="Run the literature pipeline")
    p_run.add_argument("--config", "-c", required=True, help="Path to YAML config")
    p_run.set_defaults(func=cmd_run)

    p_show = sub.add_parser("show-config", help="Print loaded config summary")
    p_show.add_argument("--config", "-c", required=True)
    p_show.set_defaults(func=cmd_show_config)

    p_acc = sub.add_parser("accept", help="Mark a paper as accepted")
    p_acc.add_argument("--config", "-c", required=True)
    p_acc.add_argument("--url", required=True)
    p_acc.add_argument("--title", required=True)
    p_acc.add_argument("--authors", required=True, help="Comma-separated")
    p_acc.add_argument("--date", required=True, help="YYYY-MM-DD")
    p_acc.add_argument("--section", required=True, help="Manuscript section, e.g. 'Discussion 4.3'")
    p_acc.add_argument("--notes", default="")
    p_acc.set_defaults(func=cmd_accept)

    p_rej = sub.add_parser("reject", help="Permanently exclude a paper from future runs")
    p_rej.add_argument("--config", "-c", required=True)
    p_rej.add_argument("--url", required=True)
    p_rej.add_argument("--title", required=True)
    p_rej.add_argument("--reason", default="")
    p_rej.set_defaults(func=cmd_reject)

    p_list = sub.add_parser("list", help="List accepted papers")
    p_list.add_argument("--config", "-c", required=True)
    p_list.set_defaults(func=cmd_list)

    p_lr = sub.add_parser("list-rejected", help="List rejected papers")
    p_lr.add_argument("--config", "-c", required=True)
    p_lr.set_defaults(func=cmd_list_rejected)

    p_cs = sub.add_parser("clear-seen", help="Reset seen-papers log")
    p_cs.add_argument("--config", "-c", required=True)
    p_cs.set_defaults(func=cmd_clear_seen)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
