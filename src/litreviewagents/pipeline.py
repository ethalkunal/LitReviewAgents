"""
Pipeline orchestrator.

Wires together: config + LLM client + sources + memory + agents + report.

Usage:
    config = Config.from_yaml("my_paper.yaml")
    pipeline = Pipeline(config)
    pipeline.run()

Custom agents:
    pipeline = Pipeline(config)
    pipeline.add_agent(MyCustomAgent(), position=2)
    pipeline.run()
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .agents import Agent, LiteratureAgent, RelevanceAgent, SynthesisAgent, ValidatorAgent
from .config import Config
from .llm import LLMClient
from .memory import PaperMemory
from .report import ArtifactStore, Report, fmt_papers_full
from .sources import get_source


@dataclass
class PipelineContext:
    """Shared state passed to every agent during a run."""

    config: Config
    llm: LLMClient
    memory: PaperMemory
    store: ArtifactStore
    last_uid: dict = field(default_factory=dict)
    synthesis_results: dict = field(default_factory=dict)
    validation_notes: str = ""

    def fetch_all(self, queries):
        """
        Fetch papers from all enabled sources for the given queries.

        Dedupes by title prefix, filters out accepted/rejected, sorts by
        citation count (None last, descending).
        """
        seen, all_papers = set(), []
        accepted_urls = self.memory.load_accepted_urls()
        rejected_urls = self.memory.load_rejected_urls()
        skipped_acc, skipped_rej = 0, 0

        active = [s for s, on in self.config.sources.items() if on]
        print(f"    Sources active: {', '.join(active)}")

        max_per = self.config.max_results_per_query

        for q in queries:
            print(f"    Querying: {q[:65]}...")
            batch = []
            for src_name in active:
                src = get_source(src_name)
                try:
                    batch += src.search(q, max_per)
                except Exception as e:
                    print(f"      [{src_name}] error: {e}")
                # Rate-limit pause between sources
                time.sleep(0.5 if src_name in ("arxiv", "crossref") else 1.0)

            for p in batch:
                key = p.title.lower()[:60]
                if key in seen:
                    continue
                if p.url in accepted_urls:
                    skipped_acc += 1
                    continue
                if p.url in rejected_urls:
                    skipped_rej += 1
                    continue
                seen.add(key)
                all_papers.append(p)

        all_papers.sort(key=lambda p: p.citations if p.citations is not None else -1, reverse=True)

        if skipped_acc:
            print(f"    Skipped {skipped_acc} already-accepted paper(s)")
        if skipped_rej:
            print(f"    Skipped {skipped_rej} rejected paper(s)")

        src_counts = {}
        for p in all_papers:
            s = p.source.split(" ")[0]
            src_counts[s] = src_counts.get(s, 0) + 1
        print(f"    Source breakdown: {src_counts}")

        return all_papers


class Pipeline:
    """High-level orchestrator."""

    def __init__(self, config: Config):
        config.validate()
        self.config = config

        self.llm = LLMClient(
            base_url=config.llm.base_url,
            model=config.llm.model,
            api_key=config.llm.api_key,
            timeout=config.llm.timeout,
            max_retries=config.llm.max_retries,
            retry_wait=config.llm.retry_wait,
            temperature=config.llm.temperature,
            extra_options=config.llm.extra_options,
            strip_preamble=config.llm.strip_preamble,
        )

        self.memory = PaperMemory(config.memory_dir)

        # Build default agent chain from config
        self.agents = [LiteratureAgent(a) for a in config.agents]

        # Wire parent UIDs — each agent chains to the previous one for lineage
        prev = None
        for a in self.agents:
            a.parent_uid_key = prev
            prev = a.name

    def add_agent(self, agent: Agent, position: int = None):
        """Insert a custom agent into the pipeline. Position defaults to end."""
        if position is None:
            self.agents.append(agent)
        else:
            self.agents.insert(position, agent)

    def run(self):
        session_id = (
            f"{self.config.project_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        session_dir = Path(self.config.output_dir) / session_id
        store = ArtifactStore(session_dir)
        ctx = PipelineContext(
            config=self.config,
            llm=self.llm,
            memory=self.memory,
            store=store,
        )

        active_sources = ", ".join(s for s, on in self.config.sources.items() if on)
        total_queries = sum(len(a.queries) for a in self.config.agents)

        print(f"\n{'=' * 62}")
        print(f"  LitReviewAgents — {self.config.project_name}")
        print(f"  Session : {session_id}")
        print(f"  Model   : {self.config.llm.model} @ {self.config.llm.base_url}")
        print(f"  Sources : {active_sources}")
        print(f"  Agents  : {len(self.agents)} ({total_queries} queries total)")
        print(f"  Output  : {session_dir}")
        print(f"{'=' * 62}\n")

        # Run literature/custom agents
        for agent in self.agents:
            agent.run(ctx)

        # Optional relevance evaluation
        if self.config.enable_relevance_agent:
            RelevanceAgent().run(ctx)

        # Optional synthesis
        if self.config.synthesis.enabled:
            SynthesisAgent().run(ctx)

        # Optional validation
        if self.config.enable_validator:
            ValidatorAgent().run(ctx)

        # Assemble final report
        results = dict(ctx.synthesis_results)

        rel_arts = store.get_by_type("relevance_report")
        if rel_arts:
            results["0. Preliminary Relevance Report (Review Before Accepting)"] = rel_arts[0]["content"]

        if ctx.validation_notes:
            results[f"{len(results) + 1}. Validation Checklist"] = ctx.validation_notes

        # Append raw paper lists at the end
        letter = ord("A")
        for art in store.artifacts.values():
            if art["type"].endswith("_papers"):
                results[f"{chr(letter)}. Raw Papers: {art['agent']}"] = fmt_papers_full(art["content"])
                letter += 1

        report = Report(session_dir, self.config.project_name, self.config.llm.model)
        report_path = report.write(results)

        print(f"\n{'=' * 62}")
        print(f"  Complete: {len(store.artifacts)} artifacts | Report: {report_path}")
        print(f"{'=' * 62}\n")

        return report_path
