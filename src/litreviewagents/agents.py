"""
Agent classes.

Hierarchy:
    Agent (abstract)
      ├── LiteratureAgent   — searches sources, analyses results with LLM
      ├── RelevanceAgent    — per-paper relevance scoring
      ├── SynthesisAgent    — assembles final manuscript sections
      └── ValidatorAgent    — quality checks

Custom agents subclass Agent or LiteratureAgent and override `run()` or
the `build_*` methods. They're registered with Pipeline via Pipeline.add_agent().
"""

import time
from abc import ABC, abstractmethod

from .report import fmt_papers_compact


class Agent(ABC):
    """Base class for all agents."""

    name: str = "Agent"

    @abstractmethod
    def run(self, ctx):
        """
        Run the agent.

        Parameters
        ----------
        ctx : PipelineContext
            Shared pipeline context. See pipeline.py.

        Returns
        -------
        artifact_uid : str
            ID of the primary artifact produced.
        """
        ...


class LiteratureAgent(Agent):
    """
    Default literature-search agent.

    Reads queries from config, fetches papers via the active sources,
    formats them into the user_prompt template, and calls the LLM.

    Two artifacts are stored:
      - {name}_papers   — raw paper list
      - {name}_analysis — LLM analysis text
    """

    def __init__(self, agent_config, parent_uid_key=None):
        self.config = agent_config
        self.name = agent_config.name
        self.parent_uid_key = parent_uid_key  # which previous agent to chain to

    def fetch_papers(self, ctx):
        """Fetch papers from all enabled sources for this agent's queries."""
        return ctx.fetch_all(self.config.queries)

    def build_user_prompt(self, papers):
        """Render the user_prompt template with the {papers} placeholder."""
        return self.config.user_prompt.format(
            papers=fmt_papers_compact(papers),
        )

    def build_system_prompt(self, ctx):
        """System prompt — appends short_context if not already templated."""
        sp = self.config.system_prompt
        if "{context}" in sp:
            return sp.format(context=ctx.config.short_context)
        return sp

    def run(self, ctx):
        print(f"\n[{self.name}] {self.config.description or 'searching...'}")
        papers = self.fetch_papers(ctx)
        print(f"  Found {len(papers)} papers")

        papers_type = f"{self.name}_papers"
        parent_uid = ctx.last_uid.get(self.parent_uid_key) if self.parent_uid_key else None
        parent_ids = [parent_uid] if parent_uid else None

        uid = ctx.store.store(self.name, papers_type, papers, parent_ids=parent_ids)

        analysis = ctx.llm.chat(
            self.build_system_prompt(ctx),
            self.build_user_prompt(papers),
            max_tokens=self.config.max_tokens,
        )

        analysis_type = f"{self.name}_analysis"
        ctx.store.store(self.name, analysis_type, analysis, parent_ids=[uid])

        ctx.last_uid[self.name] = uid
        ctx.last_uid["_last"] = uid
        return uid


class RelevanceAgent(Agent):
    """
    Per-paper relevance evaluation.

    Iterates all unique papers gathered by previous LiteratureAgents,
    skips ones already evaluated (via PaperMemory seen list), and asks
    the LLM to score each paper for the project.
    """

    name = "RelevanceAgent"

    def __init__(self, batch_size=5, cooldown=10):
        self.batch_size = batch_size
        self.cooldown = cooldown

    def run(self, ctx):
        print(f"\n[{self.name}] Evaluating papers for manuscript relevance...")

        # Collect unique papers from all *_papers artifacts
        seen_titles, all_papers = set(), []
        for art in ctx.store.artifacts.values():
            if not art["type"].endswith("_papers"):
                continue
            for p in art["content"]:
                # content may be Paper dicts or original objects
                title = p["title"] if isinstance(p, dict) else p.title
                key = title.lower()[:60]
                if key not in seen_titles:
                    seen_titles.add(key)
                    all_papers.append(p if isinstance(p, dict) else p.to_dict())

        print(f"  {len(all_papers)} unique papers across agents")

        seen_urls = ctx.memory.load_seen_urls()
        new_papers = [p for p in all_papers if p["url"] not in seen_urls]
        skipped = len(all_papers) - len(new_papers)

        if skipped:
            print(f"  Skipped {skipped} previously evaluated paper(s)")

        if not new_papers:
            msg = (
                "FAILSAFE: All papers in this run have been evaluated in previous sessions.\n"
                "No new papers to review.\n\n"
                "To re-evaluate all papers: litreviewagents clear-seen"
            )
            print(f"  {msg}")
            uid = ctx.store.store(self.name, "relevance_report", msg)
            ctx.last_uid["_relevance"] = uid
            return uid

        ctx.memory.add_seen([p["url"] for p in new_papers])
        print(f"  Logged {len(new_papers)} URLs to seen-papers record")

        # Cap to most recent N
        papers_eval = sorted(new_papers, key=lambda p: p.get("date", "2000"), reverse=True)
        cap = ctx.config.relevance_cap
        if len(papers_eval) > cap:
            papers_eval = papers_eval[:cap]
            print(f"  Capped to {cap} most recent papers")

        # Batch through LLM
        entries = []
        for i in range(0, len(papers_eval), self.batch_size):
            batch = papers_eval[i:i + self.batch_size]
            batch_fmt = "\n\n".join([
                f"[{j + 1}] TITLE: {p['title']}\n"
                f"    DATE: {p['date']} | AUTHORS: {', '.join(p['authors'])}\n"
                f"    URL: {p['url']}\n"
                f"    ABSTRACT: {p['abstract'][:180]}..."
                for j, p in enumerate(batch)
            ])

            response = ctx.llm.chat(
                f"You are a manuscript reviewer. Context: {ctx.config.short_context}",
                f"For each paper below, evaluate its relevance to our manuscript.\n\n"
                f"{batch_fmt}\n\n"
                f"For EACH paper respond with EXACTLY this format:\n"
                f"PAPER [N]\n"
                f"SECTION: [manuscript section or Not relevant]\n"
                f"STRENGTH: [Strong / Moderate / Weak / Not relevant]\n"
                f"USE: [1 sentence on how to use it]\n"
                f"ACTION: [Accept / Reject / Review]\n"
                f"---",
                max_tokens=400,
            )
            entries.append(
                f"### Papers {i + 1}–{min(i + self.batch_size, len(papers_eval))}\n\n{response}"
            )
            print(f"    Evaluated papers {i + 1}–{min(i + self.batch_size, len(papers_eval))}")
            if i + self.batch_size < len(papers_eval):
                time.sleep(self.cooldown)

        full_report = "\n\n".join(entries)
        lower = full_report.lower()
        summary = (
            f"PRELIMINARY RELEVANCE SUMMARY\n"
            f"Total papers evaluated : {len(papers_eval)}\n"
            f"Strong matches         : {lower.count('strength: strong')}\n"
            f"Moderate matches       : {lower.count('strength: moderate')}\n"
            f"Weak / not relevant    : {lower.count('strength: weak')}\n"
            f"Suggested rejects      : {lower.count('action: reject')}\n\n"
            f"NOTE: This is a preliminary AI assessment — verify each paper before citing.\n"
            f"Use 'litreviewagents accept' to log accepted papers.\n"
            f"Use 'litreviewagents reject' to permanently exclude noise.\n\n"
            f"{'=' * 50}\n\n"
        )

        content = summary + full_report
        uid = ctx.store.store(self.name, "relevance_report", content)
        ctx.last_uid["_relevance"] = uid
        return uid


class SynthesisAgent(Agent):
    """
    Assembles a final synthesis from all LiteratureAgent outputs.

    Behavior is driven by config.synthesis:
      - sections — list of {title, source_agent, max_tokens} entries
      - research_questions_prompt — optional prompt to draft RQs
      - discussion_prompt — optional prompt to draft discussion paragraphs

    If no synthesis config is provided, this agent acts as a passthrough.
    """

    name = "SynthesisAgent"

    def run(self, ctx):
        print(f"\n[{self.name}] Synthesising findings...")

        results = {}

        def get_analysis(agent_name):
            arts = ctx.store.get_by_type(f"{agent_name}_analysis")
            return arts[0]["content"] if arts else "No findings."

        # Research questions
        if ctx.config.synthesis.research_questions_prompt:
            print("    Drafting research questions...")
            rqs = ctx.llm.chat(
                f"You are an academic writing expert. Context: {ctx.config.short_context}",
                ctx.config.synthesis.research_questions_prompt,
                max_tokens=ctx.config.synthesis.max_tokens,
            )
            results["1. Research Questions"] = rqs

        # Per-section synthesis from config
        for idx, sec in enumerate(ctx.config.synthesis.sections, start=2):
            title = sec.get("title", f"Section {idx}")
            source = sec.get("source_agent")
            if source:
                results[f"{idx}. {title}"] = get_analysis(source)

        # Discussion paragraphs
        if ctx.config.synthesis.discussion_prompt:
            print("    Drafting expanded discussion...")
            # Substitute {agent_name_analysis} placeholders in the prompt
            prompt = ctx.config.synthesis.discussion_prompt
            for art in ctx.store.artifacts.values():
                if art["type"].endswith("_analysis"):
                    agent_name = art["agent"]
                    placeholder = "{" + f"{agent_name}_analysis" + "}"
                    if placeholder in prompt:
                        # truncate to keep prompt size manageable
                        prompt = prompt.replace(placeholder, art["content"][:250])
            discussion = ctx.llm.chat(
                f"You are an expert academic writer. Context: {ctx.config.short_context}",
                prompt,
                max_tokens=ctx.config.synthesis.max_tokens,
            )
            results[f"{len(results) + 1}. Expanded Discussion Drafts"] = discussion

        uid = ctx.store.store(self.name, "synthesis", results)
        ctx.last_uid["_synthesis"] = uid
        ctx.synthesis_results = results
        return uid


class ValidatorAgent(Agent):
    """Quality-check notes appended to the final report."""

    name = "ValidatorAgent"

    def run(self, ctx):
        print(f"\n[{self.name}] Running validation checks...")

        all_papers = []
        for art in ctx.store.artifacts.values():
            if art["type"].endswith("_papers"):
                all_papers.extend(art["content"])

        total = len(all_papers)
        pre_2020 = sum(1 for p in all_papers if p.get("date", "2020") < "2020-01-01")
        preprints = sum(1 for p in all_papers if "arxiv.org" in p.get("url", ""))

        notes = [
            f"Total papers retrieved: {total}",
            f"Pre-2020 papers: {pre_2020} — verify relevance before citing",
            f"Preprints (arxiv): {preprints}/{total} — check peer-review status before citing",
            "All LLM-generated text must be reviewed and rewritten in your own words before submission",
            "All paper URLs must be manually verified against published versions",
            "Citation placeholders [X] must be replaced with real reference numbers",
        ]

        text = "\n".join(f"- {n}" for n in notes)
        uid = ctx.store.store(self.name, "validation", text)
        ctx.last_uid["_validation"] = uid
        ctx.validation_notes = text
        return uid
