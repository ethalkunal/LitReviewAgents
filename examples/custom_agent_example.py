"""
Example: a custom Agent subclass.

If a YAML agent isn't expressive enough, write a Python class and plug it
into the pipeline. This example adds a "CitationCountFilterAgent" that
post-processes the RelatedWorkAgent papers and re-analyzes only the
top-cited ones.
"""

from litreviewagents import Config, LiteratureAgent, Pipeline


class HighlyCitedFilterAgent(LiteratureAgent):
    """Re-runs analysis using only the top-cited papers from another agent."""

    def __init__(self, agent_config, source_agent, min_citations=20):
        super().__init__(agent_config)
        self.source_agent = source_agent
        self.min_citations = min_citations

    def fetch_papers(self, ctx):
        # Pull papers from another agent's artifact instead of re-querying
        arts = ctx.store.get_by_type(f"{self.source_agent}_papers")
        if not arts:
            print(f"  No source artifacts from {self.source_agent}")
            return []
        papers = arts[0]["content"]
        filtered = [
            p for p in papers
            if (p["citations"] if isinstance(p, dict) else p.citations) is not None
            and (p["citations"] if isinstance(p, dict) else p.citations) >= self.min_citations
        ]
        print(f"  {len(filtered)}/{len(papers)} papers above {self.min_citations} citations")
        return filtered


if __name__ == "__main__":
    config = Config.from_yaml("ml_reproducibility.yaml")
    pipeline = Pipeline(config)

    # Insert a custom agent after PriorReproWorkAgent
    from litreviewagents.config import AgentConfig
    custom = HighlyCitedFilterAgent(
        AgentConfig(
            name="HighlyCitedPriorWork",
            description="Top-cited prior reproducibility papers only",
            queries=[],  # unused — pulls from source_agent
            system_prompt="You are a careful ML researcher synthesising the strongest prior work.",
            user_prompt="The following are the most-cited prior reproducibility papers.\n\n{papers}\n\nDraft a 200-word state-of-the-art summary citing these specifically.",
            max_tokens=900,
        ),
        source_agent="PriorReproWorkAgent",
        min_citations=20,
    )
    pipeline.add_agent(custom, position=1)
    pipeline.run()
