# LitReviewAgents

> Multi-agent literature search and synthesis pipeline for academic manuscripts.

LitReviewAgents helps you cover the literature gaps reviewers care about — by deploying focused agents that each search a curated set of paper sources, score relevance, and draft citation-ready text for specific manuscript sections.

It's YAML-configurable, model-agnostic (any OpenAI-compatible endpoint), and runs locally with Ollama by default — your manuscript context never leaves your machine.

---

## Features

- **YAML-first configuration** — declare your paper context, agents, queries, and prompts in one file.
- **Plug-in agents** — start with YAML, subclass in Python when you need more.
- **Multiple sources** — arXiv, Crossref, Semantic Scholar, PubMed. Toggle per project.
- **Any OpenAI-compatible LLM** — Ollama, vLLM, LM Studio, OpenAI, Groq, Together, etc.
- **Paper memory** — accept / reject / seen lists persist across runs.
- **Relevance scoring** — per-paper Strong / Moderate / Weak rating before you cite.
- **Reasoning-model friendly** — strips chain-of-thought preambles from local models like qwen3 and deepseek-r1.
- **Provenance** — every artifact saved as JSON with lineage to its parent agents.

---

## Install

```bash
pip install litreviewagents
```

Or from source:

```bash
git clone https://github.com/ethalkunal/LitReviewAgents
cd LitReviewAgents
pip install -e .
```

---

## Quickstart

### 1. Create a starter config

```bash
litreviewagents init my_paper.yaml
```

This writes a fully commented `my_paper.yaml`. Edit it with:

- Your paper title, claim, and known reviewer gaps (`paper_context`)
- Which LLM endpoint to use
- Which sources to search
- The agents you want — each with its queries and prompt templates

### 2. Point at an LLM

The easiest setup is **Ollama** running locally:

```bash
# install: https://ollama.com
ollama pull qwen2.5:7b
ollama serve
```

In `my_paper.yaml`:

```yaml
llm:
  base_url: "http://localhost:11434/v1"
  model: "qwen2.5:7b"
  api_key: "ollama"
```

Or any other OpenAI-compatible endpoint — see [LLM backends](#llm-backends) below.

### 3. Run

```bash
litreviewagents run --config my_paper.yaml
```

You'll get:

- A timestamped session under `output/`
- A consolidated `LITERATURE_REPORT.md` per session
- Individual agent artifacts as JSON (papers + analysis)
- A per-paper relevance report you can act on with `accept` / `reject`

### 4. Curate the results

```bash
# Mark a paper for citation
litreviewagents accept --config my_paper.yaml \
  --url "https://doi.org/10.1234/xyz" \
  --title "FIWARE for IoT" \
  --authors "Alvaro Alonso, Joaquin Salvachua" \
  --date "2023-04-15" \
  --section "Discussion 4.3" \
  --notes "Closest middleware competitor"

# Permanently exclude noise from future runs
litreviewagents reject --config my_paper.yaml \
  --url "https://arxiv.org/abs/9999.99999" \
  --title "Off-topic paper" \
  --reason "Wrong domain"

litreviewagents list --config my_paper.yaml
```

---

## How it works

```
┌─────────────────────────────────────────────────────────────┐
│  Config (YAML)                                              │
│    project_name, paper_context, short_context, agents, ...  │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  Pipeline                    │
        │    ├─ Agent 1 ─ queries ─►   │  arxiv, crossref,
        │    ├─ Agent 2 ─ queries ─►   │  semantic_scholar,
        │    ├─ ...                    │  pubmed
        │    ├─ RelevanceAgent         │
        │    ├─ SynthesisAgent         │
        │    └─ ValidatorAgent         │
        └──────────────────────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  LITERATURE_REPORT.md        │
        │   + JSON artifacts (memory)  │
        │   + accepted/rejected lists  │
        └──────────────────────────────┘
```

Each agent in the config:

1. Runs its queries against every enabled source.
2. Deduplicates and skips papers already accepted / rejected / seen.
3. Sorts by citation count.
4. Feeds the top results into a templated LLM prompt to get section-ready analysis.

Then a global relevance agent rates every unique paper, a synthesis agent assembles the final manuscript drafts, and a validator agent appends quality-check notes.

---

## Config reference

Minimal agent block:

```yaml
agents:
  - name: RelatedWorkAgent
    queries:
      - "FIWARE IoT platform security middleware"
      - "Eclipse Ditto IoT device management"
    system_prompt: "You are a rigorous academic reviewer. Context: {context}"
    user_prompt: |
      These papers describe IoT middleware comparable to ours.
      {papers}
      Draft a citation-ready paragraph.
    max_tokens: 1000
```

Required:

- `name` — unique identifier (no spaces).
- `queries` — list of search strings.
- `user_prompt` — must contain `{papers}`.

Optional:

- `system_prompt` — may contain `{context}` (substituted from `short_context`).
- `max_tokens` — default 800.
- `description` — shown in the log.

A full annotated config: [`examples/iot_compliance.yaml`](examples/iot_compliance.yaml).
A non-IoT domain example: [`examples/ml_reproducibility.yaml`](examples/ml_reproducibility.yaml).

---

## LLM backends

LitReviewAgents talks to any endpoint speaking the OpenAI `/v1/chat/completions` protocol. Set `base_url`, `model`, and `api_key` in YAML, or override at runtime with environment variables: `LITREVIEW_BASE_URL`, `LITREVIEW_MODEL`, `LITREVIEW_API_KEY`.

| Backend       | base_url                                 | api_key                |
|---------------|------------------------------------------|------------------------|
| Ollama        | `http://localhost:11434/v1`              | `ollama` (any string)  |
| vLLM          | `http://localhost:8000/v1`               | server-configured      |
| LM Studio     | `http://localhost:1234/v1`               | any string             |
| OpenAI        | `https://api.openai.com/v1`              | `$OPENAI_API_KEY`      |
| Groq          | `https://api.groq.com/openai/v1`         | `$GROQ_API_KEY`        |
| Together      | `https://api.together.xyz/v1`            | `$TOGETHER_API_KEY`    |
| OpenRouter    | `https://openrouter.ai/api/v1`           | `$OPENROUTER_API_KEY`  |

For local reasoning models like `qwen3:14b`, set:

```yaml
llm:
  extra_options:
    think: false
  strip_preamble: true
```

---

## Sources

| Source             | Free | API key needed?           | Strengths                                |
|--------------------|------|---------------------------|------------------------------------------|
| `arxiv`            | yes  | no                        | Preprints, very fresh                    |
| `crossref`         | yes  | no                        | Peer-reviewed, DOIs                      |
| `semantic_scholar` | yes  | optional (recommended)    | 200M+ papers, citation counts            |
| `pubmed`           | yes  | no                        | Biomedical, healthcare IoT, IoMT         |

To enable Semantic Scholar at higher rate limits:

```bash
export SEMANTIC_SCHOLAR_API_KEY="your_key"  # https://www.semanticscholar.org/product/api
```

To add a custom source, subclass `litreviewagents.sources.BaseSource` and call `register_source(name, cls)`.

---

## Custom agents (Python)

When YAML isn't enough — for example, when you want to post-process another agent's papers:

```python
from litreviewagents import LiteratureAgent, Config, Pipeline

class HighlyCitedFilterAgent(LiteratureAgent):
    def fetch_papers(self, ctx):
        prior = ctx.store.get_by_type("RelatedWorkAgent_papers")[0]["content"]
        return [p for p in prior if (p.get("citations") or 0) >= 50]

config = Config.from_yaml("my_paper.yaml")
pipeline = Pipeline(config)
pipeline.add_agent(MyCustom(...), position=2)
pipeline.run()
```

See [`examples/custom_agent_example.py`](examples/custom_agent_example.py).

---

## What it does NOT do

- It does **not** invent citations. If the search returns nothing relevant, the agent says so.
- It does **not** auto-submit drafts. All output is labelled `VALIDATION REQUIRED`.
- It does **not** judge fair use, peer-review status, or the truth of a paper's claims — you do.
- It does **not** replace reading the papers it surfaces.

Every paper URL must be opened and verified before citation. Every LLM-drafted paragraph must be rewritten in your own words before submission.

---

## Roadmap

- [ ] OpenAlex source backend
- [ ] PDF download & full-text extraction
- [ ] BibTeX export of accepted papers
- [ ] Anthropic native API (without OpenAI-compat shim)
- [ ] Web UI for accept/reject curation

PRs welcome.

---

## License

MIT. See [LICENSE](LICENSE).

---

## Acknowledgements

LitReviewAgents grew out of a private tool ("ManuscriptAgents") built during manuscript revisions, generalized so other researchers can adapt it to their own papers without rewriting code.
