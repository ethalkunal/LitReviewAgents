# Changelog

## [0.1.0] — 2026-05-22

Initial release.

### Added
- YAML-driven configuration with full validation
- `LiteratureAgent`, `RelevanceAgent`, `SynthesisAgent`, `ValidatorAgent`
- Source backends: arXiv, Crossref, Semantic Scholar, PubMed
- OpenAI-compatible LLM client (Ollama / vLLM / LM Studio / OpenAI / Groq / Together)
- Reasoning-preamble stripping for local chain-of-thought models
- Persistent paper memory (accepted / rejected / seen)
- CLI: `init`, `run`, `accept`, `reject`, `list`, `list-rejected`, `clear-seen`, `show-config`
- Two annotated example configs (IoT compliance, ML reproducibility)
- Custom-agent Python plugin example
- CI workflow for Python 3.9–3.12
