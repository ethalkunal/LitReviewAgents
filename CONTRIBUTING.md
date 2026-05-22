# Contributing to LitReviewAgents

Thanks for considering a contribution.

## Development setup

```bash
git clone https://github.com/ethalkunal/LitReviewAgents
cd LitReviewAgents
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest -v
```

## Linting

```bash
ruff check src tests
ruff format src tests
```

## Adding a new paper source

1. Subclass `litreviewagents.sources.BaseSource`.
2. Implement `search(query, max_results) -> list[Paper]`.
3. Register it in the `SOURCES` dict at the bottom of `src/litreviewagents/sources/__init__.py`.
4. Add a smoke test under `tests/`.
5. Document it in the README's Sources table.

## Adding a new agent type

Most agents should be expressible in YAML — only subclass `Agent` or `LiteratureAgent` when you need to override `fetch_papers`, `build_user_prompt`, or `run` itself. Put the new class under `src/litreviewagents/agents.py` and re-export it from `__init__.py`.

## Pull requests

- Keep changes focused. One feature per PR.
- Add tests for new behavior.
- Update README if behavior or config schema changes.
- Run `ruff check` and `pytest` locally before opening the PR.
