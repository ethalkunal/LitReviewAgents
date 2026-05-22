"""Smoke tests for litreviewagents. Run with: pytest"""

import json
import tempfile
from pathlib import Path

import pytest

from litreviewagents import Config, PaperMemory
from litreviewagents.llm import strip_reasoning_preamble
from litreviewagents.sources import SOURCES, BaseSource, Paper, get_source, register_source


# ── Config loading ─────────────────────────────────────────────────────────────

def test_config_loads_iot_example():
    """The bundled IoT example should load and validate."""
    config_path = Path(__file__).parent.parent / "examples" / "iot_compliance.yaml"
    config = Config.from_yaml(config_path)
    config.validate()
    assert config.project_name == "IoT Compliance Logging Review"
    assert len(config.agents) == 5
    assert config.sources["arxiv"] is True


def test_config_loads_ml_example():
    """The ML reproducibility example should also load."""
    config_path = Path(__file__).parent.parent / "examples" / "ml_reproducibility.yaml"
    config = Config.from_yaml(config_path)
    config.validate()
    assert len(config.agents) == 3


def test_config_validates_required_fields():
    """Missing fields should raise ValueError."""
    bad_yaml = """
project_name: ""
agents: []
sources:
  arxiv: false
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(bad_yaml)
        tmppath = f.name
    config = Config.from_yaml(tmppath)
    with pytest.raises(ValueError):
        config.validate()


def test_config_requires_papers_placeholder():
    """An agent user_prompt without {papers} should fail validation."""
    bad_yaml = """
project_name: "Test"
agents:
  - name: BadAgent
    queries: ["x"]
    user_prompt: "no placeholder"
sources:
  arxiv: true
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(bad_yaml)
        tmppath = f.name
    config = Config.from_yaml(tmppath)
    with pytest.raises(ValueError, match=r"\{papers\}"):
        config.validate()


# ── Memory ─────────────────────────────────────────────────────────────────────

def test_paper_memory_lifecycle(tmp_path):
    mem = PaperMemory(tmp_path)
    assert mem.load_accepted_urls() == set()
    mem.add_accepted("https://a.com", "A", ["X"], "2024-01-01", "Sec 1")
    assert "https://a.com" in mem.load_accepted_urls()
    # duplicate accept is a no-op
    mem.add_accepted("https://a.com", "A", ["X"], "2024-01-01", "Sec 1")
    with open(mem.accepted_file) as f:
        data = json.load(f)
    assert len(data["papers"]) == 1


def test_paper_memory_seen_failsafe(tmp_path):
    mem = PaperMemory(tmp_path)
    assert mem.load_seen_urls() == set()
    mem.add_seen(["https://a.com", "https://b.com"])
    mem.add_seen(["https://b.com", "https://c.com"])
    assert mem.load_seen_urls() == {"https://a.com", "https://b.com", "https://c.com"}
    mem.clear_seen()
    assert mem.load_seen_urls() == set()


# ── Reasoning preamble stripping ──────────────────────────────────────────────

def test_strip_preamble_passthrough_for_clean_text():
    clean = "## Section\n\nThis is fine."
    assert strip_reasoning_preamble(clean) == clean


def test_strip_preamble_removes_okay_lets():
    raw = "Okay, let's tackle this. The user wants...\n\n## Section\n\nReal content here."
    cleaned = strip_reasoning_preamble(raw)
    assert "Okay, let's" not in cleaned
    assert "Real content here" in cleaned


def test_strip_preamble_handles_empty():
    assert strip_reasoning_preamble("") == ""
    assert strip_reasoning_preamble(None) is None


def test_strip_preamble_detects_infinite_loop():
    looped = ("Wait, the user wants the answer. " * 30)
    out = strip_reasoning_preamble(looped)
    assert "infinite reasoning loop" in out.lower() or "reasoning loop" in out.lower()


# ── Sources registry ──────────────────────────────────────────────────────────

def test_sources_registered():
    assert "arxiv" in SOURCES
    assert "crossref" in SOURCES
    assert "semantic_scholar" in SOURCES
    assert "pubmed" in SOURCES


def test_get_source_returns_instance():
    src = get_source("arxiv")
    assert hasattr(src, "search")


def test_get_source_unknown_raises():
    with pytest.raises(ValueError, match="Unknown source"):
        get_source("not_a_real_source")


def test_register_custom_source():
    class FakeSource(BaseSource):
        name = "fake"
        def search(self, query, max_results=5):
            return [Paper(
                title="Fake paper", abstract="abstract", authors=["X"],
                url="https://fake", date="2024-01-01", source="fake",
            )]
    register_source("fake", FakeSource)
    assert "fake" in SOURCES
    papers = get_source("fake").search("anything")
    assert papers[0].title == "Fake paper"


def test_paper_to_dict():
    p = Paper(title="T", abstract="A", authors=["X"], url="https://u",
              date="2024-01-01", source="arxiv")
    d = p.to_dict()
    assert d["title"] == "T"
    assert d["citations"] is None
    assert d["peer_reviewed"] is False
