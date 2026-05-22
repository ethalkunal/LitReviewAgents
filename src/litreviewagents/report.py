"""
Artifact storage and markdown report generation.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path


class ArtifactStore:
    """Persists agent outputs as both in-memory dict and JSON files on disk."""

    def __init__(self, session_dir):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts = {}

    def store(self, agent_name, artifact_type, content, parent_ids=None):
        uid = hashlib.md5(f"{agent_name}{artifact_type}{time.time()}".encode()).hexdigest()[:8]
        artifact = {
            "id": uid,
            "agent": agent_name,
            "type": artifact_type,
            "content": content
            if not _has_papers(content)
            else [p if isinstance(p, dict) else p.to_dict() for p in content],
            "parents": parent_ids or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.artifacts[uid] = artifact
        with open(self.session_dir / f"{agent_name}_{artifact_type}_{uid}.json", "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"  [artifact] {agent_name} -> {artifact_type} ({uid})")
        return uid

    def get_by_type(self, t):
        return [a for a in self.artifacts.values() if a["type"] == t]


def _has_papers(content):
    """Detect a list of Paper objects so we can serialize them."""
    if not isinstance(content, list) or not content:
        return False
    first = content[0]
    return hasattr(first, "to_dict")


class Report:
    """Writes a markdown report from agent outputs."""

    def __init__(self, session_dir, project_name, model):
        self.session_dir = Path(session_dir)
        self.project_name = project_name
        self.model = model

    def write(self, results, filename="LITERATURE_REPORT.md"):
        path = self.session_dir / filename
        with open(path, "w") as f:
            f.write(f"# {self.project_name} — Literature Review Report\n\n")
            f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Model: {self.model}*\n\n")
            f.write("> **VALIDATION REQUIRED**: All findings must be verified before manuscript use.\n")
            f.write("> Preprint links (arxiv) should be checked for peer-review status before citing.\n\n")
            f.write("---\n\n")
            for section, content in results.items():
                f.write(f"## {section}\n\n{content}\n\n---\n\n")
        print(f"\n  Report saved: {path}")
        return path


# ── Formatters ─────────────────────────────────────────────────────────────────


def fmt_papers_compact(papers, max_n=3, abstract_len=150):
    """Compact paper formatter for LLM prompts — caps to limit prompt tokens."""
    if not papers:
        return "No papers found."
    out = []
    for i, p in enumerate(papers[:max_n]):
        title = p["title"] if isinstance(p, dict) else p.title
        date = p["date"] if isinstance(p, dict) else p.date
        authors = p["authors"] if isinstance(p, dict) else p.authors
        abstract = p["abstract"] if isinstance(p, dict) else p.abstract
        out.append(
            f"[{i + 1}] {title} ({date})\n    Authors: {', '.join(authors)}\n    {abstract[:abstract_len]}..."
        )
    return "\n\n".join(out)


def fmt_papers_full(papers):
    """Full paper formatter for the markdown report (no token limit)."""
    if not papers:
        return "No papers found."
    out = []
    for i, p in enumerate(papers):
        d = p if isinstance(p, dict) else p.to_dict()
        out.append(
            f"[{i + 1}] {d['title']} ({d['date']})\n"
            f"Authors : {', '.join(d['authors'])}\n"
            f"URL     : {d['url']}\n"
            f"Source  : {d.get('source', 'unknown')} | "
            f"Citations: {d.get('citations', '?')} | "
            f"Peer-reviewed: {'Yes' if d.get('peer_reviewed') else 'No/Unknown'}\n"
            f"Abstract: {d['abstract'][:300]}..."
        )
    return "\n\n".join(out)
