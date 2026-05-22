"""
Persistent paper memory — accepted, rejected, and seen lists.

Files live under the project's `memory_dir` (configurable):
- accepted_papers.json — papers you've explicitly accepted for citation
- rejected_papers.json — papers permanently excluded from future runs
- seen_papers.json     — failsafe to skip re-evaluation of previously seen papers
"""

import json
from datetime import datetime
from pathlib import Path


class PaperMemory:
    """Manages accepted, rejected, and seen paper lists."""

    def __init__(self, memory_dir):
        self.dir = Path(memory_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.accepted_file = self.dir / "accepted_papers.json"
        self.rejected_file = self.dir / "rejected_papers.json"
        self.seen_file = self.dir / "seen_papers.json"

    # ── accepted ──────────────────────────────────────────────────────────────

    def load_accepted_urls(self):
        if not self.accepted_file.exists():
            return set()
        with open(self.accepted_file) as f:
            return {p["url"] for p in json.load(f).get("papers", [])}

    def add_accepted(self, url, title, authors, date, section, notes=""):
        data = {"papers": []}
        if self.accepted_file.exists():
            with open(self.accepted_file) as f:
                data = json.load(f)
        if any(p["url"] == url for p in data["papers"]):
            print(f"  Already accepted: {title[:60]}")
            return
        data["papers"].append({
            "url": url,
            "title": title,
            "authors": authors if isinstance(authors, list) else [a.strip() for a in authors.split(",")],
            "date": date,
            "section": section,
            "notes": notes,
            "accepted": datetime.now().strftime("%Y-%m-%d"),
        })
        with open(self.accepted_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Accepted: [{len(data['papers'])}] {title[:60]}")

    def list_accepted(self):
        if not self.accepted_file.exists():
            print("No accepted papers yet.")
            return
        with open(self.accepted_file) as f:
            data = json.load(f)
        papers = data.get("papers", [])
        print(f"\n{'=' * 62}")
        print(f"  Accepted Papers ({len(papers)} total)")
        print(f"  File: {self.accepted_file}")
        print(f"{'=' * 62}")
        for i, p in enumerate(papers):
            authors = p["authors"]
            authors_str = ", ".join(authors) if isinstance(authors, list) else authors
            print(f"\n[{i + 1}] {p['title']}")
            print(f"    Authors : {authors_str}")
            print(f"    Date    : {p['date']}")
            print(f"    Section : {p['section']}")
            print(f"    Notes   : {p['notes'] or 'none'}")
            print(f"    URL     : {p['url']}")
        print(f"\n{'=' * 62}\n")

    # ── rejected ──────────────────────────────────────────────────────────────

    def load_rejected_urls(self):
        if not self.rejected_file.exists():
            return set()
        with open(self.rejected_file) as f:
            return {p["url"] for p in json.load(f).get("papers", [])}

    def add_rejected(self, url, title, reason=""):
        data = {"papers": []}
        if self.rejected_file.exists():
            with open(self.rejected_file) as f:
                data = json.load(f)
        if any(p["url"] == url for p in data["papers"]):
            print(f"  Already rejected: {title[:60]}")
            return
        data["papers"].append({
            "url": url,
            "title": title,
            "reason": reason,
            "rejected": datetime.now().strftime("%Y-%m-%d"),
        })
        with open(self.rejected_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Rejected: [{len(data['papers'])}] {title[:60]}")

    def list_rejected(self):
        if not self.rejected_file.exists():
            print("No rejected papers yet.")
            return
        with open(self.rejected_file) as f:
            data = json.load(f)
        papers = data.get("papers", [])
        print(f"\n{'=' * 62}")
        print(f"  Rejected Papers ({len(papers)} total — skipped in all future runs)")
        print(f"  File: {self.rejected_file}")
        print(f"{'=' * 62}")
        for i, p in enumerate(papers):
            print(f"\n[{i + 1}] {p['title']}")
            print(f"    Reason   : {p['reason'] or 'none'}")
            print(f"    Rejected : {p['rejected']}")
            print(f"    URL      : {p['url']}")
        print(f"\n{'=' * 62}\n")

    # ── seen (failsafe) ───────────────────────────────────────────────────────

    def load_seen_urls(self):
        if not self.seen_file.exists():
            return set()
        with open(self.seen_file) as f:
            return set(json.load(f).get("urls", []))

    def add_seen(self, urls):
        existing = self.load_seen_urls()
        merged = existing | set(urls)
        with open(self.seen_file, "w") as f:
            json.dump({"urls": sorted(merged), "count": len(merged)}, f, indent=2)

    def clear_seen(self):
        if self.seen_file.exists():
            self.seen_file.unlink()
        print("  Seen-papers log cleared.")
