#!/usr/bin/env python3
"""
Autonomous AI Breakthrough Wire
Ingests daily ArXiv papers, semantically filters for SOTA benchmarks and 
novel architectures, and generates machine-readable intelligence.
"""

import os
import json
import logging
from datetime import datetime, timezone
import requests
import feedparser
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = "data"
ARCHIVE_DIR = os.path.join(DATA_DIR, "archive")
os.makedirs(ARCHIVE_DIR, exist_ok=True)

class AIPaper(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    published_date: str
    pdf_url: str
    is_breakthrough: bool = Field(description="True if claims SOTA or novel architecture")
    architecture_type: str = Field(description="e.g., Transformer, MoE, Mamba, RAG")
    claimed_benchmarks: str = Field(description="e.g., MMLU: 88%, HumanEval: 90%")
    has_github_repo: bool
    github_url: str
    tldr_summary: str


def fetch_recent_arxiv_papers():
    """Fetches the 50 most recent papers in AI, ML, and NLP from ArXiv."""
    # query categories: cs.AI (Artificial Intel), cs.CL (Computation/Lang), cs.LG (Machine Learning)
    url = "http://export.arxiv.org/api/query?search_query=cat:cs.AI+OR+cat:cs.CL+OR+cat:cs.LG&sortBy=submittedDate&sortOrder=desc&max_results=50"
    try:
        feed = feedparser.parse(url)
        return feed.entries
    except Exception as e:
        logging.error(f"Failed to fetch ArXiv: {e}")
        return []


def extract_paper_intelligence(title: str, abstract: str) -> dict:
    """Uses an LLM to determine if the paper is a breakthrough and extract metrics."""
    api_key = os.getenv("LLM_API_KEY")
    api_base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if not api_key:
        return {}

    system_prompt = (
        "You are an expert AI research evaluator. Read the title and abstract of the academic paper. "
        "Extract structured data. 'is_breakthrough' must only be true if the paper explicitly claims "
        "State-of-the-Art (SOTA) results, a novel architectural paradigm, or a massive efficiency gain. "
        "If it is a minor tweak or survey paper, set it to false. Extract any mentioned GitHub URLs. "
        "Output JSON only: {\"is_breakthrough\": bool, \"architecture_type\": str, \"claimed_benchmarks\": str, "
        "\"has_github_repo\": bool, \"github_url\": str, \"tldr_summary\": str}"
    )
    user_prompt = f"Title: {title}\n\nAbstract: {abstract}"

    try:
        res = requests.post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            },
            timeout=15
        )
        return json.loads(res.json()["choices"][0]["message"]["content"])
    except Exception as e:
        logging.error(f"LLM extraction failed for {title}: {e}")
        return {}


def run():
    entries = fetch_recent_arxiv_papers()
    if not entries:
        return

    records = []
    md_lines = [
        "# Daily AI Breakthrough & Architecture Radar",
        "> Autonomous intelligence on novel architectures and SOTA benchmark claims.\n",
        "## High-Impact Papers (Last 24 Hours)"
    ]

    for entry in entries:
        title = entry.title.replace('\n', ' ')
        abstract = entry.summary.replace('\n', ' ')
        
        # Analyze abstract via LLM
        intel = extract_paper_intelligence(title, abstract)
        
        # Skip papers that aren't novel breakthroughs to save agent tokens
        if not intel.get("is_breakthrough"):
            continue

        paper = AIPaper(
            arxiv_id=entry.id.split('/abs/')[-1],
            title=title,
            authors=[author.name for author in entry.authors],
            published_date=entry.published,
            pdf_url=entry.id.replace('abs', 'pdf'),
            is_breakthrough=True,
            architecture_type=intel.get("architecture_type", "Unknown"),
            claimed_benchmarks=intel.get("claimed_benchmarks", "None specified"),
            has_github_repo=intel.get("has_github_repo", False),
            github_url=intel.get("github_url", ""),
            tldr_summary=intel.get("tldr_summary", "No summary available.")
        )
        records.append(paper)

        repo_badge = f" [GitHub]({paper.github_url})" if paper.has_github_repo else ""
        md_lines.extend([
            f"### {paper.title}",
            f"- **ArXiv ID:** [{paper.arxiv_id}]({entry.id}){repo_badge}",
            f"- **Architecture:** `{paper.architecture_type}` | **Benchmarks:** `{paper.claimed_benchmarks}`",
            f"- **TL;DR:** {paper.tldr_summary}\n"
        ])

    if not records:
        logging.info("No breakthroughs detected today.")
        return

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    
    payload = {
        "metadata": {
            "generated_at": now.isoformat(),
            "papers_scanned": len(entries),
            "breakthroughs_identified": len(records),
            "schema_version": "1.0.0"
        },
        "papers": [r.model_dump() for r in records]
    }

    # Save active endpoints
    with open(os.path.join(DATA_DIR, "today.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        
    with open(os.path.join(DATA_DIR, "llms.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Save historical archive
    month_dir = os.path.join(ARCHIVE_DIR, now.strftime("%Y"), now.strftime("%m"))
    os.makedirs(month_dir, exist_ok=True)
    with open(os.path.join(month_dir, f"{date_str}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logging.info(f"Published {len(records)} breakthroughs to wire.")


if __name__ == "__main__":
    run()
