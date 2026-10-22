#!/usr/bin/env python3
"""
Autonomous AI Breakthrough Wire
Ingests daily ArXiv papers, semantically filters for SOTA benchmarks and 
novel architectures, and generates machine-readable intelligence.
Works 100% free with heuristic parsing, with optional LLM enrichment.
"""

import os
import re
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
    architecture_type: str = Field(description="e.g., Transformer, MoE, Mamba, Diffusion, RAG")
    claimed_benchmarks: str = Field(description="Detected benchmark metrics or scores")
    has_github_repo: bool
    github_url: str
    tldr_summary: str


def fetch_recent_arxiv_papers():
    """Fetches the 50 most recent papers in AI, ML, and NLP from ArXiv."""
    url = "http://export.arxiv.org/api/query?search_query=cat:cs.AI+OR+cat:cs.CL+OR+cat:cs.LG&sortBy=submittedDate&sortOrder=desc&max_results=50"
    try:
        feed = feedparser.parse(url)
        logging.info(f"Retrieved {len(feed.entries)} entries from ArXiv.")
        return feed.entries
    except Exception as e:
        logging.error(f"Failed to fetch ArXiv: {e}")
        return []


def extract_heuristics(title: str, abstract: str) -> dict:
    """Fallback parser that detects SOTA claims, architectures, and GitHub repos without an API key."""
    text = f"{title} {abstract}".lower()
    
    # 1. Detect SOTA or breakthrough claims
    sota_patterns = [
        r"state-of-the-art", r"\bsota\b", r"outperforms?", r"superior performance",
        r"surpass(es|ed)?", r"new benchmark", r"achieves? \d+[\.\d]*%", r"groundbreaking",
        r"novel architecture", r"novel framework", r"cutting-edge"
    ]
    is_breakthrough = any(re.search(p, text) for p in sota_patterns)

    # 2. Extract GitHub repositories
    gh_match = re.search(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", abstract)
    has_github = bool(gh_match)
    github_url = gh_match.group(0).rstrip(".") if gh_match else ""

    # 3. Detect architecture family
    arch_type = "Machine Learning"
    if "transformer" in text:
        arch_type = "Transformer"
    elif "mamba" in text or "state space" in text:
        arch_type = "State Space Model (SSM/Mamba)"
    elif "diffusion" in text:
        arch_type = "Diffusion Model"
    elif "mixture-of-experts" in text or "\bmoe\b" in text:
        arch_type = "Mixture of Experts (MoE)"
    elif "rag" in text or "retrieval-augmented" in text:
        arch_type = "Retrieval-Augmented Generation (RAG)"
    elif "reinforcement learning" in text or "\brl\b" in text:
        arch_type = "Reinforcement Learning"

    # 4. Extract benchmarks mentioned
    benchmarks = []
    for b in ["MMLU", "HumanEval", "GSM8K", "MATH", "SWE-bench", "ImageNet", "ARC", "HellaSwag"]:
        if b.lower() in text:
            benchmarks.append(b)
    claimed_benchmarks = ", ".join(benchmarks) if benchmarks else "General Empirical Evaluation"

    # Short summary from first two sentences of abstract
    sentences = re.split(r'(?<=[.!?]) +', abstract.strip())
    tldr = " ".join(sentences[:2]) if sentences else title

    return {
        "is_breakthrough": is_breakthrough,
        "architecture_type": arch_type,
        "claimed_benchmarks": claimed_benchmarks,
        "has_github_repo": has_github,
        "github_url": github_url,
        "tldr_summary": tldr[:280]
    }


def extract_paper_intelligence(title: str, abstract: str) -> dict:
    """Uses LLM if key is present; otherwise falls back cleanly to heuristics."""
    api_key = os.getenv("LLM_API_KEY")
    api_base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if not api_key:
        return extract_heuristics(title, abstract)

    system_prompt = (
        "You are an expert AI research evaluator. Analyze the title and abstract. "
        "Extract structured data. 'is_breakthrough' must be true if the paper claims "
        "SOTA results, outperforming baselines, or introducing a novel architecture. "
        "Output JSON only: {\"is_breakthrough\": bool, \"architecture_type\": str, "
        "\"claimed_benchmarks\": str, \"has_github_repo\": bool, \"github_url\": str, \"tldr_summary\": str}"
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
        logging.warning(f"LLM extraction failed, using heuristics: {e}")
        return extract_heuristics(title, abstract)


def run():
    entries = fetch_recent_arxiv_papers()
    if not entries:
        logging.warning("No entries returned from ArXiv feed.")
        return

    records = []
    md_lines = [
        "# Daily AI Breakthrough & Architecture Radar",
        "> Autonomous intelligence on novel architectures and SOTA benchmark claims.\n",
        "## High-Impact Papers (Last 24 Hours)"
    ]

    for entry in entries:
        title = entry.title.replace('\n', ' ').strip()
        abstract = entry.summary.replace('\n', ' ').strip()
        
        intel = extract_paper_intelligence(title, abstract)
        
        if not intel.get("is_breakthrough"):
            continue

        raw_id = entry.id.split('/abs/')[-1] if '/abs/' in entry.id else entry.id
        pdf_url = entry.id.replace('abs', 'pdf') if 'abs' in entry.id else entry.id
        author_list = [author.name for author in getattr(entry, 'authors', [])]

        paper = AIPaper(
            arxiv_id=raw_id,
            title=title,
            authors=author_list,
            published_date=getattr(entry, 'published', datetime.now(timezone.utc).isoformat()),
            pdf_url=pdf_url,
            is_breakthrough=True,
            architecture_type=intel.get("architecture_type", "Machine Learning"),
            claimed_benchmarks=intel.get("claimed_benchmarks", "Empirical Evaluation"),
            has_github_repo=intel.get("has_github_repo", False),
            github_url=intel.get("github_url", ""),
            tldr_summary=intel.get("tldr_summary", title)
        )
        records.append(paper)

        repo_badge = f" [[GitHub]]({paper.github_url})" if paper.has_github_repo else ""
        md_lines.extend([
            f"### {paper.title}",
            f"- **ArXiv ID:** [{paper.arxiv_id}]({entry.id}){repo_badge}",
            f"- **Architecture:** `{paper.architecture_type}` | **Benchmarks:** `{paper.claimed_benchmarks}`",
            f"- **TL;DR:** {paper.tldr_summary}\n"
        ])

    if not records:
        logging.info("No breakthroughs matched criteria today.")
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
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "today.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        
    with open(os.path.join(DATA_DIR, "llms.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Save historical archive with safe directory creation
    month_dir = os.path.join(ARCHIVE_DIR, now.strftime("%Y"), now.strftime("%m"))
    os.makedirs(month_dir, exist_ok=True)
    with open(os.path.join(month_dir, f"{date_str}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logging.info(f"Published {len(records)} breakthroughs to data/today.json and archive.")


if __name__ == "__main__":
    run()
