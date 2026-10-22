#!/usr/bin/env python3
"""
Autonomous AI Breakthrough Wire
Ingests daily ArXiv papers via HTTPS, applies heuristic & LLM extraction,
and guarantees output generation for automated GitHub commits.
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
    """Fetches recent AI, ML, and NLP papers using HTTPS and an explicit User-Agent."""
    url = "https://export.arxiv.org/api/query?search_query=cat:cs.AI+OR+cat:cs.CL+OR+cat:cs.LG&sortBy=submittedDate&sortOrder=desc&max_results=30"
    headers = {
        "User-Agent": "ArXiv-Breakthrough-Wire/1.0 (https://github.com/mysukriramli/arxiv-breakthrough-wire; contact: research@example.com)"
    }
    try:
        res = requests.get(url, headers=headers, timeout=25)
        res.raise_for_status()
        feed = feedparser.parse(res.content)
        logging.info(f"Retrieved {len(feed.entries)} entries from ArXiv.")
        return feed.entries
    except Exception as e:
        logging.error(f"Failed to fetch ArXiv: {e}")
        return []


def extract_heuristics(title: str, abstract: str) -> dict:
    """Heuristic extractor for SOTA claims, architectures, and code links."""
    text = f"{title} {abstract}".lower()

    sota_patterns = [
        r"state-of-the-art", r"\bsota\b", r"outperforms?", r"superior performance",
        r"surpass(es|ed)?", r"new benchmark", r"achieves? \d+[\.\d]*%", r"groundbreaking",
        r"novel architecture", r"novel framework", r"cutting-edge", r"state of the art"
    ]
    is_breakthrough = any(re.search(p, text) for p in sota_patterns)

    gh_match = re.search(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", abstract)
    has_github = bool(gh_match)
    github_url = gh_match.group(0).rstrip(".") if gh_match else ""

    arch_type = "Machine Learning"
    if "transformer" in text:
        arch_type = "Transformer"
    elif "mamba" in text or "state space" in text:
        arch_type = "State Space Model (SSM/Mamba)"
    elif "diffusion" in text:
        arch_type = "Diffusion Model"
    elif "mixture-of-experts" in text or "moe" in text:
        arch_type = "Mixture of Experts (MoE)"
    elif "rag" in text or "retrieval-augmented" in text:
        arch_type = "Retrieval-Augmented Generation (RAG)"
    elif "reinforcement learning" in text or "rlhf" in text:
        arch_type = "Reinforcement Learning"

    benchmarks = []
    for b in ["MMLU", "HumanEval", "GSM8K", "MATH", "SWE-bench", "ImageNet", "ARC", "HellaSwag"]:
        if b.lower() in text:
            benchmarks.append(b)
    claimed_benchmarks = ", ".join(benchmarks) if benchmarks else "Empirical Evaluation"

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
    """Uses OpenAI if LLM_API_KEY is available; falls back to heuristics on error or quota issues."""
    api_key = os.getenv("LLM_API_KEY")
    api_base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if not api_key:
        return extract_heuristics(title, abstract)

    system_prompt = (
        "You are an expert AI research evaluator. Analyze the title and abstract. "
        "Extract structured data. Set 'is_breakthrough' to true if the paper claims "
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
        if res.status_code == 200:
            return json.loads(res.json()["choices"][0]["message"]["content"])
        else:
            logging.warning(f"LLM API returned {res.status_code}, falling back to heuristics.")
            return extract_heuristics(title, abstract)
    except Exception as e:
        logging.warning(f"LLM extraction error ({e}), falling back to heuristics.")
        return extract_heuristics(title, abstract)


def run():
    entries = fetch_recent_arxiv_papers()
    if not entries:
        logging.warning("No entries returned from ArXiv feed. Retrying without query filters...")
        # Fallback to direct CS feed if query syntax returned nothing
        try:
            feed = feedparser.parse("https://export.arxiv.org/rss/cs.AI")
            entries = feed.entries
        except Exception:
            entries = []

    records = []
    md_lines = [
        "# Daily AI Breakthrough & Architecture Radar",
        "> Autonomous intelligence on novel architectures and SOTA benchmark claims.\n",
        "## High-Impact Papers"
    ]

    for entry in entries:
        title = getattr(entry, "title", "").replace('\n', ' ').strip()
        abstract = getattr(entry, "summary", "").replace('\n', ' ').strip()
        if not title:
            continue

        intel = extract_paper_intelligence(title, abstract)

        # Include if flagged as breakthrough, or if it has an open-source GitHub repository
        if not (intel.get("is_breakthrough") or intel.get("has_github_repo")):
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
            is_breakthrough=bool(intel.get("is_breakthrough", True)),
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

    # Failsafe: If strict criteria yielded 0, take the top 5 recent papers as notable submissions
    if not records and entries:
        logging.info("Zero papers matched strict SOTA filter. Using top recent submissions as baseline.")
        for entry in entries[:5]:
            title = getattr(entry, "title", "").replace('\n', ' ').strip()
            raw_id = entry.id.split('/abs/')[-1] if '/abs/' in entry.id else entry.id
            records.append(AIPaper(
                arxiv_id=raw_id,
                title=title,
                authors=[author.name for author in getattr(entry, 'authors', [])],
                published_date=getattr(entry, 'published', datetime.now(timezone.utc).isoformat()),
                pdf_url=entry.id.replace('abs', 'pdf'),
                is_breakthrough=False,
                architecture_type="Machine Learning",
                claimed_benchmarks="Notable Recent Submission",
                has_github_repo=False,
                github_url="",
                tldr_summary=title
            ))
            md_lines.append(f"### {title}\n- **ArXiv ID:** [{raw_id}]({entry.id})\n")

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

    # Always write outputs to guarantee files exist on disk
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "today.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(os.path.join(DATA_DIR, "llms.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    month_dir = os.path.join(ARCHIVE_DIR, now.strftime("%Y"), now.strftime("%m"))
    os.makedirs(month_dir, exist_ok=True)
    with open(os.path.join(month_dir, f"{date_str}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logging.info(f"Successfully generated endpoints with {len(records)} papers.")


if __name__ == "__main__":
    run()
