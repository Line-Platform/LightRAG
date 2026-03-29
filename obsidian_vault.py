#!/usr/bin/env python3
"""
LightRAG ↔ Obsidian Vault integration.

Indexes all markdown notes from Clay's Vault and provides
a query interface over the knowledge graph.

Usage:
    uv run python obsidian_lightrag.py index      # Index vault (first time / update)
    uv run python obsidian_lightrag.py query "your question"
    uv run python obsidian_lightrag.py query "your question" --mode hybrid
    uv run python obsidian_lightrag.py stats

Query modes: naive | local | global | hybrid | mix (default: mix)

Required env vars (add to ~/.zshrc or .env):
    ANTHROPIC_API_KEY   — https://console.anthropic.com/
    VOYAGE_API_KEY      — https://dash.voyageai.com/ (free tier available)
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

VAULT = Path("/Users/jamesbutler/Library/Mobile Documents/iCloud~md~obsidian/Documents/Clay's Vault")
RAG_STORAGE = Path.home() / ".lightrag" / "obsidian-vault"

# Folders to skip inside the vault
SKIP_DIRS = {".obsidian", ".trash", ".claude", "Attachments", ".stversions"}
# Only ingest markdown
INCLUDE_EXTS = {".md"}

ANTHROPIC_MODEL = "claude-sonnet-4-6"
VOYAGE_EMBED_MODEL = "voyage-3"
VOYAGE_EMBED_DIM = 1024

# ── Bootstrap ──────────────────────────────────────────────────────────────────

def check_env():
    missing = []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if not os.environ.get("VOYAGE_API_KEY"):
        missing.append("VOYAGE_API_KEY")
    if missing:
        print(f"❌ Missing env vars: {', '.join(missing)}")
        print("   Add them to ~/.zshrc:")
        for k in missing:
            print(f"   export {k}='your-key-here'")
        sys.exit(1)

RAG_STORAGE.mkdir(parents=True, exist_ok=True)

# ── LightRAG setup ─────────────────────────────────────────────────────────────

def build_rag():
    import numpy as np
    from lightrag import LightRAG
    from lightrag.llm.anthropic import anthropic_complete_if_cache, anthropic_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return await anthropic_complete_if_cache(
            model=ANTHROPIC_MODEL,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=os.environ["ANTHROPIC_API_KEY"],
            **kwargs,
        )

    @wrap_embedding_func_with_attrs(
        embedding_dim=VOYAGE_EMBED_DIM,
        max_token_size=32000,
        model_name=VOYAGE_EMBED_MODEL,
    )
    async def embed_func(texts: list[str]) -> np.ndarray:
        return await anthropic_embed(
            texts,
            model=VOYAGE_EMBED_MODEL,
            api_key=os.environ["VOYAGE_API_KEY"],
        )

    return LightRAG(
        working_dir=str(RAG_STORAGE),
        llm_model_func=llm_func,
        embedding_func=embed_func,
    )

# ── Vault reader ───────────────────────────────────────────────────────────────

def collect_notes() -> list[tuple[str, str]]:
    """Return list of (relative_path, content) for all vault notes."""
    notes = []
    for md_file in VAULT.rglob("*.md"):
        # Skip hidden/system dirs
        parts = set(md_file.relative_to(VAULT).parts[:-1])
        if parts & SKIP_DIRS:
            continue
        try:
            content = md_file.read_text(encoding="utf-8").strip()
            if len(content) < 50:  # skip near-empty files
                continue
            rel = str(md_file.relative_to(VAULT))
            # Prepend filename as context header
            notes.append((rel, f"# Note: {rel}\n\n{content}"))
        except Exception:
            pass
    return notes

# ── Commands ───────────────────────────────────────────────────────────────────

async def cmd_index(rag, force: bool = False):
    from lightrag import LightRAG

    notes = collect_notes()
    print(f"📚 Found {len(notes)} notes in vault")

    # Track already-indexed files
    indexed_file = RAG_STORAGE / ".indexed_files"
    already = set()
    if indexed_file.exists() and not force:
        already = set(indexed_file.read_text().splitlines())

    to_index = [(p, c) for p, c in notes if p not in already]
    if not to_index:
        print("✅ All notes already indexed. Use --force to re-index.")
        return

    print(f"⚙️  Indexing {len(to_index)} notes (this may take a few minutes)…")
    await rag.initialize_storages()

    batch_size = 10
    for i in range(0, len(to_index), batch_size):
        batch = to_index[i : i + batch_size]
        texts = [content for _, content in batch]
        await rag.ainsert(texts)
        # Record indexed
        with open(indexed_file, "a") as f:
            for path, _ in batch:
                f.write(path + "\n")
        done = min(i + batch_size, len(to_index))
        print(f"   {done}/{len(to_index)} notes indexed…")

    print(f"\n✅ Done — {len(to_index)} notes added to knowledge graph")
    print(f"   Storage: {RAG_STORAGE}")


async def cmd_query(rag, question: str, mode: str = "mix"):
    from lightrag import QueryParam

    await rag.initialize_storages()
    print(f"\n🔍 Query [{mode}]: {question}\n")
    result = await rag.aquery(question, param=QueryParam(mode=mode))
    print(result)


async def cmd_stats():
    notes = collect_notes()
    indexed_file = RAG_STORAGE / ".indexed_files"
    indexed = 0
    if indexed_file.exists():
        indexed = len(indexed_file.read_text().splitlines())

    print(f"\n📊 Obsidian Vault — LightRAG Stats")
    print(f"   Vault:         {VAULT}")
    print(f"   Notes found:   {len(notes)}")
    print(f"   Notes indexed: {indexed}")
    print(f"   RAG storage:   {RAG_STORAGE}")
    total_size = sum(f.stat().st_size for f in RAG_STORAGE.rglob("*") if f.is_file())
    print(f"   Index size:    {total_size / 1024 / 1024:.1f} MB")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LightRAG ↔ Obsidian Vault")
    sub = parser.add_subparsers(dest="cmd")

    p_index = sub.add_parser("index", help="Index vault notes")
    p_index.add_argument("--force", action="store_true", help="Re-index all notes")

    p_query = sub.add_parser("query", help="Query the knowledge graph")
    p_query.add_argument("question", help="Your question")
    p_query.add_argument("--mode", default="mix",
                         choices=["naive", "local", "global", "hybrid", "mix"],
                         help="Retrieval mode (default: mix)")

    sub.add_parser("stats", help="Show index statistics")

    args = parser.parse_args()

    if args.cmd == "stats":
        asyncio.run(cmd_stats())
        return

    if not args.cmd:
        parser.print_help()
        return

    check_env()
    rag = build_rag()

    if args.cmd == "index":
        asyncio.run(cmd_index(rag, force=getattr(args, "force", False)))
    elif args.cmd == "query":
        asyncio.run(cmd_query(rag, args.question, args.mode))


if __name__ == "__main__":
    main()
