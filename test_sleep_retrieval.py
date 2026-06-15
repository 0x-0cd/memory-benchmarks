"""Test sleep computation effect on retrieval.

Ingests LoCoMo data, runs sleep cycle, tests 100 questions for search recall.
Uses MnemeClient (with retries) and longer timeouts.
No external LLM API needed.
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks.common.mneme_client import MnemeClient
from benchmarks.locomo.run import (
    CATEGORIES_TO_EVALUATE,
    CATEGORY_NAMES,
    download_dataset,
    load_dataset,
    get_sorted_sessions,
    session_to_chunks,
    load_evidence_lookup,
    DEFAULT_DATASET_DIR,
)

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("sleep-test")

MNEME_HOST = os.getenv("MNEME_HOST", "http://localhost:8989")


def parse_short_date(date_str: str) -> str:
    m = re.search(r"on (\d+ \w+, \d{4})", date_str)
    return m.group(1) if m else date_str


def build_evidence_text(evidence_lookup: dict, conv_idx: int, evidence_refs: list[str]) -> str:
    texts = []
    for ref in evidence_refs:
        key = (conv_idx, ref)
        if key in evidence_lookup:
            texts.append(evidence_lookup[key])
    return " ".join(texts)


async def main():
    print("=" * 60)
    print("🧠 Sleep effect on retrieval")
    print("=" * 60)

    # 1. Load dataset
    print("\n[1/7] Loading LOCOMO-10...")
    dataset_path = download_dataset(DEFAULT_DATASET_DIR, logger)
    dataset = load_dataset(dataset_path)
    evidence_lookup = load_evidence_lookup(dataset_path)
    print(f"  {len(dataset)} conversations, {len(evidence_lookup)} evidence entries")

    # 2. Select conversations
    selected = []
    for ci in range(min(2, len(dataset))):
        entry = dataset[ci]
        qa_list = entry.get("qa", [])
        valid = [q for q in qa_list if q.get("category") in CATEGORIES_TO_EVALUATE]
        selected.append((ci, entry, valid))
        print(f"  Conv {ci}: {len(valid)} valid questions")

    # 3. Sample 100 questions round-robin
    questions = []
    while len(questions) < 100:
        for ci, entry, qs in selected:
            if len(questions) >= 100:
                break
            idx = len([q for q in questions if q[0] == ci])
            if idx < len(qs):
                questions.append((ci, entry, qs[idx], qs[idx].get("evidence", [])))
    print(f"\n[2/7] Sampled {len(questions)} questions")

    # 4. Ingest conversations via MnemeClient (has retries)
    print("\n[3/7] Ingesting conversations...")
    mneme = MnemeClient(host=MNEME_HOST, timeout=120.0)

    async with mneme:
        total_ingested = 0
        for ci, entry, _, _ in [(ci, entry, None, None) for ci, entry, _ in selected]:
            conversation = entry["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]
            sorted_sessions = get_sorted_sessions(conversation)
            user_id = f"locomo_{ci}"
            ingested = 0

            for session_key, date_str, turns in sorted_sessions:
                chunks = session_to_chunks(turns, speaker_a, speaker_b)
                short_date = parse_short_date(date_str)
                session_epoch = int(time.mktime(time.strptime(date_str.split("on ")[-1], "%d %B, %Y")))

                for messages in chunks:
                    if any(not msg.get("content", "").strip() for msg in messages):
                        ingested += 1
                        continue
                    for msg in messages:
                        msg["content"] = f"[{short_date}] {msg['content']}"
                    result = await mneme.add(messages, user_id, timestamp=session_epoch)
                    if result is not None:
                        ingested += 1

            total_ingested += ingested
            print(f"  Conv {ci}: {ingested} chunks")

        print(f"  Total: {total_ingested} memories")

    # 5. Stats before sleep
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{MNEME_HOST}/v1/stats")
        stats_before = resp.json()
        print(f"\n[4/7] Before sleep: {stats_before['total']} memories (by type: {stats_before.get('by_type', {})})")

        # 6. Run sleep cycle
        print("\n[5/7] Running sleep cycle...")
        resp = await client.post(f"{MNEME_HOST}/v1/sleep", timeout=60.0)
        sleep = resp.json()
        print(f"  Consolidations: {sleep['consolidated']} pairs")
        print(f"  Decayed:        {sleep['decayed']} items")
        print(f"  Forgotten:      {sleep['forgotten']} items")
        print(f"  Memories:       {sleep['total_before']} -> {sleep['total_after']}")
        print(f"  Duration:       {sleep['duration_ms']}ms")

        resp = await client.get(f"{MNEME_HOST}/v1/stats")
        stats_after = resp.json()
        print(f"  After sleep: {stats_after['total']} memories")

    # 7. Search test
    print(f"\n[6/7] Testing {len(questions)} questions for search recall...")
    cats = {}
    no_results = 0
    evidence_hits = 0
    top1_hits = 0
    SHOW = 10

    async with httpx.AsyncClient(timeout=60.0) as client:
        for qi, (conv_idx, entry, qa, evidence_refs) in enumerate(questions):
            question = qa["question"]
            category = qa["category"]
            cat_name = CATEGORY_NAMES.get(category, "unknown")
            ground_truth = str(qa["answer"])
            user_id = f"locomo_{conv_idx}"

            if cat_name not in cats:
                cats[cat_name] = {"n": 0, "hits": 0, "top1": 0}
            cats[cat_name]["n"] += 1

            resp = await client.post(
                f"{MNEME_HOST}/v1/memories/search",
                json={"query": question, "limit": 50},
            )
            search_results = resp.json().get("results", [])

            if not search_results:
                no_results += 1
                if qi < SHOW:
                    print(f"\n  Q{qi+1} [{cat_name}]: {question[:50]}...")
                    print(f"     No results")
                continue

            evidence_text = build_evidence_text(evidence_lookup, conv_idx, evidence_refs)
            ev_words = set(evidence_text.lower().split()) if evidence_text else set()
            gt_words = set(ground_truth.lower().split())

            hit = False
            top1 = False
            for ri, r in enumerate(search_results):
                content = r.get("content", r.get("memory", "")).lower()
                if (ev_words and ev_words & set(content.split())) or \
                   (gt_words and gt_words & set(content.split())):
                    hit = True
                    if ri == 0:
                        top1 = True
                    break

            if hit:
                evidence_hits += 1
                cats[cat_name]["hits"] += 1
            if top1:
                top1_hits += 1
                cats[cat_name]["top1"] += 1

            if qi < SHOW:
                print(f"\n  Q{qi+1} [{cat_name}]: {question[:50]}...")
                print(f"     GT: {ground_truth[:60]}")
                print(f"     Results: {len(search_results)} | Hit: {'✅' if hit else '❌'}")
                if search_results:
                    print(f"     Top: {search_results[0].get('content','')[:60]}...")

    # 8. Summary
    print(f"\n{'=' * 60}")
    print("📊 RESULTS")
    print(f"{'=' * 60}")
    n_total = len(questions)
    n_with = n_total - no_results
    print(f"\n  Total questions:  {n_total}")
    print(f"  With results:    {n_with}")
    print(f"  No results:      {no_results}")
    print(f"  Evidence hit:    {evidence_hits}/{n_with} ({evidence_hits/max(n_with,1)*100:.1f}%)")
    print(f"  Top-1 hit:       {top1_hits}/{n_with} ({top1_hits/max(n_with,1)*100:.1f}%)")

    print(f"\n  By category:")
    for cat, s in sorted(cats.items()):
        hit_pct = s["hits"] / max(s["n"], 1) * 100
        top1_pct = s["top1"] / max(s["n"], 1) * 100
        print(f"    {cat:15s}: {s['n']:3d}q | hit {s['hits']:3d} ({hit_pct:5.1f}%) | top1 {s['top1']:2d} ({top1_pct:5.1f}%)")

    print(f"\n  Sleep effect:")
    print(f"    Consolidations: {sleep['consolidated']} pairs")
    print(f"    Decayed:        {sleep['decayed']} items")
    print(f"    Forgotten:      {sleep['forgotten']} items")
    print(f"    Memories:       {sleep['total_before']} -> {sleep['total_after']}")

    # 9. Cleanup
    print(f"\n[7/7] Cleaning up...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        await client.delete(f"{MNEME_HOST}/v1/memories")
    print("  Mneme cleared ✅")

    print(f"\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
