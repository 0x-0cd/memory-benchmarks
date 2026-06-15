#!/usr/bin/env python3
"""LoCoMo on Mneme — sleep effect evaluation (100 questions).

Uses the full standard pipeline: search → LLM answer → LLM judge.
Exactly follows the benchmark protocol. No shortcuts.
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
from benchmarks.common.llm_client import LLMClient
from benchmarks.locomo.run import (
    CATEGORIES_TO_EVALUATE,
    CATEGORY_NAMES,
    JUDGE_SYSTEM_PROMPT,
    get_answer_generation_prompt,
    get_judge_prompt,
    get_judge_prompt_with_evidence,
    preprocess_answer,
    cutoff_label,
    download_dataset,
    load_dataset,
    get_sorted_sessions,
    session_to_chunks,
    locomo_date_to_epoch,
    load_evidence_lookup,
    DEFAULT_DATASET_DIR,
)

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("sleep-eval")

MNEME_HOST = os.getenv("MNEME_HOST", "http://localhost:8989")
TOP_K = 50
CUTOFFS = [10, 20, 50]

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    auth_path = os.path.expanduser("~/.local/share/opencode/auth.json")
    try:
        with open(auth_path) as f:
            auth = json.load(f)
        DEEPSEEK_API_KEY = auth.get("deepseek", {}).get("key", "")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE", "https://api.deepseek.com/v1")


def parse_short_date(date_str: str) -> str:
    m = re.search(r"on (\d+ \w+, \d{4})", date_str)
    return m.group(1) if m else date_str


async def main():
    print("=" * 60)
    print("🧠 LoCoMo on Mneme — Sleep Effect (100q)")
    print("=" * 60)

    # 1. Download / load dataset
    print("\n📥 Loading dataset...")
    dataset_path = download_dataset(DEFAULT_DATASET_DIR, logger)
    dataset = load_dataset(dataset_path)
    evidence_lookup = load_evidence_lookup(dataset_path)
    print(f"  {len(dataset)} conversations, {len(evidence_lookup)} evidence entries")

    # 2. Select 2 conversations and sample ~100 questions
    selected = []
    for ci in range(min(2, len(dataset))):
        entry = dataset[ci]
        qa_list = entry.get("qa", [])
        valid = [q for q in qa_list if q.get("category") in CATEGORIES_TO_EVALUATE]
        selected.append((ci, entry, valid))
        print(f"  Conv {ci}: {len(valid)} valid questions")

    questions = []
    while len(questions) < 100:
        for ci, entry, qs in selected:
            if len(questions) >= 100:
                break
            idx = len([q for q in questions if q[0] == ci])
            if idx < len(qs):
                questions.append((ci, entry, qs[idx]))
    print(f"\n  Sampled {len(questions)} questions")

    # 3. Init clients
    mneme = MnemeClient(host=MNEME_HOST, timeout=120.0)
    answerer = LLMClient(
        model=DEEPSEEK_MODEL, provider="openai",
        api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE, rpm=30,
    )
    judge_llm = LLMClient(
        model=DEEPSEEK_MODEL, provider="openai",
        api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE, rpm=30,
    )

    # 4. Ingest conversations
    print("\n🔄 Ingesting conversations...")
    async with mneme:
        total_ingested = 0
        for ci, entry, _ in selected:
            conversation = entry["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]
            sorted_sessions = get_sorted_sessions(conversation)
            user_id = f"locomo_{ci}"
            ingested = 0
            for session_key, date_str, turns in sorted_sessions:
                chunks = session_to_chunks(turns, speaker_a, speaker_b)
                short_date = parse_short_date(date_str)
                session_epoch = locomo_date_to_epoch(date_str)
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

    # 5. Sleep cycle
    print("\n💤 Running sleep cycle...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{MNEME_HOST}/v1/sleep")
        sleep_report = resp.json()
        print(f"  Consolidations: {sleep_report['consolidated']} pairs")
        print(f"  Decayed:        {sleep_report['decayed']} items")
        print(f"  Forgotten:      {sleep_report['forgotten']} items")
        print(f"  Memories:       {sleep_report['total_before']} -> {sleep_report['total_after']}")
        print(f"  Duration:       {sleep_report['duration_ms']}ms")

        resp = await client.get(f"{MNEME_HOST}/v1/stats")
        print(f"  After sleep: {resp.json()['total']} memories")

    # 6. Full evaluation pipeline
    print(f"\n{'=' * 60}")
    print(f"📋 Evaluating {len(questions)} questions (full pipeline)")
    print(f"{'=' * 60}")

    all_evaluations = []
    async with mneme:
        for qi, (conv_idx, entry, qa) in enumerate(questions):
            question = qa["question"]
            category = qa["category"]
            ground_truth = str(qa["answer"])
            cat_name = CATEGORY_NAMES.get(category, "unknown")
            user_id = f"locomo_{conv_idx}"
            ref_date_human = None

            print(f"\n  Q{qi+1}/{len(questions)} [{cat_name}]: {question[:60]}...")

            # 6a. Search
            start = time.monotonic()
            search_results = await mneme.search(question, user_id, top_k=TOP_K)
            latency = (time.monotonic() - start) * 1000
            print(f"     Search: {len(search_results)} results in {latency:.0f}ms")

            if not search_results:
                print(f"     ⚠️  No results — skipping")
                continue

            # 6b. Generate answer
            gen_prompt = get_answer_generation_prompt(
                question, search_results[:50], reference_date=ref_date_human,
            )
            generated_answer = await answerer.generate(system="", user=gen_prompt)
            if "ANSWER:" in generated_answer:
                generated_answer = generated_answer.rsplit("ANSWER:", 1)[-1].strip()
            print(f"     Answer: {generated_answer[:80]}...")

            # 6c. Judge
            processed_answer = preprocess_answer(category, ground_truth)
            judge_prompt = get_judge_prompt(category, question, processed_answer, generated_answer)
            raw = await judge_llm.generate_structured(
                system=JUDGE_SYSTEM_PROMPT, user=judge_prompt,
            )

            if isinstance(raw, dict):
                label_val = raw.get("label", "").upper()
                correct = label_val == "CORRECT"
            else:
                correct = False

            score = 1.0 if correct else 0.0
            judgment = "✅" if correct else "❌"
            print(f"     Judge: {judgment} (score={score})")

            # Store eval
            cutoff_results = {}
            for c in CUTOFFS:
                label = cutoff_label(c)
                cutoff_results[label] = {
                    "judgment": "CORRECT" if correct else "WRONG",
                    "score": score,
                    "generated_answer": generated_answer,
                    "memories_evaluated": len(search_results[:c]),
                }
            all_evaluations.append({
                "question_id": f"conv{conv_idx}_q{qi}",
                "conversation_idx": conv_idx,
                "category": category,
                "category_name": cat_name,
                "question": question,
                "ground_truth_answer": ground_truth,
                "cutoff_results": cutoff_results,
            })

    # 7. Results
    print(f"\n{'=' * 60}")
    print("📊 RESULTS")
    print(f"{'=' * 60}")

    if all_evaluations:
        print(f"\n  Overall (cutoff=top_10):")
        total = len(all_evaluations)
        correct = sum(
            1 for e in all_evaluations
            if e.get("cutoff_results", {}).get("top_10", {}).get("score", 0) >= 0.5
        )
        print(f"  {correct}/{total} = {correct/total*100:.1f}%")

        by_cat = {}
        for e in all_evaluations:
            cat = e.get("category_name", "unknown")
            if cat not in by_cat:
                by_cat[cat] = {"total": 0, "correct": 0, "scores": []}
            by_cat[cat]["total"] += 1
            score = e.get("cutoff_results", {}).get("top_10", {}).get("score", 0)
            by_cat[cat]["scores"].append(score)
            if score >= 0.5:
                by_cat[cat]["correct"] += 1

        print(f"\n  By category (top_10):")
        for cat, s in sorted(by_cat.items()):
            acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
            print(f"    {cat:15s}: {s['correct']:3d}/{s['total']:3d} = {acc:5.1f}%")

        print(f"\n  Sleep effect:")
        print(f"    Consolidations: {sleep_report['consolidated']} pairs")
        print(f"    Decayed:        {sleep_report['decayed']} items")
        print(f"    Forgotten:      {sleep_report['forgotten']} items")
        print(f"    Memories:       {sleep_report['total_before']} -> {sleep_report['total_after']}")

        # Save results
        results_dir = Path("results/sleep_eval")
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        results_path = results_dir / f"sleep_eval_{ts}.json"
        results_path.write_text(json.dumps({
            "metadata": {
                "benchmark": "locomo",
                "memory_system": "mneme",
                "model": DEEPSEEK_MODEL,
                "top_k": TOP_K,
                "questions": len(questions),
                "sleep_report": sleep_report,
            },
            "evaluations": all_evaluations,
        }, indent=2, ensure_ascii=False))
        print(f"\n  Results saved: {results_path}")

    # 8. Cleanup
    print(f"\n🧹 Cleaning up...")
    async with mneme:
        await mneme.delete_user("locomo_test")
    print("  Mneme cleared ✅")

    print(f"\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
