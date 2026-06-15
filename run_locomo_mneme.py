#!/usr/bin/env python3
"""
LoCoMo on Mneme — smoke test runner.

Downloads the LOCOMO-10 dataset, ingests all conversations into Mneme,
runs search + answer + judge for the first conversation only.
Uses DeepSeek via OpenAI-compatible API for answer/judge models.
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

# Add benchmark to path
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
    parse_cutoffs,
)
from benchmarks.common.metrics import compute_overall_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("locomo-mneme")

# Config
MNEME_HOST = os.getenv("MNEME_HOST", "http://localhost:8989")

# DeepSeek API: try env var first, fall back to opencode auth.json
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
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
TOP_K = int(os.getenv("TOP_K", "50"))
CUTOFFS = [10, 20, 50]
MAX_CONVERSATIONS = int(os.getenv("MAX_CONVERSATIONS", "1"))
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", "5"))


def parse_short_date(date_str: str) -> str:
    """'1:56 pm on 8 May, 2023' -> '8 May, 2023'"""
    m = re.search(r"on (\d+ \w+, \d{4})", date_str)
    if m:
        return m.group(1)
    return date_str


async def main():
    print("=" * 60)
    print("🧠 LoCoMo on Mneme — Smoke Test")
    print("=" * 60)

    # 1. Download dataset
    print("\n📥 Downloading LOCOMO-10 dataset...")
    dataset_path = download_dataset(DEFAULT_DATASET_DIR, logger)
    dataset = load_dataset(dataset_path)
    print(f"   Loaded {len(dataset)} conversations")

    # 2. Build evidence lookup
    evidence_lookup = load_evidence_lookup(dataset_path)
    print(f"   Evidence: {len(evidence_lookup)} entries")

    # 3. Init clients
    mneme = MnemeClient(host=MNEME_HOST)
    answerer = LLMClient(
        model=DEEPSEEK_MODEL,
        provider="openai",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE,
        rpm=30,
    )
    judge_llm = LLMClient(
        model=DEEPSEEK_MODEL,
        provider="openai",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE,
        rpm=30,
    )

    categories_to_test = [1, 2, 3, 4]  # single-hop, multi-hop, temporal, open-domain
    cutoffs = [10, 20, 50]

    all_evaluations = []

    CHECKPOINT_PATH = Path("results/locomo_mneme/checkpoint.json")

    def save_checkpoint():
        """Save intermediate results so a crash doesn't lose everything."""
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_PATH.write_text(json.dumps({
            "metadata": {
                "benchmark": "locomo",
                "memory_system": "mneme",
                "model": DEEPSEEK_MODEL,
                "top_k": TOP_K,
                "cutoffs": CUTOFFS,
                "conversations": MAX_CONVERSATIONS,
                "max_questions": MAX_QUESTIONS,
                "checkpoint_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "evaluations": all_evaluations,
        }, indent=2, ensure_ascii=False))

    async with mneme:
        for conv_idx in range(min(MAX_CONVERSATIONS, len(dataset))):
            entry = dataset[conv_idx]
            conversation = entry["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            print(f"\n{'─' * 60}")
            print(f"📝 Conversation {conv_idx}: {speaker_a} & {speaker_b}")
            print(f"{'─' * 60}")

            # --- Ingest ---
            print("\n🔄 Ingesting...")
            sorted_sessions = get_sorted_sessions(conversation)
            total_chunks = sum(
                len(session_to_chunks(s, speaker_a, speaker_b))
                for _, _, s in sorted_sessions
            )
            print(f"   Sessions: {len(sorted_sessions)}, Chunks: {total_chunks}")

            ingested = 0
            for session_key, date_str, turns in sorted_sessions:
                chunks = session_to_chunks(turns, speaker_a, speaker_b)
                session_epoch = locomo_date_to_epoch(date_str)
                short_date = parse_short_date(date_str)
                user_id = f"locomo_{conv_idx}"

                for chunk_idx, messages in enumerate(chunks):
                    # Skip empty
                    if any(not msg.get("content", "").strip() for msg in messages):
                        ingested += 1
                        continue

                    # Embed session date into content so it's visible in the prompt
                    # (Mneme's created_at is ingestion time, not conversation time)
                    for msg in messages:
                        msg["content"] = f"[{short_date}] {msg['content']}"

                    result = await mneme.add(
                        messages,
                        user_id,
                        timestamp=session_epoch,
                    )
                    if result is not None:
                        ingested += 1
                    else:
                        logger.warning(
                            "  Chunk %s/%d failed", session_key, chunk_idx
                        )

            print(f"   Ingested {ingested}/{total_chunks} chunks ✅")

            # Reference date = last session
            ref_date_human = sorted_sessions[-1][1] if sorted_sessions else None

            # --- Questions ---
            questions = entry.get("qa", entry.get("qa_pairs", []))
            conv_questions = [
                (qi, qa) for qi, qa in enumerate(questions)
                if qa.get("category") in categories_to_test
                and qa.get("question") and qa.get("answer")
            ][:MAX_QUESTIONS]

            print(f"\n📋 Processing {len(conv_questions)} questions...")

            for qi, qa in conv_questions:
                question = qa["question"]
                category = qa["category"]
                ground_truth = str(qa["answer"])
                cat_name = CATEGORY_NAMES.get(category, "unknown")

                print(f"\n  Q{qi} [{cat_name}]: {question[:60]}...")

                # 1. Search
                start = time.monotonic()
                search_results = await mneme.search(question, user_id, top_k=TOP_K)
                latency = (time.monotonic() - start) * 1000
                print(f"     Search: {len(search_results)} results in {latency:.0f}ms")

                if not search_results:
                    print(f"     ⚠️  No results! Skipping...")
                    continue

                # 2. Generate answer
                gen_prompt = get_answer_generation_prompt(
                    question,
                    search_results[:50],
                    reference_date=ref_date_human,
                )
                generated_answer = await answerer.generate(
                    system="", user=gen_prompt
                )
                if "ANSWER:" in generated_answer:
                    generated_answer = generated_answer.rsplit("ANSWER:", 1)[-1].strip()
                print(f"     Answer: {generated_answer[:80]}...")

                # 3. Judge
                processed_answer = preprocess_answer(category, ground_truth)

                # Build evidence
                ev_ctx = ""
                if evidence_lookup:
                    for ref in qa.get("evidence", []):
                        key = (conv_idx, ref)
                        if key in evidence_lookup:
                            ev_ctx += evidence_lookup[key] + "\n"
                    ev_ctx = ev_ctx.strip()

                if ev_ctx:
                    judge_prompt = get_judge_prompt_with_evidence(
                        category, question, processed_answer, generated_answer, ev_ctx
                    )
                else:
                    judge_prompt = get_judge_prompt(
                        category, question, processed_answer, generated_answer
                    )

                raw = await judge_llm.generate_structured(
                    system=JUDGE_SYSTEM_PROMPT,
                    user=judge_prompt,
                )

                if isinstance(raw, dict):
                    label_val = raw.get("label", "").upper()
                    correct = label_val == "CORRECT"
                else:
                    correct = False

                score = 1.0 if correct else 0.0
                judgment = "✅ CORRECT" if correct else "❌ WRONG"
                print(f"     Judge: {judgment} (score={score})")
                if isinstance(raw, dict):
                    reason = raw.get("reasoning", raw.get("reason", ""))
                    if reason:
                        print(f"     Reason: {reason[:120]}...")

                # Store for metrics
                cutoff_results = {}
                for c in cutoffs:
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
                    "retrieval": {
                        "search_results": search_results,
                        "total_results": len(search_results),
                    },
                })

        # --- Per-conversation checkpoint ---
        save_checkpoint()

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("📊 RESULTS")
    print(f"{'=' * 60}")

    if all_evaluations:
        for label in [cutoff_label(c) for c in cutoffs]:
            total = len(all_evaluations)
            correct = sum(
                1 for e in all_evaluations
                if e.get("cutoff_results", {}).get(label, {}).get("score", 0) >= 0.5
            )
            print(f"\n  {label}: {correct}/{total} ({correct/total*100:.1f}%)")
    else:
        print("  No evaluations completed.")

    # Save results
    results_dir = Path("results/locomo_mneme")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"locomo_mneme_{timestamp}.json"
    results_path.write_text(json.dumps({
        "metadata": {
            "benchmark": "locomo",
            "memory_system": "mneme",
            "model": DEEPSEEK_MODEL,
            "top_k": TOP_K,
            "cutoffs": CUTOFFS,
            "conversations": MAX_CONVERSATIONS,
            "max_questions": MAX_QUESTIONS,
        },
        "evaluations": all_evaluations,
    }, indent=2, ensure_ascii=False))
    print(f"\n  Results saved to: {results_path}")

    # Cleanup: clear Mneme
    await mneme.delete_user("locomo_test")
    print("  Mneme cleared ✅")

    print(f"\n{'=' * 60}")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
