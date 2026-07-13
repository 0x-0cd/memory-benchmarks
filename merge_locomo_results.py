"""merge_locomo_results.py — 合并多个 LoCoMo 评测结果文件

用法：
    python merge_locomo_results.py results/locomo_mneme/locomo_*.json -o results/locomo_mneme/merged.json

输出：
    合并后的 UnifiedResult JSON，包含全量 evaluations + 按 cutoff 的 metrics
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks.common.metrics import compute_overall_metrics
from benchmarks.common.schema import UnifiedResult
from benchmarks.common.utils import cutoff_label


def _expand_inputs(patterns: list[str]) -> list[str]:
    """Accept explicit files and/or glob patterns; return sorted unique paths."""
    paths: set[str] = set()
    for p in patterns:
        matches = glob.glob(p) if any(c in p for c in "*?[") else ([p] if os.path.exists(p) else [])
        if not matches and not os.path.exists(p):
            print(f"  ⚠️  No match: {p}", file=sys.stderr)
        paths.update(matches)
    return sorted(paths)


def merge(files: list[str]) -> tuple[list[dict], dict, list[str]]:
    """Merge evaluations from multiple result JSON files.

    Returns (merged_evaluations, merged_metadata, merged_from).
    """
    all_evaluations: list[dict] = []
    seen_ids: set[str] = set()
    merged_from = []
    base_metadata: dict = {}

    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        if base_metadata == {} and isinstance(data.get("metadata"), dict):
            base_metadata = dict(data["metadata"])
        merged_from.append(os.path.basename(fp))
        for e in data.get("evaluations", []):
            qid = e.get("question_id") or f"conv{e.get('conversation_idx')}_q{e.get('qi', '')}"
            if qid in seen_ids:
                continue
            seen_ids.add(qid)
            all_evaluations.append(e)
    return all_evaluations, base_metadata, merged_from


def main() -> None:
    parser = argparse.ArgumentParser(description="合并多个 LoCoMo 评测结果 JSON 文件")
    parser.add_argument("inputs", nargs="+", help="结果文件路径或 glob 模式（支持多个）")
    parser.add_argument("-o", "--output", required=True, help="合并后输出路径")
    parser.add_argument(
        "--cutoffs", default="10,20,50",
        help="逗号分隔的 cutoff 数值列表（默认 10,20,50）",
    )
    parser.add_argument("--group-key", default="category_name", help="分组键（默认 category_name）")
    args = parser.parse_args()

    files = _expand_inputs(args.inputs)
    if not files:
        print("❌ 没有找到任何输入文件", file=sys.stderr)
        sys.exit(1)
    print(f"📥 合并 {len(files)} 个文件：")
    for fp in files:
        print(f"   - {fp}")

    merged_evaluations, base_metadata, merged_from = merge(files)
    print(f"\n   合并后：{len(merged_evaluations)} 条 evaluations")

    cutoff_nums = [int(c.strip()) for c in args.cutoffs.split(",") if c.strip()]
    cutoff_labels = [cutoff_label(c) for c in cutoff_nums]

    metrics = compute_overall_metrics(
        merged_evaluations, args.group_key, cutoffs=cutoff_labels
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    extras = {
        "memory_system": base_metadata.get("memory_system", "mneme"),
        "mode": base_metadata.get("mode", "local"),
        "merged": True,
        "merged_from": merged_from,
        "merge_timestamp": timestamp,
    }
    for k in ("batch_name", "conv_start", "conv_end", "conversations", "max_questions", "top_k", "model"):
        if k in base_metadata:
            extras[k] = base_metadata[k]

    metadata = {
        "benchmark": base_metadata.get("benchmark", "locomo"),
        "timestamp": timestamp,
        "answerer_model": base_metadata.get("model", ""),
        "top_k": base_metadata.get("top_k", 0),
        "top_k_cutoffs": cutoff_labels,
        "total_questions": len(merged_evaluations),
        "config": extras,
    }

    unified = UnifiedResult(
        metadata=metadata,
        metrics=metrics,
    )
    payload = unified.model_dump()
    payload["evaluations"] = merged_evaluations
    # Surface merged_from / memory_system at top-level metadata for convenience too
    payload["metadata"].update(extras)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    )
    print(f"\n💾 已写入：{output_path}")
    print(f"\n{'=' * 60}")
    print("📊 概览")
    print(f"{'=' * 60}")
    print(f"  总问题数: {len(merged_evaluations)}")
    for label in cutoff_labels:
        cm = metrics.by_cutoff.get(label)
        if cm and cm.overall:
            o = cm.overall
            print(
                f"  {label}: {o.get('correct', 0)}/{o.get('total', 0)} "
                f"({o.get('accuracy', 0):.1f}%)"
            )


if __name__ == "__main__":
    main()