from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return recall fraction based on expected substrings found in answer."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    hits = sum(1 for exp in expected if exp.lower() in ans_lower)
    return hits / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for offline responses (0.0 to 1.0).

    Components:
    - 60% Recall accuracy
    - 20% Formatting and structured bullets
    - 20% Non-empty and concise length
    """
    recall = recall_points(answer, expected)
    if recall == 0.0:
        return 0.1

    lines = answer.splitlines()
    has_bullets = any(line.strip().startswith(("-", "*", "•")) for line in lines)
    format_bonus = 0.2 if has_bullets else 0.1
    length_bonus = 0.2 if 20 <= len(answer) <= 900 else 0.1

    return min(1.0, round(recall * 0.6 + format_bonus + length_bonus, 2))


def run_agent_benchmark(
    agent_name: str,
    agent: Any,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Evaluate an agent across conversations and test recall in fresh threads."""
    total_recall_points = 0.0
    total_quality_points = 0.0
    num_questions = 0

    for conv in conversations:
        user_id = conv["user_id"]
        thread_id = conv["id"]

        # 1. Feed turns in original conversation thread
        for turn in conv["turns"]:
            agent.reply(user_id=user_id, thread_id=thread_id, message=turn)

        # 2. Ask recall questions in FRESH threads to test cross-session memory
        for i, q_item in enumerate(conv.get("recall_questions", [])):
            question = q_item["question"]
            expected = q_item["expected_contains"]
            recall_thread = f"recall-{conv['id']}-{i}"
            res = agent.reply(user_id=user_id, thread_id=recall_thread, message=question)
            ans = res["reply"]

            r_score = recall_points(ans, expected)
            q_score = heuristic_quality(ans, expected)
            total_recall_points += r_score
            total_quality_points += q_score
            num_questions += 1

    avg_recall = (total_recall_points / num_questions) if num_questions > 0 else 0.0
    avg_quality = (total_quality_points / num_questions) if num_questions > 0 else 0.0

    user_ids = {conv["user_id"] for conv in conversations}
    total_mem_bytes = sum(agent.memory_file_size(u) for u in user_ids)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent.token_usage(),
        prompt_tokens_processed=agent.prompt_token_usage(),
        recall_score=round(avg_recall, 4),
        response_quality=round(avg_quality, 4),
        memory_growth_bytes=total_mem_bytes,
        compactions=agent.compaction_count(),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a clean table."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = [
        [
            r.agent_name,
            f"{r.agent_tokens_only:,}",
            f"{r.prompt_tokens_processed:,}",
            f"{r.recall_score * 100:.1f}%",
            f"{r.response_quality:.2f}",
            f"{r.memory_growth_bytes:,} B",
            r.compactions,
        ]
        for r in rows
    ]

    try:
        from tabulate import tabulate

        return tabulate(table_data, headers=headers, tablefmt="github")
    except ImportError:
        # Fallback to manual markdown table formatting
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        data_lines = [
            "| " + " | ".join(str(cell) for cell in row) + " |" for row in table_data
        ]
        return "\n".join([header_line, sep_line] + data_lines)


def main() -> None:
    """Run both standard benchmark and long-context stress benchmark."""
    config = load_config(Path(__file__).resolve().parent.parent)

    std_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    print("=" * 80)
    print("PHASE 2 - TRACK 3 - DAY 17: MEMORY SYSTEMS BENCHMARK")
    print("=" * 80)

    # 1. Standard Benchmark (10 conversations)
    print("\n[1] STANDARD BENCHMARK (10 Conversations, Cross-Session Recall)")
    std_convs = load_conversations(std_path)

    baseline_std = BaselineAgent(config=config, force_offline=True)
    row_baseline_std = run_agent_benchmark("Baseline Agent", baseline_std, std_convs, config)

    advanced_std = AdvancedAgent(config=config, force_offline=True)
    row_advanced_std = run_agent_benchmark("Advanced Agent", advanced_std, std_convs, config)

    print(format_rows([row_baseline_std, row_advanced_std]))

    # 2. Long-Context Stress Benchmark
    print("\n[2] LONG-CONTEXT STRESS BENCHMARK (16 Long Turns, Compact Memory)")
    stress_convs = load_conversations(stress_path)

    baseline_stress = BaselineAgent(config=config, force_offline=True)
    row_baseline_stress = run_agent_benchmark(
        "Baseline Agent", baseline_stress, stress_convs, config
    )

    advanced_stress = AdvancedAgent(config=config, force_offline=True)
    row_advanced_stress = run_agent_benchmark(
        "Advanced Agent", advanced_stress, stress_convs, config
    )

    print(format_rows([row_baseline_stress, row_advanced_stress]))
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
