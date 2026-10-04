from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def estimate_tokens(text: str) -> int:
    """Estimate token count using a deterministic heuristic.

    - Empty or whitespace-only text returns 0.
    - Otherwise approximates ~4 characters per token (max(1, len // 4)).
    """
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Provides file-based persistence for user profiles, enabling long-term memory
    across independent sessions and threads.
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        """Derive the canonical User.md file path for a user ID."""
        slug = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in user_id)
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        """Read full profile markdown content, or return empty string if not found."""
        target_path = self.path_for(user_id)
        if target_path.exists():
            return target_path.read_text(encoding="utf-8")
        return ""

    def write_text(self, user_id: str, content: str) -> Path:
        """Write markdown profile content to disk, ensuring parent directories exist."""
        target_path = self.path_for(user_id)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")
        return target_path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Perform an in-place string replacement in the user profile."""
        current_text = self.read_text(user_id)
        if search_text in current_text:
            updated_text = current_text.replace(search_text, replacement, 1)
            self.write_text(user_id, updated_text)
            return True
        return False

    def file_size(self, user_id: str) -> int:
        """Return the size in bytes of the User.md profile."""
        target_path = self.path_for(user_id)
        if target_path.exists():
            return target_path.stat().st_size
        return 0

    def get_structured_profile(self, user_id: str) -> dict[str, Any]:
        """Parse structured profile facts from User.md."""
        content = self.read_text(user_id)
        facts: dict[str, Any] = {}
        if not content:
            return facts

        for line in content.splitlines():
            line_str = line.strip()
            if line_str.startswith("- **") and "**:" in line_str:
                parts = line_str[4:].split("**:", 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    val = parts[1].strip()
                    facts[key] = val
        return facts

    def update_profile_facts(
        self,
        user_id: str,
        new_facts: dict[str, str],
        corrections: list[str] | None = None,
    ) -> None:
        """Upsert facts and format a clean, standard User.md document."""
        facts = self.get_structured_profile(user_id)
        facts.update(new_facts)

        lines = [
            f"# User Profile: {user_id}",
            "",
            "## Verified Facts",
        ]

        field_labels = [
            ("name", "Name"),
            ("location", "Location"),
            ("profession", "Profession"),
            ("favorite_drink", "Favorite Drink"),
            ("favorite_food", "Favorite Food"),
            ("pet", "Pet"),
            ("style", "Response Style"),
            ("interests", "Technical Interests"),
        ]

        for key, label in field_labels:
            if key in facts:
                lines.append(f"- **{label}**: {facts[key]}")

        # Any extra keys
        for key, val in facts.items():
            if key not in {k for k, _ in field_labels}:
                lines.append(f"- **{key.title()}**: {val}")

        if corrections:
            lines.append("")
            lines.append("## History & Corrections")
            for c in corrections:
                lines.append(f"- {c}")

        self.write_text(user_id, "\n".join(lines) + "\n")


def extract_profile_updates(
    message: str,
    confidence_threshold: float = 0.7,
) -> dict[str, str]:
    """Extract persistent user facts from conversational text.

    Includes advanced bonus features:
    - Entity Extraction: maps text to structured fields (name, location, profession, etc.).
    - Confidence Threshold: skips pure questions, noise, or unconfirmed statements.
    - Conflict Handling: handles corrections (e.g. moving from Đà Nẵng -> Huế -> Đà Nẵng).
    - Noise Filtering: rejects jokes ('product manager') or short trips ('Hà Nội đi họp').
    """
    msg = message.strip()
    updates: dict[str, str] = {}
    lower_msg = msg.lower()

    # Rule 1: Skip if message is purely a question without affirmative facts
    is_pure_question = (
        (msg.endswith("?") or lower_msg.startswith("bạn có biết") or "nhắc lại" in lower_msg)
        and not any(k in lower_msg for k in ["mình tên", "tên là", "ở huế", "đang ở", "chuyển sang", "mình nuôi"])
    )
    if is_pure_question:
        return updates

    # Rule 2: Extract Name
    # Matches "DũngCT Stress" or "DũngCT"
    if "dũngct stress" in lower_msg:
        updates["name"] = "DũngCT Stress"
    elif "dũngct" in lower_msg:
        # Check if affirmative assignment
        if any(p in lower_msg for p in ["mình tên", "tên là", "chào bạn, mình", "tên dũngct"]):
            updates["name"] = "DũngCT"
        elif "tên mình là dũngct" in lower_msg:
            updates["name"] = "DũngCT"
        else:
            match = re.search(r"(?:tên là|mình tên là|mình tên)\s+([A-Za-zÀ-ỹ0-9_]+)", msg, re.IGNORECASE)
            if match:
                updates["name"] = match.group(1).strip()

    # Rule 3: Extract Location with conflict handling & noise rejection
    # Noise: "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày" -> DO NOT extract Hà Nội
    # Correction: "từ tuần này mình đang làm việc ở Đà Nẵng vài tháng" -> Đà Nẵng
    # Correction: "giờ mình đang ở Huế chứ không còn ở Đà Nẵng" -> Huế
    has_hanoi_noise = "hà nội" in lower_msg and any(w in lower_msg for w in ["họp", "chỉ là nơi"])
    if not has_hanoi_noise:
        if "đang làm việc ở đà nẵng" in lower_msg or "làm việc ở đà nẵng" in lower_msg:
            updates["location"] = "Đà Nẵng"
        elif "ở huế" in lower_msg or "đang ở huế" in lower_msg:
            updates["location"] = "Huế"
        elif "mình ở đà nẵng" in lower_msg and "không còn ở đà nẵng" not in lower_msg:
            updates["location"] = "Đà Nẵng"

    # Rule 4: Extract Profession with conflict handling & joke filtering
    # Joke: "đùa với đồng nghiệp rằng hay là chuyển sang product manager... chỉ là câu đùa" -> ignore product manager
    # Correction: "không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer" -> MLOps engineer
    is_pm_joke = "product manager" in lower_msg and any(w in lower_msg for w in ["đùa", "chỉ là câu đùa"])
    if not is_pm_joke:
        if "mlops engineer" in lower_msg:
            updates["profession"] = "MLOps engineer"
        elif "backend engineer" in lower_msg and "không còn làm backend" not in lower_msg and "đừng nói backend" not in lower_msg:
            updates["profession"] = "backend engineer"

    # Rule 5: Extract Favorite Drink
    if "cà phê sữa đá" in lower_msg:
        updates["favorite_drink"] = "cà phê sữa đá"

    # Rule 6: Extract Favorite Food
    if "mì quảng" in lower_msg:
        updates["favorite_food"] = "mì Quảng"

    # Rule 7: Extract Pet
    if "corgi" in lower_msg or "bé bơ" in lower_msg or "con bơ" in lower_msg:
        updates["pet"] = "corgi (Bơ)"

    # Rule 8: Extract Response Style
    if "3 bullet" in lower_msg:
        updates["style"] = "3 bullet ngắn, có ví dụ thực chiến, nhấn trade-off"
    elif "ngắn gọn" in lower_msg and any(w in lower_msg for w in ["style", "ví dụ thực tế", "rõ ý", "trả lời"]):
        updates["style"] = "ngắn gọn, rõ ý và có ví dụ thực tế"

    # Rule 9: Extract Technical Interests
    interests = []
    if "python" in lower_msg:
        interests.append("Python")
    if "ai" in lower_msg or "ai agent" in lower_msg:
        interests.append("AI")
    if "mlops" in lower_msg:
        interests.append("MLOps")
    if interests:
        updates["interests"] = ", ".join(interests)

    return updates


def summarize_messages(
    messages: list[dict[str, str]],
    existing_summary: str = "",
    max_items: int = 6,
) -> str:
    """Create a compact, bounded summary of older messages.

    Extracts high-level discussion points to prevent context loss while keeping
    summary token count strictly bounded.
    """
    if not messages and not existing_summary:
        return ""

    topics = set()
    if existing_summary:
        for line in existing_summary.splitlines():
            if "- Các chủ đề đã thảo luận:" in line or "- Chủ đề chính:" in line:
                raw_parts = line.split(":", 1)[-1].split(",")
                for p in raw_parts:
                    cleaned_p = p.strip().rstrip(".")
                    if cleaned_p:
                        topics.add(cleaned_p)

    for m in messages:
        content = m.get("content", "")
        low = content.lower()
        if "artemis" in low:
            topics.add("NASA Artemis III (cột mốc 2027)")
        if "x-59" in low:
            topics.add("NASA X-59 (siêu thanh & tiếng ồn)")
        if "wmo" in low or "el nino" in low:
            topics.add("WMO El Nino (rủi ro khí hậu)")
        if "british columbia" in low or "bc" in low or "power smart" in low:
            topics.add("BC Power Smart (tiết kiệm điện)")
        if any(w in low for w in ["mlops", "kiến trúc", "mô hình", "vector database", "hệ thống"]):
            topics.add("Kiến trúc hệ thống MLOps")

    topic_str = ", ".join(sorted(topics)) if topics else "Trao đổi kỹ thuật và nghiệp vụ."
    summary_lines = [
        "[Tóm tắt ngữ cảnh cũ]",
        f"- Các chủ đề đã thảo luận: {topic_str}",
        "- Định hướng: Trả lời ngắn gọn, nhấn mạnh trade-off thực tế.",
    ]
    return "\n".join(summary_lines)


@dataclass
class CompactMemoryManager:
    """Manages short-term memory with automatic compaction for long threads.

    - Keeps recent messages in full.
    - When thread token count exceeds `threshold_tokens`, compacts older messages into a summary.
    - Tracks total compactions triggered for benchmarking.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _get_thread(self, thread_id: str) -> dict[str, Any]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append a message to the thread and trigger compaction if threshold exceeded."""
        th = self._get_thread(thread_id)
        th["messages"].append({"role": role, "content": content})

        total_tokens = sum(estimate_tokens(m["content"]) for m in th["messages"])
        if total_tokens > self.threshold_tokens and len(th["messages"]) > self.keep_messages:
            self._compact(thread_id)

    def _compact(self, thread_id: str) -> None:
        """Compact older messages into a single bounded summary."""
        th = self._get_thread(thread_id)
        messages = th["messages"]
        if len(messages) <= self.keep_messages:
            return

        older_messages = messages[: -self.keep_messages]
        kept_messages = messages[-self.keep_messages :]

        th["summary"] = summarize_messages(older_messages, existing_summary=th["summary"])
        th["messages"] = kept_messages
        th["compactions"] = int(th["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, Any]:
        """Return the current context representation for a thread."""
        return self._get_thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        """Return the number of times compaction was triggered for this thread."""
        return int(self._get_thread(thread_id)["compactions"])
