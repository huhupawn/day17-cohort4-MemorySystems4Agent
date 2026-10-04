from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated test configuration."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    dummy_provider = ProviderConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.0,
    )

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=50,  # low threshold for test triggers
        compact_keep_messages=2,
        model=dummy_provider,
        judge_model=dummy_provider,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify User.md can be created, read, edited, and queried for file size."""
    profiles_dir = tmp_path / "profiles"
    store = UserProfileStore(root_dir=profiles_dir)

    # Initial state
    assert store.read_text("test_user") == ""
    assert store.file_size("test_user") == 0

    # Write
    written_path = store.write_text("test_user", "# Profile\n- **Name**: Nguyen Van A\n- **City**: Hanoi\n")
    assert written_path.exists()
    assert store.file_size("test_user") > 0
    assert "Nguyen Van A" in store.read_text("test_user")

    # Edit
    success = store.edit_text("test_user", "Hanoi", "Da Nang")
    assert success is True
    updated = store.read_text("test_user")
    assert "Da Nang" in updated
    assert "Hanoi" not in updated

    # Failed edit on missing substring
    failed = store.edit_text("test_user", "MissingTarget", "Replacement")
    assert failed is False


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify that exceeding token thresholds triggers compaction."""
    manager = CompactMemoryManager(threshold_tokens=40, keep_messages=2)
    thread_id = "test-thread"

    # Append messages that each contain ~15 tokens, total will exceed 40
    manager.append(thread_id, "user", "Message 1: This is a sufficiently long message intended to increase token count.")
    manager.append(thread_id, "assistant", "Message 2: Assistant acknowledges the first message with additional tokens.")
    manager.append(thread_id, "user", "Message 3: Third turn pushing cumulative token count past forty tokens threshold.")
    manager.append(thread_id, "assistant", "Message 4: Fourth turn ensuring older items are pruned.")

    ctx = manager.context(thread_id)
    assert manager.compaction_count(thread_id) >= 1
    assert ctx["summary"] != ""
    assert len(ctx["messages"]) <= 2


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify Advanced remembers across separate threads while Baseline does not."""
    config = make_config(tmp_path)

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    # Thread 1: Introduce user
    intro_msg = "Chào bạn, mình tên là DũngCT và hiện đang ở Huế."
    baseline.reply(user_id="dungct", thread_id="thread-1", message=intro_msg)
    advanced.reply(user_id="dungct", thread_id="thread-1", message=intro_msg)

    # Thread 2: Query in fresh thread
    query_msg = "Mình tên gì và hiện đang ở đâu?"
    res_baseline = baseline.reply(user_id="dungct", thread_id="thread-2", message=query_msg)
    res_advanced = advanced.reply(user_id="dungct", thread_id="thread-2", message=query_msg)

    # Baseline has no cross-session memory
    assert "DũngCT" not in res_baseline["reply"]

    # Advanced recalls from User.md
    assert "DũngCT" in res_advanced["reply"]
    assert "Huế" in res_advanced["reply"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt tokens processed between Baseline and Advanced on long threads."""
    config = make_config(tmp_path)
    # Threshold 60 tokens, keep 2 messages
    config.compact_threshold_tokens = 60
    config.compact_keep_messages = 2

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    long_turns = [
        "Đoạn văn dài thứ 1 thảo luận về kiến trúc hệ thống và quy trình triển khai mô hình học máy trong doanh nghiệp.",
        "Đoạn văn dài thứ 2 phân tích trade-off giữa độ chính xác của mô hình và độ trễ phản hồi trong môi trường production.",
        "Đoạn văn dài thứ 3 mở rộng về việc quản trị chi phí token khi lượng người dùng đồng thời tăng đột biến.",
        "Đoạn văn dài thứ 4 trình bày giải pháp lưu trữ vector database và đánh chỉ mục để giảm thời gian truy vấn.",
        "Đoạn văn dài thứ 5 đề xuất cơ chế tóm tắt lịch sử hội thoại tự động khi độ dài phiên vượt ngưỡng an toàn.",
        "Đoạn văn dài thứ 6 kiểm tra việc đồng bộ dữ liệu giữa bộ nhớ tạm thời và hệ thống lưu trữ bền vững.",
        "Đoạn văn dài thứ 7 tổng kết các chỉ số hiệu năng đạt được sau khi áp dụng kỹ thuật compact memory.",
        "Đoạn văn dài thứ 8 đánh giá khả năng mở rộng của hệ thống trong kịch bản tải cao liên tục nhiều ngày.",
    ]

    thread_id = "long-stress-thread"
    for turn in long_turns:
        baseline.reply(user_id="test_user", thread_id=thread_id, message=turn)
        advanced.reply(user_id="test_user", thread_id=thread_id, message=turn)

    # Advanced must have triggered compaction
    assert advanced.compaction_count(thread_id) > 0

    # Advanced must process fewer prompt context tokens than Baseline
    assert advanced.prompt_token_usage(thread_id) < baseline.prompt_token_usage(thread_id)
