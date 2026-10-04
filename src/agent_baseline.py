from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: Baseline Agent with short-term within-thread memory only.

    Characteristics:
    - Maintains conversation history solely within the active thread.
    - No persistent User.md file (memory growth = 0).
    - No compaction mechanism (carries full raw conversation context).
    - Forgets all facts across new threads/sessions.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = self._maybe_build_langchain_agent() if not force_offline else None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Process an incoming turn and return response along with token metrics."""
        if self.langchain_agent is not None and not self.force_offline:
            try:
                # Live execution path
                response = self.langchain_agent.invoke(
                    {"input": message},
                    config={"configurable": {"thread_id": thread_id}},
                )
                output_text = response.get("output", str(response))
                prompt_tokens = estimate_tokens(message)
                reply_tokens = estimate_tokens(output_text)
                session = self.sessions.setdefault(thread_id, SessionState())
                session.token_usage += reply_tokens
                session.prompt_tokens_processed += prompt_tokens
                return {"reply": output_text, "tokens": reply_tokens, "prompt_tokens": prompt_tokens}
            except Exception:
                # Fallback to deterministic offline mode on any runtime error
                pass

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str | None = None) -> int:
        """Return cumulative generated response tokens for a thread or all threads."""
        if thread_id is not None:
            session = self.sessions.get(thread_id)
            return session.token_usage if session else 0
        return sum(s.token_usage for s in self.sessions.values())

    def prompt_token_usage(self, thread_id: str | None = None) -> int:
        """Return cumulative prompt context tokens processed for a thread or all threads."""
        if thread_id is not None:
            session = self.sessions.get(thread_id)
            return session.prompt_tokens_processed if session else 0
        return sum(s.prompt_tokens_processed for s in self.sessions.values())

    def compaction_count(self, thread_id: str | None = None) -> int:
        """Baseline agent does not have a compaction manager."""
        return 0

    def memory_file_size(self, user_id: str) -> int:
        """Baseline agent does not write persistent profile files."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline mode for fair and reproducible benchmarking."""
        session = self.sessions.setdefault(thread_id, SessionState())

        # Baseline must carry the full thread history into prompt tokens on every turn
        prompt_parts = [f"{m['role']}: {m['content']}" for m in session.messages]
        prompt_parts.append(f"user: {message}")
        prompt_text = "\n".join(prompt_parts)
        turn_prompt_tokens = estimate_tokens(prompt_text)
        session.prompt_tokens_processed += turn_prompt_tokens

        # Baseline forgets everything when a new thread starts
        if len(session.messages) == 0:
            reply_text = (
                "Chào bạn! Vì đây là một phiên trò chuyện mới và tôi không có bộ nhớ dài hạn, "
                "tôi không có thông tin về bạn hay các chủ đề đã trao đổi từ các phiên trước."
            )
        else:
            reply_text = (
                f"Chào bạn! Tôi đã ghi nhận nội dung: '{message[:80]}...'. "
                "Tôi đang tiếp tục trao đổi trong cùng phiên làm việc này."
            )

        reply_tokens = estimate_tokens(reply_text)
        session.token_usage += reply_tokens

        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": reply_text})

        return {
            "reply": reply_text,
            "tokens": reply_tokens,
            "prompt_tokens": turn_prompt_tokens,
        }

    def _maybe_build_langchain_agent(self) -> Any:
        """Optionally instantiate a live LangChain agent if libraries are available."""
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            model = build_chat_model(self.config.model)
            checkpointer = MemorySaver()
            return create_react_agent(model=model, tools=[], checkpointer=checkpointer)
        except Exception:
            return None
