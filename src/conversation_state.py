"""Disk-backed conversation state for chat_id reuse on standard OpenAI requests."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from .helpers import error_log


MAX_CACHE_ENTRIES = 4096
BOOTSTRAP_HISTORY_MAX_CHARS = 12000
DEFAULT_RETENTION_DAYS = 7
STATE_FILE_PATH = Path(__file__).resolve().parent.parent / "data" / "conversation_state.json"


@dataclass
class ConversationSession:
    chat_id: str
    token: str
    last_request_id: str
    model: str
    updated_at: float = field(default_factory=time.time)


class ConversationState:
    def __init__(self) -> None:
        self._lock = Lock()
        self._history_to_session: Dict[str, ConversationSession] = {}
        self._chat_to_keys: Dict[str, set[str]] = {}
        self._state_file_path = STATE_FILE_PATH
        self._load_from_disk()

    def resolve_session(self, model: str, messages: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[ConversationSession]]:
        history_messages = self._history_messages(messages)
        if not history_messages:
            return None, None

        history_key = self._build_history_key(model, history_messages)
        with self._lock:
            changed = self._cleanup_locked()
            session = self._history_to_session.get(history_key)
            if session:
                session.updated_at = time.time()
                changed = True
            if changed:
                self._persist_locked()
        return history_key, session

    def store_next_turn(
        self,
        model: str,
        request_messages: List[Dict[str, Any]],
        assistant_message: Dict[str, Any],
        chat_id: str,
        token: str,
        last_request_id: str,
    ) -> Optional[str]:
        normalized_messages = self._sanitize_messages(request_messages)
        if not normalized_messages:
            return None

        next_history_messages = normalized_messages + [self._sanitize_message(assistant_message)]
        history_key = self._build_history_key(model, next_history_messages)
        session = ConversationSession(
            chat_id=chat_id,
            token=token,
            last_request_id=last_request_id,
            model=model,
        )

        with self._lock:
            self._cleanup_locked()
            previous = self._history_to_session.get(history_key)
            if previous:
                previous_keys = self._chat_to_keys.get(previous.chat_id)
                if previous_keys:
                    previous_keys.discard(history_key)
                    if not previous_keys:
                        self._chat_to_keys.pop(previous.chat_id, None)

            self._history_to_session[history_key] = session
            self._chat_to_keys.setdefault(chat_id, set()).add(history_key)
            self._trim_locked()
            self._persist_locked()

        return history_key

    def invalidate_chat(self, chat_id: str) -> None:
        if not chat_id:
            return

        with self._lock:
            keys = self._chat_to_keys.pop(chat_id, set())
            changed = bool(keys)
            for key in keys:
                self._history_to_session.pop(key, None)
            if self._cleanup_locked() or changed:
                self._persist_locked()

    def build_bootstrap_content(self, messages: List[Dict[str, Any]]) -> str:
        normalized_messages = self._sanitize_messages(messages)
        if not normalized_messages:
            return "..."

        latest_message = normalized_messages[-1]
        history_messages = normalized_messages[:-1]
        latest_text = self._render_message_content(latest_message.get("content")) or "..."

        if not history_messages:
            return latest_text

        rendered_history = self._render_history_messages(history_messages)
        history_text = "\n\n".join(rendered_history)

        return (
            "以下是此前对话历史，请你基于这些上下文继续回答用户的最新问题。\n\n"
            f"【历史对话】\n{history_text}\n\n"
            f"【用户最新消息】\n{latest_text}"
        )

    def _history_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized_messages = self._sanitize_messages(messages)
        if len(normalized_messages) <= 1:
            return []
        return normalized_messages[:-1]

    def _build_history_key(self, model: str, messages: List[Dict[str, Any]]) -> str:
        payload = {
            "model": model,
            "messages": messages,
        }
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _sanitize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._sanitize_message(message) for message in messages if isinstance(message, dict)]

    def _sanitize_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {
            "role": message.get("role", "user"),
            "content": self._sanitize_content(message.get("content")),
        }

        name = message.get("name")
        if isinstance(name, str) and name.strip():
            sanitized["name"] = name.strip()

        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            sanitized["tool_calls"] = self._sanitize_content(tool_calls)

        tool_call_id = message.get("tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id.strip():
            sanitized["tool_call_id"] = tool_call_id.strip()

        return sanitized

    def _sanitize_content(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: self._sanitize_content(sub_value)
                for key, sub_value in sorted(value.items())
                if sub_value is not None
            }
        if isinstance(value, list):
            return [self._sanitize_content(item) for item in value]
        if value is None:
            return ""
        return value

    def _render_history_messages(self, history_messages: List[Dict[str, Any]]) -> List[str]:
        rendered_messages: List[str] = []
        used_chars = 0

        for message in reversed(history_messages):
            segment = self._render_message(message)
            if not segment:
                continue

            segment_len = len(segment)
            if rendered_messages and used_chars + segment_len > BOOTSTRAP_HISTORY_MAX_CHARS:
                break

            rendered_messages.append(segment)
            used_chars += segment_len

        rendered_messages.reverse()

        omitted = len(history_messages) - len(rendered_messages)
        if omitted > 0:
            rendered_messages.insert(0, f"[已折叠更早的 {omitted} 条历史消息]")

        return rendered_messages or ["[无可用历史消息]"]

    def _render_message(self, message: Dict[str, Any]) -> str:
        role_map = {
            "system": "系统",
            "user": "用户",
            "assistant": "助手",
            "tool": "工具",
        }
        role_label = role_map.get(message.get("role", "user"), message.get("role", "user"))
        content = self._render_message_content(message.get("content"))
        return f"{role_label}：{content or '[空内容]'}"

    def _render_message_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        text = item.get("text")
                        if isinstance(text, str) and text.strip():
                            parts.append(text.strip())
                        continue

                    if item_type == "image_url":
                        image_url = item.get("image_url", {}).get("url") if isinstance(item.get("image_url"), dict) else None
                        if isinstance(image_url, str) and image_url.strip():
                            parts.append(f"[图片] {image_url.strip()}")
                        else:
                            parts.append("[图片]")
                        continue

                parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))

            return "\n".join(part for part in parts if part).strip()

        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False, sort_keys=True)

        return str(content).strip() if content is not None else ""

    def _retention_seconds(self) -> int:
        try:
            days = int(os.getenv("CONVERSATION_STATE_RETENTION_DAYS", str(DEFAULT_RETENTION_DAYS)))
        except ValueError:
            days = DEFAULT_RETENTION_DAYS
        return max(days, 0) * 24 * 60 * 60

    def _cleanup_locked(self) -> bool:
        cutoff = time.time() - self._retention_seconds()
        expired_keys = [
            key for key, session in self._history_to_session.items()
            if session.updated_at < cutoff
        ]
        if not expired_keys:
            return False

        for key in expired_keys:
            session = self._history_to_session.pop(key, None)
            if not session:
                continue
            keys = self._chat_to_keys.get(session.chat_id)
            if keys:
                keys.discard(key)
                if not keys:
                    self._chat_to_keys.pop(session.chat_id, None)
        return True

    def _trim_locked(self) -> bool:
        overflow = len(self._history_to_session) - MAX_CACHE_ENTRIES
        if overflow <= 0:
            return False

        oldest_items = sorted(
            self._history_to_session.items(),
            key=lambda item: item[1].updated_at,
        )[:overflow]

        for key, session in oldest_items:
            self._history_to_session.pop(key, None)
            keys = self._chat_to_keys.get(session.chat_id)
            if keys:
                keys.discard(key)
                if not keys:
                    self._chat_to_keys.pop(session.chat_id, None)
        return True

    def _load_from_disk(self) -> None:
        with self._lock:
            if not self._state_file_path.exists():
                self._state_file_path.parent.mkdir(parents=True, exist_ok=True)
                self._persist_locked()
                return

            try:
                payload = json.loads(self._state_file_path.read_text(encoding="utf-8"))
            except Exception as exc:
                error_log("[CONVERSATION] 加载持久化状态失败", error=str(exc))
                self._history_to_session = {}
                self._chat_to_keys = {}
                return

            stored_sessions = payload.get("history_to_session", {}) if isinstance(payload, dict) else {}
            self._history_to_session = {}
            self._chat_to_keys = {}

            for history_key, raw_session in stored_sessions.items():
                if not isinstance(history_key, str) or not isinstance(raw_session, dict):
                    continue

                chat_id = raw_session.get("chat_id")
                token = raw_session.get("token")
                last_request_id = raw_session.get("last_request_id")
                model = raw_session.get("model")
                updated_at = raw_session.get("updated_at", time.time())

                if not all(isinstance(value, str) and value.strip() for value in [chat_id, token, last_request_id, model]):
                    continue

                chat_id_str = str(chat_id)
                token_str = str(token)
                last_request_id_str = str(last_request_id)
                model_str = str(model)

                try:
                    updated_at_value = float(updated_at)
                except (TypeError, ValueError):
                    continue

                session = ConversationSession(
                    chat_id=chat_id_str,
                    token=token_str,
                    last_request_id=last_request_id_str,
                    model=model_str,
                    updated_at=updated_at_value,
                )
                self._history_to_session[history_key] = session
                self._chat_to_keys.setdefault(chat_id_str, set()).add(history_key)

            if self._cleanup_locked() or self._trim_locked():
                self._persist_locked()

    def _persist_locked(self) -> None:
        try:
            self._state_file_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "history_to_session": {
                    history_key: {
                        "chat_id": session.chat_id,
                        "token": session.token,
                        "last_request_id": session.last_request_id,
                        "model": session.model,
                        "updated_at": session.updated_at,
                    }
                    for history_key, session in self._history_to_session.items()
                }
            }

            temp_path = self._state_file_path.with_suffix(".json.tmp")
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                encoding="utf-8",
            )
            temp_path.replace(self._state_file_path)
        except Exception as exc:
            error_log("[CONVERSATION] 持久化状态失败", error=str(exc))


conversation_state = ConversationState()
