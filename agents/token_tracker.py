"""OpenAI API 토큰 사용량 추적."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path(__file__).parent.parent / "logs" / "token_usage"


@dataclass
class _Session:
    calls: list[dict] = field(default_factory=list)

    @property
    def prompt_tokens(self) -> int:
        return sum(c["prompt_tokens"] for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c["completion_tokens"] for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_cost_krw(self) -> float:
        # gpt-4o-mini 기준: input $0.15/1M, output $0.60/1M (1USD=1400원)
        cost_usd = (self.prompt_tokens * 0.15 + self.completion_tokens * 0.60) / 1_000_000
        return cost_usd * 1400


_session = _Session()


def record(caller: str, prompt_tokens: int, completion_tokens: int) -> None:
    """LLM 호출 후 토큰 수를 기록한다."""
    _session.calls.append(
        {
            "caller": caller,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "timestamp": datetime.now().isoformat(),
        }
    )


def print_summary() -> None:
    """세션 전체 토큰 사용량을 출력한다."""
    if not _session.calls:
        return
    print("\n" + "─" * 50)
    print("  OpenAI 토큰 사용량 요약")
    print("─" * 50)
    for c in _session.calls:
        print(
            f"  [{c['caller']}]  "
            f"입력 {c['prompt_tokens']:,}  출력 {c['completion_tokens']:,}"
        )
    print("─" * 50)
    print(f"  총 입력:  {_session.prompt_tokens:,} 토큰")
    print(f"  총 출력:  {_session.completion_tokens:,} 토큰")
    print(f"  합계:     {_session.total_tokens:,} 토큰")
    print(f"  예상 비용: ₩{_session.estimated_cost_krw:.1f}")
    print("─" * 50 + "\n")


def save_log() -> None:
    """사용량을 JSON 파일로 저장한다."""
    if not _session.calls:
        return
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    filename = _LOG_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_prompt_tokens": _session.prompt_tokens,
                "total_completion_tokens": _session.completion_tokens,
                "total_tokens": _session.total_tokens,
                "estimated_cost_krw": _session.estimated_cost_krw,
                "calls": _session.calls,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  토큰 로그 저장: {filename}")
