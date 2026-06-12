"""Agent Task용 LLM 플러밍 — 모델 파라미터, LLM 사용 여부, ChatOpenAI 인스턴스.

도구 바인딩/루프/structured 출력은 loop.py가 담당하고, 여기서는 모델 설정만 공유한다."""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agents.agent_task.helpers import num, to_int
from agents.agent_task.schemas import AgentTaskAgentRequest

DEFAULT_MODEL = "gpt-5.4-mini"
_ROOT = Path(__file__).parent.parent.parent

load_dotenv(_ROOT / ".env")


def llm_tuning_kwargs(model: str, temperature: float) -> dict[str, Any]:
    """모델별 생성 파라미터. gpt-5 계열(reasoning 모델)은 temperature 미지원이고,
    chat completions에서는 함수 도구+reasoning_effort 조합을 거부(400)하므로
    Responses API로 전환해 reasoning effort를 낮춘다(지연↓). 그 외 모델은 기존 temperature 유지."""
    if model.startswith("gpt-5"):
        return {
            "use_responses_api": True,
            "reasoning": {"effort": os.getenv("OPENAI_REASONING_EFFORT", "low")},
        }
    return {"temperature": temperature}


def should_use_llm(req: AgentTaskAgentRequest) -> bool:
    """LLM(agentic) 경로를 쓸지 여부. 기본은 키가 있으면 ON(에이전트가 기본).
    params.useLlm 또는 env AGENT_TASK_USE_LLM=false로 끌 수 있다."""
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    explicit = req.params.get("useLlm")
    if isinstance(explicit, bool):
        return explicit and has_key
    if isinstance(explicit, str):
        return explicit.lower() == "true" and has_key
    override = os.getenv("AGENT_TASK_USE_LLM", os.getenv("USER_INVOKED_AGENT_USE_LLM", "")).lower()
    if override in ("false", "0", "off", "no"):
        return False
    return has_key


def llm_model_name(req: AgentTaskAgentRequest) -> str:
    requested = req.params.get("llmModel")
    if isinstance(requested, str) and requested.strip():
        return requested.strip()
    return os.getenv("AGENT_TASK_MODEL", os.getenv("USER_INVOKED_AGENT_MODEL", DEFAULT_MODEL))


def make_chat_llm(req: AgentTaskAgentRequest) -> ChatOpenAI:
    """Agent Task용 ChatOpenAI 인스턴스(도구 바인딩/structured 출력 공용).

    gpt-5 계열은 reasoning 토큰도 max_completion_tokens 한도에 포함되므로 여유를 둔다."""
    model = llm_model_name(req)
    return ChatOpenAI(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=num(os.getenv("AGENT_TASK_TIMEOUT_SEC", os.getenv("USER_INVOKED_AGENT_TIMEOUT_SEC")), 30.0),
        max_retries=to_int(os.getenv("AGENT_TASK_MAX_RETRIES", os.getenv("USER_INVOKED_AGENT_MAX_RETRIES")), 1),
        max_completion_tokens=to_int(os.getenv("AGENT_TASK_MAX_TOKENS", os.getenv("USER_INVOKED_AGENT_MAX_TOKENS")), 2000),
        **llm_tuning_kwargs(
            model,
            num(os.getenv("AGENT_TASK_TEMPERATURE", os.getenv("USER_INVOKED_AGENT_TEMPERATURE")), 0.2),
        ),
    )
