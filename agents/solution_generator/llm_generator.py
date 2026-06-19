"""규칙 엔진이 확정한 파라미터를 받아 expected_effect / rationale 텍스트 생성."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agents import config
from agents.logger import get_logger
from agents.prompt_store import get_active_prompt
from agents.schemas.alert import BottleneckAlert
from agents.schemas.cause import CauseReport
from agents.token_tracker import record as _record_tokens

_log = get_logger(__name__)

load_dotenv(Path(__file__).parent.parent.parent / ".env")

_MAX_TOKENS = 800
_PLAN_LABELS = {"conservative": "보수안", "standard": "표준안", "aggressive": "강화안"}

# ── 프롬프트 ──────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """당신은 반도체 FAB 운영 전문가로, 확정된 Lot Release 파라미터 조정안의
기대 효과(expected_effect)와 선택 근거(rationale) 텍스트를 생성합니다.
반드시 한국어로 답변하며, 아래 JSON 형식으로만 출력합니다.

{
  "conservative": {"expected_effect": "...", "rationale": "..."},
  "standard":     {"expected_effect": "...", "rationale": "..."},
  "aggressive":   {"expected_effect": "...", "rationale": "..."}
}

규칙:
- expected_effect: 파라미터 조정이 WIP / wait_ratio / Cycle Time에 미치는 기대 효과 (1~2문장)
- rationale: 해당 파라미터 조합을 선택한 근거 (1~2문장)
- 파라미터 수치나 종류를 변경하거나 다른 필드를 추가하지 않는다."""


def _build_system_prompt() -> str:
    return get_active_prompt("ACTION_PLAN_GEN", _SYSTEM_PROMPT)


def _build_user_prompt(
    alert: BottleneckAlert,
    cause_report: CauseReport,
    candidates: dict,
) -> str:
    shap_lines = "\n".join(
        f"  - {sf.feature}: SHAP={sf.shap_value:+.4f}, 현재값={sf.kpi_value:.4f}"
        for sf in cause_report.shap_top[:5]
    )
    param_lines = []
    for lv, label in _PLAN_LABELS.items():
        p = candidates[lv]
        pct = p["release_interval_delta_pct"] or 0.0
        pct_str = f"+{pct:.1f}%" if pct > 0 else "유지"
        pri = p["priority_direction"] or "변경 없음"
        sh = "활성화" if p["superhotlot_enable"] else "비활성화"
        param_lines.append(f"  {label}: RELEASE_INTERVAL {pct_str}, PRIORITY {pri}, SUPERHOTLOT {sh}")

    return f"""[병목 정보]
- TG: {alert.toolgroup}
- severity: {alert.severity.value}
- 병목 확률: {alert.probability:.1%}
- 위험 lot 수: {alert.impact.at_risk_lots:.0f}개

[SHAP 상위 피처]
{shap_lines if shap_lines else "  (없음)"}

원인 요약: {cause_report.cause_summary[:200]}

[확정된 파라미터 조합]
{chr(10).join(param_lines)}

위 파라미터 조합에 대한 expected_effect와 rationale을 JSON으로 생성하십시오."""


# ── fallback 템플릿 ───────────────────────────────────────────────────────────

def _safe_fallback(candidates: dict, cause_report: CauseReport) -> dict[str, dict[str, str]]:
    dominant = candidates["conservative"]["target_kpi"]
    result = {}
    for lv, label in _PLAN_LABELS.items():
        p = candidates[lv]
        pct = p["release_interval_delta_pct"] or 0.0
        effect = (
            f"RELEASE_INTERVAL을 {pct:.0f}% 증가하여 TG 투입량을 억제합니다."
            if pct > 0
            else f"{dominant} 악화에 대응하여 투입 우선순위를 조정합니다."
        )
        result[lv] = {
            "expected_effect": effect,
            "rationale": f"[LLM_FALLBACK] {dominant} 악화 감지. {label} 조정안 적용.",
        }
    return result


# ── 메인 API ─────────────────────────────────────────────────────────────────

def generate_texts(
    alert: BottleneckAlert,
    cause_report: CauseReport,
    candidates: dict,
) -> dict[str, dict[str, str]]:
    """보수/표준/강화 각각의 expected_effect + rationale 생성.

    반환:
        {"conservative": {"expected_effect": ..., "rationale": ...}, ...}

    LLM 실패 시 템플릿 fallback.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        _log.warning("[llm_generator] API 키 없음 — fallback 사용")
        return _safe_fallback(candidates, cause_report)

    try:
        from openai import APIError, OpenAI, RateLimitError

        client = OpenAI(api_key=api_key)
        user_prompt = _build_user_prompt(alert, cause_report, candidates)

        @retry(
            retry=retry_if_exception_type((RateLimitError, APIError)),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        def _call():
            return client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": _build_system_prompt()},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
            )

        response = _call()
        if response.usage:
            _record_tokens(
                "llm_generator",
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

        raw = response.choices[0].message.content or ""
        parsed = json.loads(raw)

        if not all(lv in parsed for lv in ("conservative", "standard", "aggressive")):
            raise ValueError(f"응답 키 누락: {list(parsed.keys())}")
        for lv in ("conservative", "standard", "aggressive"):
            if not isinstance(parsed[lv], dict):
                raise ValueError(f"{lv} 값이 dict가 아님")
            parsed[lv].setdefault("expected_effect", "")
            parsed[lv].setdefault("rationale", "")

        return {lv: parsed[lv] for lv in ("conservative", "standard", "aggressive")}

    except json.JSONDecodeError as e:
        _log.warning(f"[llm_generator] JSON 파싱 실패({e}) — fallback 사용")
    except Exception as e:
        _log.warning(f"[llm_generator] {type(e).__name__}: {e} — fallback 사용")

    return _safe_fallback(candidates, cause_report)
