"""규칙 기반 대응안 후보를 LLM으로 다듬어 엔지니어용 실행 지침으로 변환한다."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from agents.schemas.alert import BottleneckAlert
from agents.schemas.cause import CauseReport
from agents.schemas.solution import SolutionCandidate

load_dotenv(Path(__file__).parent.parent.parent / ".env")

_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _build_prompt(
    alert: BottleneckAlert,
    cause_report: CauseReport,
    candidates: list[SolutionCandidate],
) -> str:
    cause_summary = cause_report.cause_summary
    forecast_text = ""
    if cause_report.sim_forecast:
        wip = cause_report.sim_forecast.kpi_delta.get("wip")
        wr = cause_report.sim_forecast.kpi_delta.get("wait_ratio")
        if wip:
            forecast_text = f"2시간 후 WIP {wip.now}→{wip.future} ({wip.pct_change:+.1f}%)"
        if wr:
            forecast_text += f", wait_ratio {wr.now}→{wr.future} ({wr.pct_change:+.1f}%)"

    cand_text = "\n".join(
        f"[대응안 {c.rank}] {c.name}\n"
        f"  파라미터: {c.params.model_dump(exclude_none=True)}\n"
        f"  기대 효과: {c.expected_effect}\n"
        f"  근거: {c.rationale}"
        for c in candidates
    )

    return f"""당신은 반도체 FAB 운영 전문가입니다. 반드시 한국어로만 답변하세요.

병목 공정: {alert.toolgroup}  심각도: {alert.severity.value}  확률: {alert.probability:.1%}
원인 요약: {cause_summary}
{f"시뮬 예측: {forecast_text}" if forecast_text else ""}

아래 대응안 후보들을 검토하고, 각 대응안에 대해:
1. 구체적인 실행 방법 (어떤 파라미터를 얼마나 바꿀 것인가)
2. 예상 효과와 주의사항
을 1~2문장으로 정리해주세요.

{cand_text}

각 대응안 번호를 유지하며 답변:"""


def refine_candidates(
    alert: BottleneckAlert,
    cause_report: CauseReport,
    candidates: list[SolutionCandidate],
) -> list[SolutionCandidate]:
    """LLM으로 대응안 설명 보강. 실패 시 원본 반환."""
    if not candidates:
        return candidates
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    if not hf_token or hf_token.startswith("your_"):
        return candidates

    try:
        from huggingface_hub import InferenceClient

        prompt = _build_prompt(alert, cause_report, candidates)
        client = InferenceClient(token=hf_token)
        response = client.chat_completion(
            model=_HF_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.2,
        )
        llm_text = response.choices[0].message.content.strip()

        # LLM 전체 응답을 첫 번째 대응안의 expected_effect에 통합
        # (구조화 파싱보다 전체 설명으로 제공하는 게 실용적)
        enriched = candidates[0].model_copy(update={"expected_effect": llm_text})
        return [enriched] + candidates[1:]

    except Exception as e:
        print(f"  [Solution LLM 폴백] {type(e).__name__} — 규칙 기반 유지")
        return candidates
