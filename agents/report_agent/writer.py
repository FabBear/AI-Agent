from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agents import config

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# ── LLM 설정 ──────────────────────────────────────────────────────────────────
_llm: Optional[ChatOpenAI] = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY가 설정되지 않았습니다.\n"
                "  프로젝트 루트의 .env 파일에 OPENAI_API_KEY를 추가하세요."
            )
        _llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=api_key,
            temperature=config.LLM_TEMPERATURE,
            max_completion_tokens=3000,
        )
    return _llm


def _llm_write(system_prompt: str, user_prompt: str) -> str:
    response = _get_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return response.content.strip()


# ── State 정의 ────────────────────────────────────────────────────────────────

class ReportState(TypedDict):
    # ── 입력 (앞 Agent로부터 받는 데이터) ──
    process_name: str
    severity: str
    detected_at: str

    bottleneck_info: dict
    fab_kpi: dict
    bottleneck_trend: list
    tool_status: list
    affected_lots_detail: list
    diffusion_analysis: dict
    shap_analysis: dict
    feature_trend: list
    cause_analysis: dict
    action_effects: list
    recommendation: dict
    approval_info: dict

    # ── 생성된 섹션 ──
    section_header: str
    section_review: str
    section_summary: str
    section_diffusion: str
    section_cause: str
    section_actions: str

    # ── 최종 출력 ──
    final_report: str
    output_path: str
    json_output_path: str


# ── 시스템 프롬프트 ────────────────────────────────────────────────────────────

_SYS = """[역할]
당신은 반도체 FAB 공정 병목 대응 의사결정 보고서를 작성하는 전문 AI입니다.
병목 감지 · 원인 분석 · 대응안 생성 · 효과 검증 · 비교 분석을 수행한
5개 AI Agent의 결과를 종합하여, 공정 관리자가 즉시 판단하고 행동할 수 있는
최종 보고서를 작성합니다.

[독자]
- 반도체 FAB 공정 관리자, 생산 엔지니어
- WIP · Q-time · CQT · CT · TH · Load Ratio · CR · SHAP · SuperHotLot ·
  Dispatch Rule 등 FAB 공정 용어에 익숙합니다
- 병목 발생 시 빠른 의사결정이 필요하므로 핵심 수치와 근거 중심으로 판단합니다
- 모호한 표현을 신뢰하지 않으며, 데이터에 기반한 명확한 서술을 요구합니다

[도메인 지식 — 이 기준으로 데이터를 해석하세요]
risk_score 등급 (= 대기시간 30 + 부하 30 + 처리량갭 25 + 확산위험 15):
  · 정상   : 25 이하
  · MEDIUM : 26~47  → 주의 필요
  · HIGH   : 47~60  → 비정상 진입, 즉각 모니터링
  · CRITICAL: 60 이상 → 즉시 대응 필요

Load Ratio 해석:
  · < 0.75       : 여유 있음
  · 0.75 ~ 0.90  : 주의
  · 0.90 ~ 1.00  : capacity 여유 낮음
  · > 1.00       : 과부하 / 2차 병목 가능

원인 후보 분류:
  · machine_downtime   : 장비 정지 · 비가동 영향
  · setup_overhead     : setup 전환 부담 증가
  · upstream_overload  : 앞 공정에서 유입량 증가
  · capacity_saturation: 처리 능력 포화
  · queue_buildup      : 일반적인 큐 적체

[작성 원칙]
1. 반드시 한국어로 작성합니다.
   단, 장비명(예: Diffusion_FE_120) · KPI 지표명(WIP · Q-time · CT · TH) ·
   Lot명 · 대응안 종류(REQUEUE_TOOL 등)는 원문 그대로 유지합니다.
2. 제공된 수치는 반드시 그대로 사용합니다.
   반올림 · 단위 변환 · 재계산 금지. (45.2% → "약 45%" 금지, 180분 → "3시간" 금지)
3. 데이터에 없는 내용은 절대 추측하거나 창작하지 않습니다.
   없는 항목은 "-"로 표시합니다.
4. 객관적 · 사실 기반 문장으로 작성합니다.
   "아마도" · "가능성이 있습니다" 같은 모호한 표현은 금지합니다.
   단, 시뮬레이션 기반 예측값을 서술할 때는 "시뮬레이션 기준" 또는 "예측값"임을 명시합니다.
5. 능동태와 단문을 사용합니다. 한 문장은 최대 두 줄을 넘지 않습니다.
6. 서론 · 결론 · 부연 설명은 추가하지 않습니다.
7. 문체는 "~됨", "~함", "~임"으로 통일합니다. (~습니다 · ~이다 · ~합니다 금지)

[출력 규칙]
- Markdown 형식만 출력합니다.
- 출력 앞뒤에 "다음은 보고서입니다" · "이상으로 마칩니다" 등의 문장을 붙이지 않습니다.
- ```markdown 코드블록으로 감싸지 않습니다.
- 지시된 섹션 구조 외에 임의로 섹션을 추가하거나 삭제하지 않습니다.
- 표의 컬럼명과 순서는 지시된 형식 그대로 유지합니다.

[금지 사항]
- 데이터에 없는 원인 · 수치 · 대응안 창작 금지
- 수치 변경 · 단위 변환 금지
- 영어와 한국어 혼용 금지 (지정된 FAB 용어 제외)
- 지시된 형식 이외의 내용 추가 금지"""


# ── LLM Node 함수들 ───────────────────────────────────────────────────────────

def write_summary(state: ReportState) -> dict:
    """1. 요약 — 3줄 핵심 요약 + 핵심 지표"""
    bi = state.get("bottleneck_info") or {}
    section = _llm_write(
        _SYS,
        f"""아래 데이터를 바탕으로 FAB 병목 보고서의 '요약' 섹션을 작성하세요.

공정명: {state['process_name']}
심각도: {state['severity']}
데이터: {json.dumps(bi, ensure_ascii=False)}

출력 형식 (이 형식 그대로):

## 1. 요약

> **[핵심 요약 1줄: 병목 상황 + risk_score 언급]**
> **[핵심 요약 2줄: 생산 영향 + load_ratio 언급]**
> **[핵심 요약 3줄: 긴급 조치 필요성]**

| 지표 | 값 |
|------|----|
| 심각도 | {state['severity']} |
| risk_score | {bi.get('risk_score', '-')} |
| load_ratio (wait_ratio) | {bi.get('load_ratio', '-')} |
| 지연 주문 수 | {bi.get('delayed_orders', '-')}건 |
| 현재 평균 대기시간 | {bi.get('avg_queue_time_min', '-')}분 |
| 최대 대기시간 | {bi.get('peak_q_time_min', '-')}분 |
| WIP | {bi.get('wip_count', '-')}개 |
| 가동률 | {bi.get('utilization_pct', '-')}% |
| 가용 호기 비율 | {bi.get('available_tool_ratio', '-')} |

---""",
    )
    return {"section_summary": section}


def write_diffusion(state: ReportState) -> dict:
    """2. 확산 영향 분석"""
    da = state.get("diffusion_analysis") or {}
    fab = state.get("fab_kpi", {})
    trend = state.get("bottleneck_trend", [])
    section = _llm_write(
        _SYS,
        f"""아래 데이터를 바탕으로 '확산 영향 분석' 섹션을 작성하세요.

확산 데이터: {json.dumps(da, ensure_ascii=False)}
FAB 전체 KPI: {json.dumps(fab, ensure_ascii=False)}
병목 추이 데이터: {json.dumps(trend, ensure_ascii=False)}

출력 형식 (이 형식 그대로):

## 2. 확산 영향 분석

**탐지시각**: {state['detected_at']}

### ① 병목 발생 여부 및 위치
[병목 발생 여부와 정확한 위치를 한 문장으로 서술]
- **확산 경로**: [diffusion_path 항목을 → 화살표로 연결하여 한 줄로 표기]

### ② 확산 현황 (공정별 가동률 현황)

| 공정 | 가동률(%) | wait_ratio | WIP | 상태 |
|------|-----------|-----------|-----|------|
[affected_processes 데이터를 행으로 채울 것. utilization_pct·wait_ratio·wip 값 사용. 값 없으면 -]

### ③ 위험도 및 전 라인 정지 예상 시간
- **위험도**: {da.get('risk_level', '-')}
- **전 라인 정지 예상**: {da.get('line_stop_expected_min', '-')}분 이내
- [위험도에 대한 한 줄 해석]
- **병목 추이**: [bottleneck_trend 데이터를 보고 Q-time과 WIP가 심화/완화/유지 중인지 한 문장으로 요약]

### ④ 병목 공정 KPI 현황

| 지표 | 값 |
|------|----|
| 전체 WIP | {fab.get('wip_total', '-')}개 |
| 평균 가동률 | {fab.get('utilization_avg_pct', '-')}% |
| 평균 대기시간 | {fab.get('q_time_min', '-')}분 |
| wait_ratio | {fab.get('wait_ratio', '-')} |

### ⑤ Forward Simulation ({da.get('forward_simulation', {}).get('horizon_min', 120)}분 후 예측)

| Tool Group | q_time (미래) | wait_ratio (미래) | WIP (미래) | 병목 예측 |
|------------|-------------|-----------------|----------|----------|
[forward_simulation.results 데이터를 행으로 채울 것. q_time_future·wait_ratio_future·wip_future 값 사용. y_bottleneck=1이면 ✅ 병목, 0이면 ✅ 정상. forward_simulation이 비어 있으면 "| - | - | - | - | - |" 한 행만 출력]

---""",
    )
    return {"section_diffusion": section}


def write_cause(state: ReportState) -> dict:
    """3. 원인 분석 TOP 3"""
    ca = state.get("cause_analysis") or {}
    ft = state.get("feature_trend", [])
    shap = state.get("shap_analysis", {})
    section = _llm_write(
        _SYS,
        f"""아래 데이터를 바탕으로 '원인 분석 TOP 3' 섹션을 작성하세요.

원인 요약: {ca.get("summary", "")}
SHAP 기여도 순위: {json.dumps(ca.get("shap_top", []), ensure_ascii=False)}
컨센서스 판정: {json.dumps(ca.get("consensus"), ensure_ascii=False)}
feature 트렌드 데이터: {json.dumps(ft, ensure_ascii=False)}
ML SHAP 분석: {json.dumps(shap, ensure_ascii=False)}

출력 형식 (이 형식 그대로):

## 3. 원인 분석 TOP 3

| 순위 | 원인(feature) | 기여도(%) | 현재값 |
|------|--------------|----------|--------|
[shap_top의 rank·feature·contribution_pct·kpi_value를 행으로 채울 것]

### 판정 요약
[consensus.summary 내용을 2~3문장으로 정리. confidence_level 포함]

### ML 모델 SHAP 분석 (Top 3 Feature)
> 모델: {shap.get('model', '-')}

| 피처명 | 현재값 | 기여도(%) | 방향 |
|--------|--------|----------|------|
[shap_analysis.top_features를 행으로 채울 것. feature·value·share_abs_pct·direction 값 사용]

### feature 트렌드 (탐지 전 4시간)

| 시각 | q_time_min | wait_ratio | wip | max_util |
|------|-----------|-----------|-----|---------|
[feature_trend 데이터를 행으로 채울 것. 값 없는 feature는 -]

---""",
    )
    return {"section_cause": section}


def write_actions(state: ReportState) -> dict:
    """4. 승인된 대응안 — 효과 요약 + A/B/C 비교표"""
    effects = state.get("action_effects") or []
    rec = state.get("recommendation") or {}
    ai = state.get("approval_info") or {}
    is_rejected = ai.get("status") == "반려"

    if is_rejected:
        rejection_reason = ai.get("rejection_reason", "-")
        section = _llm_write(
            _SYS,
            f"""아래 데이터를 바탕으로 '승인된 대응안' 섹션을 작성하세요. 반려된 상황입니다.

반려 사유: {rejection_reason}
전체 대응안 비교: {json.dumps(effects, ensure_ascii=False)}

출력 형식 (이 형식 그대로):

## 4. 승인된 대응안

### ① 승인된 대응안 예상 효과
승인된 대응안 없음. (반려됨)
- **반려 사유**: {rejection_reason}

### ② 대응안 A / B / C 비교

| 대응안 | 종류 | 설명 | 대기시간 변화 | WIP 변화 | 처리량 변화 | 시뮬 신뢰도 |
|--------|------|------|-------------|---------|-----------|------------|
[각 대응안을 행으로 채울 것, ✅ 표시 없음, simulation_confidence는 % 단위로]

---""",
        )
    else:
        action_label = rec.get("action_label", "")
        approved = next(
            (e for e in effects if e.get("label", "").split()[0] == action_label),
            effects[0] if effects else {}
        )
        lots = state.get("affected_lots_detail", [])
        section = _llm_write(
            _SYS,
            f"""아래 데이터를 바탕으로 '승인된 대응안' 섹션을 작성하세요.

승인된 대응안: {json.dumps(approved, ensure_ascii=False)}
선택 이유: {rec.get('reason', '-')}
전체 대응안 비교: {json.dumps(effects, ensure_ascii=False)}
영향 Lot 상세: {json.dumps(lots, ensure_ascii=False)}

출력 형식 (이 형식 그대로):

## 4. 승인된 대응안

### ① 승인된 대응안 예상 효과
- **대응안**: {approved.get('action_kind')} — {approved.get('description')}
- **선택 이유**: {rec.get('reason')}
- **예상 효과 요약**: [kpi_delta를 해석하여 한 문장으로]

### ② 대응안 A / B / C 비교

| 대응안 | 종류 | 설명 | 대기시간 변화 | WIP 변화 | 처리량 변화 | 시뮬 신뢰도 |
|--------|------|------|-------------|---------|-----------|------------|
[각 대응안을 행으로, 승인된 것(label={rec.get('action_label')})에는 ✅ 표시, simulation_confidence는 % 단위로]

---""",
        )
    return {"section_actions": section}
