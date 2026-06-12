"""report_agent 입력 정규화 계층.

- compare_result는 경로(Phase 1 all-in-one vs Phase 2 webhook)에 따라
  서로 다른 shape으로 들어온다. 보고서는 이 차이를 모르고 동일하게 읽어야 한다.
- 업스트림이 만든 NaN/Inf는 json.dumps 시 ECMA-262 위반 토큰이 되어
  Spring Jackson / 브라우저 JSON.parse를 깨뜨린다. 직렬화 전에 None으로 치환한다.

원칙: 이 파일은 어떤 키도 새로 만들어내지 않는다 — 업스트림이 준 값만 정규화한다.
"""

from __future__ import annotations

import math
from typing import Any


def normalize_compare_result(compare_result: dict) -> dict:
    """compare_result를 보고서가 읽기 쉬운 형태로 정규화한다.

    두 가지 입력 shape을 모두 흡수한다.

    Phase 1 (compare_agent/node.py:_build_compare_result):
        {toolgroup, result_v2: {action_options, recommendation, decision_meta, ...},
         approval_info, json_output_path}

    Phase 2 (run_phase2.py:_reconstruct_state):
        {toolgroup, recommendation, action_effects, approval_info, json_output_path}
        ※ result_v2 키가 없고, 일부 데이터가 누락된 경우가 있음

    반환 형태(보고서 내부 표준):
        {
          action_options: list,
          recommendation:  dict,
          decision_meta:   dict,
          approval_info:   dict,
          # 추가 컨텍스트(있을 때만)
          meta, current_state, cause, cascade, data_quality
        }
    """
    if not isinstance(compare_result, dict):
        return _empty_normalized()

    result_v2 = compare_result.get("result_v2")

    # Phase 1: result_v2 안에 모든 게 들어있는 케이스
    if isinstance(result_v2, dict) and result_v2:
        return {
            "action_options": list(result_v2.get("action_options") or []),
            "recommendation": dict(result_v2.get("recommendation") or {}),
            "decision_meta":  dict(result_v2.get("decision_meta") or {}),
            "approval_info":  dict(compare_result.get("approval_info") or {}),
            "meta":           dict(result_v2.get("meta") or {}),
            "current_state":  dict(result_v2.get("current_state") or {}),
            "cause":          dict(result_v2.get("cause") or {}),
            "cascade":        dict(result_v2.get("cascade") or {}),
            "data_quality":   dict(result_v2.get("data_quality") or {}),
        }

    # Phase 2: top-level에 평탄화된 케이스 — 가능한 키들을 모두 살핀다
    return {
        "action_options": list(
            compare_result.get("action_options")
            or compare_result.get("action_effects")
            or []
        ),
        "recommendation": dict(compare_result.get("recommendation") or {}),
        "decision_meta":  dict(compare_result.get("decision_meta") or {}),
        "approval_info":  dict(compare_result.get("approval_info") or {}),
        "meta":           dict(compare_result.get("meta") or {}),
        "current_state":  dict(compare_result.get("current_state") or {}),
        "cause":          dict(compare_result.get("cause") or {}),
        "cascade":        dict(compare_result.get("cascade") or {}),
        "data_quality":   dict(compare_result.get("data_quality") or {}),
    }


def _empty_normalized() -> dict:
    return {
        "action_options": [],
        "recommendation": {},
        "decision_meta":  {},
        "approval_info":  {},
        "meta":           {},
        "current_state":  {},
        "cause":          {},
        "cascade":        {},
        "data_quality":   {},
    }


def has_actions(normalized: dict) -> bool:
    """action_options에 baseline(현재상태) 외의 실제 후보가 하나라도 있으면 True."""
    options = normalized.get("action_options") or []
    return any(not opt.get("is_baseline") for opt in options if isinstance(opt, dict))


def pick_approved_option(action_options: list, action_label: str) -> dict:
    """승인된 label에 해당하는 action_option을 찾는다.

    - label 정확히 일치하는 후보가 있으면 그것을 반환.
    - 일치하는 게 없으면 baseline(is_baseline=True)이 아닌 첫 후보를 반환.
    - 후보가 전혀 없거나 모두 baseline이면 {} 반환.

    fallback이 baseline을 잡지 않도록 명시적으로 막는 것이 핵심.
    """
    if not action_options:
        return {}

    if action_label:
        for opt in action_options:
            if isinstance(opt, dict) and opt.get("label") == action_label:
                return dict(opt)

    for opt in action_options:
        if isinstance(opt, dict) and not opt.get("is_baseline"):
            return dict(opt)

    return {}


def sanitize_for_json(obj: Any) -> Any:
    """NaN/Inf/-Inf를 None으로 치환한다 (재귀).

    표준 JSON(ECMA-262)에는 NaN/Inf 토큰이 없다.
    python json.dumps는 기본적으로 NaN을 그대로 직렬화하므로
    Spring Jackson / 브라우저 JSON.parse가 SyntaxError로 깨진다.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj
