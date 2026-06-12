"""챗봇 라우팅 회귀 평가셋 — 프롬프트/도구 변경 시 라우팅이 깨지지 않는지 점검.

실행: cd AI-Agent && .venv/bin/python scripts/chat_eval.py
판정: 응답의 toolsUsed(호출 도구) / ui.type / sources 유무 / 금지 패턴(약속형 답변)으로 라우팅을 검증.
비용: 문항당 LLM 1~3콜(gpt-5.4-mini) ≈ $0.005~0.01 → 전체(~32문항) 1회 ≈ $0.2~0.3 수준.
"""

import json
import os
import re
import sys
import urllib.request

BASE = os.getenv("CHAT_EVAL_BASE", "http://localhost:8000")
FAB_ID = os.getenv("CHAT_EVAL_FAB_ID", "9db3c0e4-612a-4d84-b6bf-fafed1f44d03")
HEADERS = {
    "Content-Type": "application/json",
    "X-Internal-Token": os.getenv("INTERNAL_API_TOKEN", "fabbear-internal-token-2024"),
    "X-User-Id": "00000000-0000-0000-0000-000000000001",
    "X-User-Role": "USER",
    "X-Factory-Id": FAB_ID,
    "X-Request-Id": "00000000-0000-0000-0000-000000000002",
}

# (질문, 기대 ui.type 또는 None, RAG 출처 기대 여부, 기대 호출 도구 집합 또는 None(미검증), 비고)
# want_tools는 부분집합 판정(기대 도구가 모두 호출됐는지) — 추가 도구 호출은 허용.
CASES: list[tuple[str, str | None, bool, set[str] | None, str]] = [
    # ── 현황/추세/적체 (기존) ──
    ("지금 라인 전체 괜찮아?", "status", False, {"get_fab_status"}, "현황"),
    ("현재 전체 공장 상황 브리핑 해줘", "status", False, {"get_fab_status"}, "브리핑"),
    # '변했어'는 추세(get_kpi_trend)·비교(compare_periods) 둘 다 합리적 → 도구/카드 미검증(모호 케이스).
    ("오후 들어 가동률 어떻게 변했어?", None, False, None, "추세/비교 모호 — 빈 답변·약속형만 검증"),
    ("최근 6시간 WIP 변화 알려줘", "trend", False, {"get_kpi_trend"}, "추세"),
    ("내일 Litho WIP 예측해줘", "trend", False, {"get_kpi_trend"}, "미지원 forecast → 최근 추세 기반 안내"),
    ("전체 구역 트렌드 보여줘", "trend", False, {"get_kpi_trend"}, "전체 추세(데모 실패건)"),
    ("구역별 트렌드 보여줘", "trend", False, {"get_kpi_trend"}, "STT 보정 후 구역별 트렌드"),
    # 적체 위치는 get_lot_status·get_top_toolgroups(wip) 둘 다 정답 → 카드(lot)만 검증.
    ("어디가 제일 막혀있어?", "lot", False, None, "적체→lot 카드(도구는 둘 다 허용)"),
    ("대기 Lot 많은 구역 알려줘", "lot", False, {"get_lot_status"}, "대기"),
    ("현재 전체 구역중 wip 가 가장 높은 구역과 tg, tool?", "lot", False, {"get_lot_status"}, "구역/TG/툴 계층형 WIP top"),
    ("대기 lot이 가장 많은 곳의 6시간 가동률 추이", "trend", False, {"get_top_toolgroups", "get_kpi_trend"}, "대상탐색+추세"),
    # ── 툴그룹(TG) 순위/현황 ──
    ("가동률이 가장 높은 툴그룹 톱5 알려줘", "lot", False, {"get_top_toolgroups"}, "TG 순위(데모 실패건)"),
    ("WIP 가장 많은 툴그룹 어디야?", "lot", False, {"get_top_toolgroups"}, "TG WIP 순위"),
    ("내가 지금 봐야 할 위험한 툴그룹 알려줘", "lot", False, {"get_top_toolgroups"}, "priority TG(데모 실패건)"),
    ("내가 지금 봐야 할 툴그룹 알려줘", "lot", False, {"get_top_toolgroups"}, "STT 보정 후 priority TG"),
    ("Q-time 긴 툴그룹 순위 보여줘", "lot", False, {"get_top_toolgroups"}, "TG Q-time 순위"),
    ("Def_Met 구역 툴그룹별 추세 보여줘", "trend", False, {"get_kpi_trend"}, "TG 단위 추세(group_by=tg)"),
    # ── 개별 툴(설비) ──
    ("Dielectric_FE_31 안의 설비별 상태 알려줘", "lot", False, {"get_tool_status"}, "TG 내 툴 현황(데모 실패건)"),
    ("Litho 구역 장비별 상태 보여줘", "lot", False, {"get_tool_status"}, "구역 내 설비"),
    ("Litho 구역에 있는 기기 정보 가져와", "lot", False, {"get_tool_status"}, "STT 보정 후 기기 정보"),
    ("Dielectric_FE_31#2 설비 최근 6시간 추이 보여줘", None, False, {"get_tool_activity"}, "툴 활동(데이터 없으면 카드 생략 허용)"),
    ("Dielectric_FE_31#2 요즘 괜찮았어?", None, False, {"get_tool_activity"}, "툴 활동 구어체"),
    # ── 병목 케이스: 목록 vs 상세 ──
    ("오늘 병목 케이스 있었어?", None, False, {"search_bottleneck_cases"}, "케이스 목록(DB 비면 카드 없음 허용)"),
    ("최근 HIGH 등급 병목 케이스 이력 보여줘", None, False, {"search_bottleneck_cases"}, "케이스 목록 필터"),
    ("가장 최근 병목 케이스 원인이 뭐였어?", None, False, {"get_case_detail"}, "케이스 상세-원인(빈 DB면 '없음' 답변도 PASS)"),
    ("최근 케이스 대응안이랑 예상 효과 알려줘", None, False, {"get_case_detail"}, "케이스 상세-대응안"),
    ("최근 병목 케이스 리포트 요약해줘", None, False, {"get_case_detail"}, "케이스 상세-리포트"),
    # ── 기간 비교 ──
    ("아까보다 나아졌어?", None, False, {"compare_periods"}, "기간비교"),
    ("오전이랑 비교해서 Q-time 어때?", None, False, {"compare_periods"}, "기간비교 지표 지정"),
    # ── 개념(RAG) ──
    ("Q-time이 뭐야?", None, True, {"search_knowledge"}, "개념→RAG 출처"),
    ("Reentrant 공정이 뭐야?", None, True, {"search_knowledge"}, "개념→RAG 출처"),
    ("X-factor가 뭐야?", None, True, {"search_knowledge"}, "개념→RAG 출처"),
    # ── 복합 ──
    ("Q-time이 뭐고 지금 Q-time 긴 데는 어디야?", "lot", True, {"search_knowledge"}, "복합(개념+현황)"),
    ("OEE가 뭐고 가동률 낮은 툴그룹은 어디야?", "lot", True, {"search_knowledge", "get_top_toolgroups"}, "복합(개념+TG순위)"),
    # ── 잡담 ──
    ("고마워 수고했어", None, False, None, "잡담(도구·카드 없음)"),
    ("안녕", None, False, None, "잡담 인사"),
]

PROMISE_PATTERN = re.compile(r"(잠시만\s*기다|확인해\s*보겠|불러오겠|조회해\s*보겠)")
ARITHMETIC_LOOP_PATTERN = re.compile(r"(?:\d+\s*\+){8,}\d+")
# 잡담에 카드가 붙으면 실패로 간주할 케이스 인덱스
NO_CARD_REQUIRED = {"고마워 수고했어", "안녕"}


# 운영 경로에선 Spring이 liveStatus를 주입 — 평가셋은 자가완결을 위해 고정 fixture 사용.
LIVE_STATUS_FIXTURE = (
    "[현재 FAB 실시간 현황 · 기준 2020-01-01T00:00:00Z]\n"
    "전체: 가동률 21.8%, WIP 572 Lot, 설비 가동 440/대기 1071/셋업 0/비가동 51, 가용률 96.5%\n"
    "구역별(WIP 많은 순):\n"
    "- Litho: WIP 107, 평균가동률 25.7%, 가용률 94.5%\n"
    "- Diffusion: WIP 31, 평균가동률 30.7%, 가용률 100.0%"
)


def call(message: str) -> dict:
    body = json.dumps({"message": message, "fabId": FAB_ID, "liveStatus": LIVE_STATUS_FIXTURE}).encode()
    req = urllib.request.Request(f"{BASE}/api/chat/message", data=body, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=90) as res:
        return json.load(res)["data"]


def main() -> int:
    passed = 0
    for message, want_ui, want_rag, want_tools, note in CASES:
        try:
            d = call(message)
        except Exception as e:  # noqa: BLE001
            print(f"  [ERR ] {message} — {e}")
            continue
        ui = (d.get("ui") or {}).get("type")
        has_rag = bool(d.get("sources"))
        tools_used = set(d.get("toolsUsed") or [])
        answer = d.get("answer") or ""
        problems = []
        if want_ui is not None and ui != want_ui:
            problems.append(f"ui={ui}(기대 {want_ui})")
        if message in NO_CARD_REQUIRED and ui is not None:
            problems.append(f"잡담에 카드({ui})")
        if want_rag and not has_rag:
            problems.append("RAG 출처 없음")
        if want_tools is not None and not want_tools <= tools_used:
            problems.append(f"tools={sorted(tools_used)}(기대 ⊇ {sorted(want_tools)})")
        if PROMISE_PATTERN.search(answer):
            problems.append("약속형 답변(미완수)")
        if ARITHMETIC_LOOP_PATTERN.search(answer):
            problems.append("긴 산술식/계산 반복")
        if message == "내일 Litho WIP 예측해줘" and ("예측" not in answer or "지원" not in answer):
            problems.append("미지원 forecast 안내 없음")
        if message == "현재 전체 구역중 wip 가 가장 높은 구역과 tg, tool?":
            if re.search(r"가장\s*많은\s*구역은\s*Wet_Etch[\s\S]{0,80}그중[\s\S]{0,80}Dry_Etch", answer):
                problems.append("구역/TG 계층 불일치 표현")
        if not answer.strip():
            problems.append("빈 답변")
        if problems:
            print(f"  [FAIL] {message} — {', '.join(problems)}  ({note})")
        else:
            passed += 1
            print(f"  [PASS] {message}  (ui={ui}, rag={'Y' if has_rag else 'N'}, tools={sorted(tools_used)})")
    total = len(CASES)
    print(f"\n결과: {passed}/{total} 통과")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
