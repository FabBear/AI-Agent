"""챗봇 입력 가드레일 — 실행·변경성 명령은 챗봇이 직접 수행하지 않고 담당 화면으로 안내한다.

챗봇은 조회/설명/요약 전용이다. "승인해/실행해/Lot 순서 바꿔" 같은 운영 상태 변경 명령은
LLM·도구로 흘리지 않고 navigation 안내로 차단한다. 단, '승인됐어?/리포트 요약해줘' 같은
조회·질문은 막지 않도록 명령형(…해/…해줘/…돌려)만 매칭한다."""

import re

# (카테고리, 패턴). 명령형 어미(해|해줘|해주세요|줘|하라|돌려|시작해|바꿔|올려|내려|줄여|늘려)를 요구해
# 질문형('승인됐어?', '리포트 요약해줘'는 조회)과 구분한다.
_EXEC_PATTERNS: tuple[tuple[str, str], ...] = (
    ("approval", r"(승인|반려|기각|결재|재가)\s*(해줘|해주세요|해라|해|처리해|진행해)"),
    ("schedule", r"(lot|랏|로트)?\s*(순서|우선순위|투입(량)?|디스패칭|스케줄)\s*\S{0,6}?(바꿔|변경해|조정해|적용해|반영해|확정해|올려|내려|줄여|늘려|재배열해)"),
    ("schedule", r"(핫\s*랏|hot\s*lot|핫랏|priority|우선순위)\s*\S{0,6}?(우선|올려|먼저|승격|적용해|반영해)"),
    ("schedule", r"(hold|홀드)\s*\S{0,6}?(걸어|잡아|적용해|반영해)"),
    ("schedule", r"(release|릴리즈|투입)\s*\S{0,6}?(막아|중지해|제한해|줄여|늘려)"),
    ("schedule", r"(pm|예방정비)\s*\S{0,6}?(잡아|예약해|등록해|확정해)"),
    ("db_write", r"(케이스|리포트|보고서|알림|이력|기록|코멘트|메모)\s*\S{0,6}?(생성해|만들어|저장해|등록해|삭제해|지워|발송해|보내|남겨|추가해|발행해)"),
    ("admin_change", r"(모델|model)\s*\S{0,6}?(재학습|학습|배포|바꿔|변경해|적용해|반영해)"),
    ("admin_change", r"(threshold|임계값|쓰레숄드)\s*\S{0,6}?(바꿔|변경해|조정해|적용해|반영해|올려|내려)"),
    ("agent_run", r"(진단|분석|파이프라인|시뮬|시뮬레이션|보고서|리포트|에이전트)\s*\S{0,4}?(돌려|실행해|시작해|가동해|구동해)"),
    ("agent_run", r"(실행|구동|가동)\s*시작해"),
)


def detect_execution_intent(message: str) -> str | None:
    """실행·변경성 명령이면 카테고리를, 아니면 None을 반환."""
    text = (message or "").lower()
    if not text.strip():
        return None
    for category, pattern in _EXEC_PATTERNS:
        if re.search(pattern, text):
            return category
    return None


_NAV_BY_CATEGORY = {
    "approval": ("승인 화면으로 이동", "/agent/approvals", "HITL 승인/반려는 담당 화면에서만 처리합니다."),
    "schedule": ("스케줄링 화면으로 이동", "/scheduling", "Lot 순서·투입량·우선순위 변경은 스케줄링 영역입니다."),
    "db_write": ("담당 화면으로 이동", "/agent/tasks", "생성/저장/삭제 등 변경 작업은 담당 화면에서 확인 후 처리합니다."),
    "admin_change": ("관리 화면으로 이동", "/admin/thresholds", "모델·임계값 변경은 관리자 화면에서 확인 후 처리합니다."),
    "agent_run": ("Agent 작업 화면으로 이동", "/agent/tasks", "진단/파이프라인 실행은 Agent 작업 화면에서 실행합니다."),
}


def execution_block_response(category: str) -> dict:
    """차단 응답(answer + navigation 카드). 챗봇이 직접 실행하지 않고 이동 후보만 제시."""
    label, route, reason = _NAV_BY_CATEGORY.get(
        category, ("담당 화면으로 이동", "/agent/tasks", "이 작업은 담당 화면에서 처리합니다.")
    )
    answer = (
        "이 요청은 실제 운영 상태를 변경할 수 있어서 챗봇이 직접 수행할 수 없습니다. "
        "권한이 있는 화면에서 확인 후 처리해 주세요."
    )
    return {
        "answer": answer,
        "sources": [],
        "followUps": [],
        "spokenSummary": "이 작업은 챗봇이 직접 처리하지 않고 담당 화면에서 진행합니다.",
        "ui": {"type": "navigation", "props": {"label": label, "route": route, "reason": reason}},
        "toolsUsed": [],
        "confidence": "HIGH",
        "warnings": [],
    }
