"""실행성 차단 가드레일 회귀 — 명령형(실행/승인/스케줄/DB write)은 차단, 조회·질문은 통과."""

from app.chatbot.guardrails import detect_execution_intent, execution_block_response


def test_blocks_approval_commands():
    assert detect_execution_intent("1번 대응안 승인해줘") == "approval"
    assert detect_execution_intent("이 케이스 반려해") == "approval"


def test_blocks_scheduling_and_db_write_and_agent_run():
    assert detect_execution_intent("Lot 순서 바꿔줘") == "schedule"
    assert detect_execution_intent("이 TG 투입량 줄여") == "schedule"
    assert detect_execution_intent("핫랏 우선순위 올려") == "schedule"
    assert detect_execution_intent("리포트 저장해줘") == "db_write"
    assert detect_execution_intent("이력 삭제해") == "db_write"
    assert detect_execution_intent("진단 돌려") == "agent_run"
    assert detect_execution_intent("파이프라인 실행해") == "agent_run"
    assert detect_execution_intent("hold 걸어") == "schedule"
    assert detect_execution_intent("release 막아") == "schedule"
    assert detect_execution_intent("PM 잡아") == "schedule"
    assert detect_execution_intent("알림 보내") == "db_write"
    assert detect_execution_intent("코멘트 남겨") == "db_write"
    assert detect_execution_intent("리포트 발행해") == "db_write"


def test_blocks_admin_change_commands():
    assert detect_execution_intent("모델 재학습해") == "admin_change"
    assert detect_execution_intent("모델 배포해") == "admin_change"
    assert detect_execution_intent("threshold 바꿔") == "admin_change"
    assert detect_execution_intent("임계값 올려") == "admin_change"


def test_allows_read_and_question_forms():
    assert detect_execution_intent("리포트 요약해줘") is None
    assert detect_execution_intent("최근 케이스 승인됐어?") is None
    assert detect_execution_intent("승인 대기 건수 알려줘") is None
    assert detect_execution_intent("대응안 추천해줘") is None
    assert detect_execution_intent("PM 상태 알려줘") is None
    assert detect_execution_intent("threshold 값 보여줘") is None
    assert detect_execution_intent("지금 WIP 가장 높은 구역 어디야?") is None
    assert detect_execution_intent("가동률 추세 보여줘") is None
    assert detect_execution_intent("") is None


def test_block_response_has_navigation_card():
    resp = execution_block_response("approval")
    assert resp["ui"]["type"] == "navigation"
    assert resp["ui"]["props"]["route"] == "/agent/approvals"
    assert resp["toolsUsed"] == [] and resp["confidence"] == "HIGH"


def test_admin_change_response_has_navigation_card():
    resp = execution_block_response("admin_change")
    assert resp["ui"]["type"] == "navigation"
    assert resp["ui"]["props"]["route"] == "/admin/thresholds"
