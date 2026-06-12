"""음성 전사 도메인 보정 회귀 — 한국어 음차 오인식 사전 + TG명 정규화(결정론적 전처리, LLM 없음)."""

import logging

from app.services.stt_correction import correct_transcript, correct_transcript_detailed, load_correction_rules


def test_korean_mishear_dictionary():
    assert correct_transcript("내가 지금 봐야 할 툴구름 알려줘.") == "내가 지금 봐야 할 툴그룹 알려줘."
    assert correct_transcript("구역발 트렌드 보여줘.") == "구역별 트렌드 보여줘."
    assert correct_transcript("해던 구역에 있는 기기 정보 가져와.") == "해당 구역에 있는 기기 정보 가져와."
    assert correct_transcript("리소 맷 WIP 추이 보여줘.") == "Litho_Met WIP 추이 보여줘."


def test_demo_phrase_misrecognitions():
    # 데모에서 반복되는 발화의 고빈도 오인식(트렌드→툴그룹, 세리디벨벳→3D 맵 등)
    assert correct_transcript("가장 먼저 봐야 할 트렌드가 뭐야?") == "가장 먼저 봐야 할 툴그룹 뭐야?"
    assert correct_transcript("세리디벨벳에서 위치 찍어줄 수 있어?") == "3D 맵에서 위치 찍어줘?"


def test_past_tense_haetdeon_preserved_when_not_location():
    # '아까 했던 질문'의 '했던'은 표준 과거형 → 치환 금지(해당 구역류 문맥만 보정)
    assert correct_transcript("아까 했던 질문 다시 보여줘") == "아까 했던 질문 다시 보여줘"


def test_vocab_correction_normalizes_known_tool_groups():
    vocab = ["Dielectric_BE_60", "Dry_Etch", "DEF_MET_FE_118", "DefMet_FE_43"]
    assert correct_transcript("dielectric be 60 현황", vocab) == "Dielectric_BE_60 현황"
    assert correct_transcript("dry etch 대기 lot", vocab) == "Dry_Etch 대기 lot"
    assert correct_transcript("다이 일렉트릭 비 이 육십 현황", vocab) == "Dielectric_BE_60 현황"
    assert correct_transcript("데프멧 에프이 백십팔 위치 찍어줘", vocab) == "DEF_MET_FE_118 위치 찍어줘"
    assert correct_transcript("대프 멧 에프 이 사삼 리포트", vocab) == "DefMet_FE_43 리포트"


def test_rules_are_loaded_from_external_dictionary():
    rules = load_correction_rules()
    assert rules.korean_fixes
    assert "FE" in rules.part_aliases
    assert rules.numeric_words[8] == "팔"


def test_fuzzy_vocab_correction_is_confidence_gated():
    vocab = ["DEF_MET_FE_118"]
    assert (
        correct_transcript("데프멧 에프이 백십파 위치", vocab, confidence=0.9)
        == "DEF_MET_FE_118 위치"
    )
    assert (
        correct_transcript("데프멧 에프이 백십파 위치", vocab, confidence=0.3)
        == "Def_Met 에프이 백십파 위치"
    )


def test_detailed_result_exposes_correction_metadata(caplog):
    caplog.set_level(logging.INFO)
    result = correct_transcript_detailed("툴구름 알려줘", confidence=0.8)
    if result.text != result.raw:
        logging.getLogger("app.api.voice").info(
            "STT correction applied raw=%r corrected=%r confidence=%s rules=%s changed=%s",
            result.raw[:300],
            result.text[:300],
            0.8,
            result.applied_rules,
            result.changed_terms,
        )
    assert result.text == "툴그룹 알려줘"
    assert result.applied_rules
    assert "STT correction applied" in caplog.text


def test_empty_input_is_safe():
    assert correct_transcript("") == ""
    assert correct_transcript("   ") == ""
