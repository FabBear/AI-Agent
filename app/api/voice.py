"""온프렘 STT — faster-whisper. 오디오를 외부로 보내지 않고 사내에서 전사한다."""

import io
import logging
import math
import os
import re
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, Form, UploadFile

from app.api.deps import InternalUser, get_internal_user
from app.api.schemas import ApiModel
from app.common.responses import ApiResponse, success
from app.services.stt_correction import correct_transcript_detailed

logger = logging.getLogger(__name__)
router = APIRouter()

_KPI_TERMS = (
    "WIP, 가동률, 가용률, Q-time, 대기시간, 셋업, 비가동, OEE, 라인 밸런스, 추세, "
    "TAT, Throughput, 처리량, 납기준수율, RTF, 우선순위 부스트, Hot Lot, Hold Lot, 대기 Lot, "
    "병목, PM, BM, MTTF, MTTR, 디스패칭, "
    "툴그룹, 툴, 구역, 구역별, 공정, 트렌드, 현황, 비교, 순위"
)

@lru_cache(maxsize=1)
def _model():
    # GPU 미정 → 우선 CPU int8. 환경변수로 모델/디바이스 조정 가능.
    # 한국어 정확도 위해 기본 small(여전히 CPU 가능). GPU 확보 시 large-v3로 격상.
    from faster_whisper import WhisperModel

    name = os.getenv("VOICE_STT_MODEL", "small")
    device = os.getenv("VOICE_STT_DEVICE", "cpu")
    compute = os.getenv("VOICE_STT_COMPUTE", "int8")
    logger.info("STT 모델 로드: %s (%s/%s)", name, device, compute)
    return WhisperModel(name, device=device, compute_type=compute)


def preload_model() -> None:
    """앱 기동 시 백그라운드 스레드로 STT 모델을 미리 로드(첫 발화 지연 제거).
    VOICE_STT_PRELOAD=1일 때만 동작 — 개발 환경 기동 속도는 건드리지 않는다."""
    if os.getenv("VOICE_STT_PRELOAD") != "1":
        return
    import threading

    def _warm() -> None:
        try:
            _model()
        except Exception:  # noqa: BLE001 - 프리워밍 실패는 첫 요청 때 lazy load로 복구된다.
            logger.exception("STT 모델 프리워밍 실패")

    threading.Thread(target=_warm, name="stt-preload", daemon=True).start()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _initial_prompt(vocab: list[str]) -> str:
    # 프롬프트는 간결해야 효과적(긴 TG 덤프는 오히려 희석). 대표 어휘만 힌트로.
    # 사람이 말하는 단위는 보통 '구역명'(Diffusion/Dielectric/Litho...) → 그것 위주로 노출.
    # whisper initial_prompt는 직전 발화 이어쓰기처럼 동작 → 실제 데모 발화 예문이 스타일 힌트가 된다.
    hint = ", ".join(vocab[:30]) if vocab else ""
    base = (
        "반도체 FAB 운영 현장 질문. 예: 전체 구역별 트렌드 보여줘. 가동률이 가장 높은 툴그룹 알려줘. "
        f"주요 용어: {_KPI_TERMS}."
    )
    return f"{base} 설비/구역: {hint}." if hint else base


class TranscribeResult(ApiModel):
    text: str
    raw: str
    language: str | None = None
    confidence: float | None = None


def _estimate_confidence(segments: list[object]) -> float:
    if not segments:
        return 0.0
    logprob_scores = []
    no_speech_scores = []
    for segment in segments:
        avg_logprob = getattr(segment, "avg_logprob", None)
        no_speech_prob = getattr(segment, "no_speech_prob", None)
        if isinstance(avg_logprob, int | float):
            logprob_scores.append(max(0.0, min(1.0, math.exp(float(avg_logprob)))))
        if isinstance(no_speech_prob, int | float):
            no_speech_scores.append(max(0.0, min(1.0, 1.0 - float(no_speech_prob))))
    if not logprob_scores and not no_speech_scores:
        return 0.0
    logprob_score = sum(logprob_scores) / len(logprob_scores) if logprob_scores else 0.5
    no_speech_score = sum(no_speech_scores) / len(no_speech_scores) if no_speech_scores else 0.5
    return round(max(0.0, min(1.0, logprob_score * 0.75 + no_speech_score * 0.25)), 3)


def _looks_like_hallucination(text: str) -> bool:
    """whisper의 잡음 환각/반복루프 탐지 → True면 버린다.
    예: '분이신 분이신 분이신...'(토큰 반복), 같은 어절 과다 반복."""
    t = text.strip()
    if not t:
        return True
    words = t.split()
    if len(words) >= 4:
        # 고유 어절 비율이 너무 낮으면 반복 루프(예: 같은 단어 반복)로 간주.
        uniq_ratio = len(set(words)) / len(words)
        if uniq_ratio < 0.35:
            return True
        # 같은 어절이 연속 4회 이상 반복.
        run = 1
        for i in range(1, len(words)):
            run = run + 1 if words[i] == words[i - 1] else 1
            if run >= 4:
                return True
    # 공백 없는 같은 음절 덩어리 과다 반복(예: 'ㅋㅋㅋ...', '아아아...').
    if re.search(r"(.{1,3})\1{5,}", t.replace(" ", "")):
        return True
    return False


@router.post("/transcribe", response_model=ApiResponse[TranscribeResult])
async def transcribe(
    _: Annotated[InternalUser, Depends(get_internal_user)],
    audio: UploadFile,
    vocab: Annotated[str, Form()] = "",
    language: Annotated[str, Form()] = "ko",
) -> ApiResponse[TranscribeResult]:
    vocab_list = [v.strip() for v in vocab.split(",") if v.strip()]
    data = await audio.read()
    beam_size = _int_env("VOICE_STT_BEAM_SIZE", 5)
    min_silence_ms = _int_env("VOICE_STT_MIN_SILENCE_MS", 500)
    segments, info = _model().transcribe(
        io.BytesIO(data),
        language=language or None,
        initial_prompt=_initial_prompt(vocab_list),
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=min_silence_ms),
        beam_size=beam_size,
        condition_on_previous_text=False,
        temperature=0,
        # 환각/잡음 가드: 무음 확률 높거나(no_speech), 반복적(compression_ratio)이거나,
        # 신뢰도 낮은(log_prob) 세그먼트는 whisper가 스스로 거른다.
        no_speech_threshold=float(os.getenv("VOICE_STT_NO_SPEECH", "0.6")),
        compression_ratio_threshold=float(os.getenv("VOICE_STT_COMPRESSION", "2.2")),
        log_prob_threshold=float(os.getenv("VOICE_STT_LOGPROB", "-0.8")),
    )
    segment_list = list(segments)
    raw = "".join(segment.text for segment in segment_list).strip()
    confidence = _estimate_confidence(segment_list)
    # 반복루프/잡음 환각은 텍스트·신뢰도를 0으로 → 호출측이 무시(전송 안 함).
    if _looks_like_hallucination(raw):
        logger.info("STT 환각/반복 탐지 → 폐기: %r", raw[:60])
        return success(TranscribeResult(text="", raw=raw, language=getattr(info, "language", language), confidence=0.0))
    correction = correct_transcript_detailed(raw, vocab_list, confidence=confidence)
    if correction.text != correction.raw:
        logger.info(
            "STT correction applied raw=%r corrected=%r confidence=%s rules=%s changed=%s",
            correction.raw[:300],
            correction.text[:300],
            confidence,
            correction.applied_rules,
            correction.changed_terms,
        )
    return success(
        TranscribeResult(
            text=correction.text,
            raw=raw,
            language=getattr(info, "language", language),
            confidence=confidence,
        )
    )
