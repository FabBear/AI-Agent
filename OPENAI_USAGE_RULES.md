# OpenAI API 사용 규칙

이 프로젝트에서 OpenAI API를 사용하는 모든 코드는 아래 규칙을 반드시 따른다.

## 1. max_tokens 제한

| 용도 | max_tokens |
|---|---|
| 원인 요약 (`llm_summarizer.py`) | 800 |
| 대응안 보강 (`llm_generator.py`) | 800 |
| 리포트 생성 | 2000 이하 |
| 기타 일반 답변 | 800~1200 |

응답 생성 시 `max_tokens`는 필요한 만큼만 설정한다. 무제한(`None`) 또는 과도하게 큰 값은 사용하지 않는다.

## 2. 테스트 시 소량 데이터만 사용

- 파이프라인 개발·테스트 중에는 **10~20건 샘플**로만 실행한다.
- 전체 데이터(106개 TG 전체) 실행은 **1회로 제한**한다.
- `run_detection.py` 실행 전 반드시 샘플 수를 확인한다.

## 3. 동일 요청 재사용 (캐싱)

- 동일한 입력(toolgroup + snapshot_time)에 대해 이미 생성된 결과가 있으면 LLM을 재호출하지 않는다.
- 결과는 파일(`compare_agent_out/`, `report_agent_out/`)로 저장하고 재사용한다.

## 4. 자동 반복 호출 금지

- 스케줄러, 루프, 테스트 코드에서 LLM을 **자동으로 반복 호출하는 구조**를 만들지 않는다.
- 테스트 코드에서 LLM 호출이 필요한 경우 mock 또는 저장된 결과를 사용한다.

## 5. API 키 관리

- API 키는 `.env` 파일에만 저장하며 코드에 직접 하드코딩하지 않는다.
- `.env`는 `.gitignore`에 포함되어야 한다.

```
# .env
OPENAI_API_KEY=sk-proj-...
```

## 6. 모델 설정

현재 사용 모델: `gpt-4o-mini`

모델 변경 시 `agents/cause_analyzer/llm_summarizer.py`와 `agents/solution_generator/llm_generator.py`의 `_MODEL` 상수를 수정한다.
