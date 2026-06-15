# Snapshot 3780 Demo

이 폴더는 대응안 비교 데모 재현에 필요한 목데이터를 모아 둔 곳입니다.

## 데모 실행

`AI-Agent/` 디렉토리에서 실행합니다.

```bash
uv run python run_detection.py --snapshot 3780 --demo-mock
```

첫 실행 시 `rag_cases/`의 과거 대응 보고서가 Qdrant에 자동 적재됩니다.

## 실행 전 준비

1. Docker 컨테이너(DB, Qdrant)가 올라와 있어야 합니다.
2. `.env`에 `OPENAI_API_KEY`가 설정되어 있어야 합니다.
3. `uv sync`로 의존성이 설치되어 있어야 합니다.

## 포함 데이터

- `verification_data.py`: 효과 검증 단계에서 사용하는 120분 합성 통계와 TG별 KPI 전망
- `rag_cases/*.md`: RAG가 검색하는 conservative / standard / aggressive 과거 대응 보고서
- `bootstrap.py`: Qdrant 목보고서 적재 스크립트

## 실행 흐름

`--demo-mock`을 사용하면 다음 순서로 동작합니다.

1. 실제 snapshot 데이터로 병목 탐지, 원인 분석, 대응안 생성을 수행합니다.
2. 효과 검증 단계부터 `verification_data.py`의 합성 통계를 사용합니다.
3. Qdrant의 `bottleneck_cases_demo_3780` 컬렉션이 비어 있으면 `rag_cases/` 문서를 자동 적재합니다.
4. compare/HITL에서 목보고서를 검색해 RAG 근거로 표시합니다.
5. 사용자가 대응안을 선택하면 기존 흐름대로 최종 보고서를 생성합니다.

## RAG 데이터 재적재

`rag_cases/`를 수정한 뒤 Qdrant에 반영할 때 (`AI-Agent/` 디렉토리에서 실행):

```bash
# 컬렉션 초기화 후 재적재 (중복 방지, 권장)
uv run python -m demo.bootstrap --clean

# 기존 컬렉션에 upsert만 (중복 가능성 있음)
uv run python -m demo.bootstrap --force
```

> `python demo/bootstrap.py`로 직접 실행하면 모듈 경로 오류가 납니다.
> 반드시 `python -m demo.bootstrap` 형식을 사용하세요.
