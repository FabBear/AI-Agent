# fabBear Backend

Spring Boot와 FastAPI 기반 백엔드 저장소 협업 규칙입니다.

## AI Agent uv 환경 설정

AI Agent는 `uv`로 Python 가상환경과 의존성을 관리합니다.

### 1. uv 설치

```bash
brew install uv
```

### 2. 의존성 설치

```bash
cd AI-Agent
uv sync
```

`uv sync`를 실행하면 `.python-version`의 Python 버전에 맞춰 `.venv/`가 생성되고,
`pyproject.toml`과 `uv.lock` 기준으로 동일한 의존성이 설치됩니다.

### 3. 환경 변수 설정

```bash
cp .env.example .env
```

`.env`에 아래 값을 팀원 각자의 키로 채워 넣습니다.

```dotenv
OPENAI_API_KEY=your_key
TAVILY_API_KEY=your_key
HUGGINGFACEHUB_API_TOKEN=your_key
LANGCHAIN_API_KEY=your_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=FabBear
```

실제 키가 들어간 `.env`는 커밋하지 않습니다. 공유가 필요한 변수명은 `.env.example`만
수정해서 커밋합니다.

### 4. 실행 방법

Python 스크립트는 아래처럼 실행합니다.

```bash
uv run python path/to/script.py
```

패키지를 추가할 때는 `pip install` 대신 아래 명령을 사용합니다.

```bash
uv add package-name
uv sync
```

개발 도구 의존성은 아래처럼 추가합니다.

```bash
uv add --dev package-name
```

## Branch Strategy

```
main       ← 배포용 브랜치 (직접 push 금지)
 └─ dev    ← 개발 통합 브랜치
     └─ feat/#12-auth-api
     └─ fix/#24-token-expired-error
     └─ chore/#31-update-ci
```

브랜치 이름은 `타입/#이슈번호-설명` 형식으로 작성합니다.

### 브랜치 타입

| 타입       | 설명                                    |
| ---------- | --------------------------------------- |
| `feat`     | 신규 기능 또는 기존 기능 개선           |
| `fix`      | 버그 수정                               |
| `refactor` | 구조 개선 또는 리팩토링                 |
| `test`     | 테스트 코드 추가 또는 수정              |
| `docs`     | 문서 수정                               |
| `chore`    | 빌드, 패키지, 설정 등 프로젝트 관리 작업 |

---

## Issue Convention

이슈는 반드시 제공된 **템플릿**을 사용해 작성합니다.

### 이슈 타입

| 타입       | 설명                                  |
| ---------- | ------------------------------------- |
| ✨ Feature | 새로운 기능 추가 또는 기존 기능 개선 |
| 🐞 BugFix  | 예상과 다른 동작, 오류, 예외 상황     |
| 🧹 Chore   | 설정, 문서, 빌드, 의존성 등 관리 작업 |

`API`, `Database`, `Infra`, `Security`처럼 변경 영역을 구분해야 할 때는 이슈 타입을 늘리지 않고 라벨 또는 PR의 변경 범위에 표시합니다.

### 이슈 제목 규칙

```
[타입] 작업 내용 요약
```

```
[FEAT] 병목 감지 결과 저장 기능 구현
[BUG] 토큰 만료 시 500 응답 발생 오류 수정
[CHORE] Gradle 의존성 정리
```

> 이슈를 등록하면 작성자가 자동으로 assignee로 지정됩니다.

---

## Commit Convention

### 형식

```
[타입] 작업 내용 요약
```

### 커밋 타입

| 타입         | 설명                              | 예시                                           |
| ------------ | --------------------------------- | ---------------------------------------------- |
| `[FEAT]`     | 새로운 기능 추가                  | `[FEAT] 병목 감지 결과 저장 기능 구현`         |
| `[FIX]`      | 버그 수정                         | `[FIX] 토큰 만료 시 500 응답 발생 오류 수정`   |
| `[REFACTOR]` | 코드 리팩토링                     | `[REFACTOR] 인증 검증 로직 분리`               |
| `[TEST]`     | 테스트 코드 추가 또는 수정        | `[TEST] AuthService 단위 테스트 작성`          |
| `[DOCS]`     | 문서 수정                         | `[DOCS] API 실행 방법 추가`                    |
| `[CHORE]`    | 빌드, 패키지, 설정 등 관리 작업   | `[CHORE] 사용하지 않는 의존성 제거`            |
| `[HOTFIX]`   | 긴급 버그 수정                    | `[HOTFIX] 운영 DB 연결 설정 오류 수정`         |

### 작성 규칙

- 제목은 **명령문·현재형**으로 작성합니다. (과거형 ❌)
- 한 커밋에는 **하나의 논리적 변경**만 담습니다.
- 제목은 **50자 이내**로 작성합니다.
- API, DB, Infra 변경 여부는 PR 본문에 명시합니다.

---

## PR Convention

### PR 제목 규칙

커밋 타입과 동일한 prefix를 사용합니다.

```
[FEAT] #12 LOT 이벤트 조회 기능 구현
```

PR 제목의 prefix를 기준으로 **라벨이 자동 부여**됩니다. API, DB, Infra처럼 변경 영역이 명확한 경우에는 PR 제목 또는 라벨로 함께 표시합니다.

```
[FEAT] #12 LOT 이벤트 조회 기능 구현
[FIX] #24 토큰 만료 예외 처리 수정
[CHORE] #31 백엔드 CI 설정 정리
```

### PR 작성 규칙

- **Draft PR**: 작업 중인 PR은 Draft로 생성합니다. Draft 상태에서는 🚧 Not Ready for Review 라벨이 자동으로 붙습니다.
- **이슈 연결**: PR 본문의 `Close #이슈번호`를 반드시 작성합니다. merge 시 연결된 이슈가 자동으로 닫힙니다.
- **변경 범위**: Spring Boot, FastAPI, DB, Infra 중 어떤 영역이 변경되었는지 작성합니다.
- **검증 결과**: 실행한 테스트, API 호출, 마이그레이션 검증 결과를 작성합니다.
- **리뷰 요청**: Ready for Review 전환 후 리뷰어를 지정합니다.
- **셀프 머지 금지**: 본인이 작성한 PR은 본인이 merge하지 않습니다.
