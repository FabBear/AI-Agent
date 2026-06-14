# DB ↔ Agent 파이프라인 Gap 분석 (통합본)

> 작성일: 2026-06-13 / 최종 갱신: 2026-06-13  
> 대상: AI-Agent 파이프라인이 자동으로 채워야 하는 `public` 스키마 테이블들의 현황·문제·권고사항  
> 범위: `tt_ml_bottleneck_pred`는 ML 담당자 소관으로 이 문서에서 제외  
> 통합: `db-agent-frontend-sync-20260613.md` (프론트 요청 사항) 반영 완료  
> Flyway 버전: 현재 최신 `V11` 기준. 신규 마이그레이션은 `V12`부터 순차 적용 (버전 번호 최종 확정은 구현 시점에 결정)  
> Flyway 파일 위치: DDL(스키마 변경) → `Backend/src/main/resources/db/migration/core/` / Seed 데이터 → `Backend/src/main/resources/db/migration/dev/`  
> 운영 프로필은 `core`만, dev 프로필은 `core + dev` 모두 실행 (`application.yml` vs `application-dev.yml`)

---

## 목차

1. [파이프라인 전체 흐름 요약](#1-파이프라인-전체-흐름-요약)
2. [에이전트 자동 적재 대상 테이블 현황](#2-에이전트-자동-적재-대상-테이블-현황)
3. [❌ 완전 미적재: ps_tg_metrics.bottleneck_prob / risk_grade](#3-완전-미적재-ps_tg_metricsbottleneck_prob--risk_grade)
4. [⚠️ td_action_plan — KPI 델타 컬럼 전부 NULL](#4-td_action_plan--kpi-델타-컬럼-전부-null)
5. [⚠️ td_cause_analysis — 컬럼 의미 오용 및 누락 데이터](#5-td_cause_analysis--컬럼-의미-오용-및-누락-데이터)
6. [⚠️ td_response_report — 일부 컬럼 항상 NULL](#6-td_response_report--일부-컬럼-항상-null)
7. [⚠️ th_agent_step_log — model_version_id 미연결](#7-th_agent_step_log--model_version_id-미연결)
8. [컬럼 변경 권고 요약](#8-컬럼-변경-권고-요약)
9. [Drop 대상 테이블](#9-drop-대상-테이블)
10. [⚠️ th_hitl_decision — 스냅샷 컬럼 누락 및 감사 이력 취약](#10-th_hitl_decision--스냅샷-컬럼-누락-및-감사-이력-취약)
11. [프론트 요청 신규 DB 항목](#11-프론트-요청-신규-db-항목)
12. [Agent/MLOps 쓰기 경로 참조](#12-agentMLOps-쓰기-경로-참조)
13. [Frontend/API 읽기 경로 참조](#13-frontendapi-읽기-경로-참조)
14. [작업 우선순위 제안](#작업-우선순위-제안)

---

## 1. 파이프라인 전체 흐름 요약

```
MonitoringRealtimeScheduler (5분 주기)
  └─ BottleneckDetectionService.detectAndTrigger()
       └─ AgentClient → FastAPI POST /api/ml/predict
            ↓ HIGH/CRITICAL 예측 결과
       └─ tt_bottleneck_case 생성 (Spring Boot)
       └─ AgentClient → FastAPI POST /api/agent/run
            ↓ 에이전트 파이프라인 실행
            ├─ th_agent_step_log (FastAPI 직접)
            ├─ td_cause_analysis (FastAPI 직접)
            ├─ td_action_plan   (FastAPI 직접)
            └─ SpringClient → POST /api/internal/snapshot
                 └─ td_bottleneck_snapshot (Spring Boot)
                 └─ tt_notification        (Spring Boot)

HITL 승인 후
  └─ FastAPI POST /api/agent/hitl-result
       └─ td_response_report (FastAPI 직접)
```

---

## 2. 에이전트 자동 적재 대상 테이블 현황

| 테이블 | 적재 주체 | 현재 상태 |
|---|---|---|
| `th_agent_step_log` | FastAPI `AgentStepRepository` | ✅ 정상 |
| `td_cause_analysis` | FastAPI `CauseAnalysisRepository` | ⚠️ 컬럼 오용 |
| `td_action_plan` | FastAPI `ActionPlanRepository` | ⚠️ KPI 델타 컬럼 NULL |
| `td_response_report` | FastAPI `ResponseReportRepository` | ⚠️ 일부 컬럼 NULL |
| `th_drift_alert` | FastAPI `DriftAlertRepository` | ✅ 정상 |
| `tt_bottleneck_case` | Spring Boot (`BottleneckDetectionService`) | ✅ 정상 |
| `td_bottleneck_snapshot` | Spring Boot (`InternalSnapshotService`) | ✅ 정상 |
| `tt_notification` | Spring Boot (`InternalSnapshotService`) | ✅ 정상 |
| **`ps_tg_metrics.bottleneck_prob`** | **없음** | ❌ 완전 미적재 |
| **`ps_tg_metrics.risk_grade`** | **없음** | ❌ 완전 미적재 |

---

## 3. 완전 미적재: `ps_tg_metrics.bottleneck_prob` / `risk_grade`

### 문제

`ps_tg_metrics`에는 `bottleneck_prob NUMERIC(5,4)`, `risk_grade VARCHAR(10)` 컬럼이 있다.  
ML 예측 결과(`FastAPI /predict` → `BottleneckDetectionService`)가 나와도 이 컬럼을 **UPDATE하는 코드가 없다**.  
시뮬레이션 KPI 적재(`04__pivot_tg_metrics.sql`) 때 두 컬럼이 NULL로 Insert되고, 이후 ML 추론이 돌아도 계속 NULL이다.

### 현재 프론트엔드에서 위험등급이 "보이는" 이유

| 표시 위치 | 출처 | 실제 동작 |
|---|---|---|
| SSE `processSummaries[].riskGrade` (Area 단위) | `maxUtilizationRate` 임계값 파생 (CRITICAL≥0.9 / HIGH≥0.85 / MEDIUM≥0.7) | ✅ 정상 (ML 무관) |
| SSE `toolGroups[].riskGrade` (TG 단위) | `ps_tg_metrics.risk_grade` 직접 읽기 | ❌ 항상 NULL |
| SSE `toolGroups[].bottleneckProb` | `ps_tg_metrics.bottleneck_prob` 직접 읽기 | ❌ 항상 NULL |
| SSE `processSummaries[].riskCounts` | `ps_tg_metrics.risk_grade` 집계 | ❌ 항상 모두 0 |
| 병목 모니터링 REST API (TG 목록·상세·순위) | `TgMetricsSnapshotReader` → `ps_tg_metrics` | ❌ 항상 NULL |
| 대시보드 공정맵 | `TgMetricsSnapshotReader` → `ps_tg_metrics` | ❌ 항상 NULL |
| 챗봇 `top_toolgroups(metric="bottleneck")` | `ps_tg_metrics.bottleneck_prob` | ❌ 항상 NULL |

즉, 현재 화면에서 **공정(Area) 단위** 색상은 utilization 기반으로 정상 표시되지만, **TG 단위** 병목 확률·위험등급은 모두 NULL이다.

### 해결 방향 (확정)

**Spring Boot 측 작업.** DB 스키마 변경 없이 코드만 추가하면 된다.

#### 신규 파일: `TgMetricsBottleneckWriter.java`

- 위치: `com.fabbear.backend.internal.dao`
- `NamedParameterJdbcTemplate`를 주입받아 아래 SQL을 배치 실행한다.
- 입력: `List<PredictionRow(UUID tgId, BigDecimal prob, String grade)>`

```sql
-- tg_id별 최신 measured_at 행에 ML 결과를 덮어쓴다
UPDATE ps_tg_metrics
SET bottleneck_prob = :prob,
    risk_grade      = :grade
WHERE tg_id = :tgId
  AND measured_at = (
      SELECT MAX(measured_at) FROM ps_tg_metrics WHERE tg_id = :tgId
  );
```

> TimescaleDB 하이퍼테이블은 `DISTINCT ON` + `ORDER BY measured_at DESC` 서브쿼리를 써도 동일하게 동작한다.  
> PK가 `(tg_id, measured_at)` 복합키이므로 UPDATE 조건은 반드시 두 컬럼 모두 사용해야 인덱스를 탄다.

#### 수정 파일: `BottleneckDetectionService.java`

- `TgMetricsBottleneckWriter`를 생성자 주입
- `agentClient.predict()` 직후, **HIGH/CRITICAL 필터링 전** 전체 예측 결과(`predictResult.predictions()`)를 `batchUpdate()`에 넘긴다
- HIGH/CRITICAL만 UPDATE하면 나머지 TG의 `risk_grade`가 이전 값으로 남아 오염되므로, **반드시 전체 TG를 갱신**해야 한다

```
detectAndTrigger()
  ├─ agentClient.predict(request) → predictResult
  ├─ [추가] tgMetricsBottleneckWriter.batchUpdate(predictResult.predictions())  ← 전체 TG
  └─ predictResult.predictions().stream().filter(isHighOrCritical) → 케이스 생성·에이전트 트리거
```

---

## 4. `td_action_plan` — KPI 델타 컬럼 전부 NULL

### 스키마 vs 파이프라인 출력 비교

`ActionPlanRepository.bulk_insert()`가 현재 채우는 컬럼:

| 컬럼 | 현재 값 |
|---|---|
| `case_id` | ✅ |
| `plan_seq` | ✅ (1~N) |
| `plan_type` | ✅ (`LOT_RELEASE_INTERVAL` / `DISPATCH_RULE_OVERRIDE` 등) |
| `plan_title` | ✅ (candidate의 `name`/`description`/`plan_id`) |
| `plan_detail` | ✅ (candidate의 `expected_effect`) |
| `simulation_basis` | ✅ (candidate 전체를 `json.dumps()`로 덤프) |
| `est_throughput_delta` | ❌ NULL |
| `est_avg_wait_delta` | ❌ NULL |
| `est_delivery_compliance_delta` | ❌ NULL |
| `est_delay_delta` | ❌ NULL |
| `actual_*` 4개 + `validated_at` | ❌ NULL (운영 후 배치 기입 예정 — 의도적) |

### 파이프라인이 실제로 계산하는 KPI 델타 (버리고 있는 값)

`VerificationAgent` → `CompareAgent` 결과 (`kpi_stats` dict):

| 키 | 의미 |
|---|---|
| `q_time_min.mean_delta` | Q-time 평균 변화량 (분) |
| `wip.mean_delta` | WIP 평균 변화량 |
| `wait_ratio.mean_delta` | Wait ratio 변화량 |
| `utilization_avg.mean_delta` | 가동률 변화량 |
| `available_tool_ratio.mean_delta` | 가용 장비 비율 변화량 |
| `paired_t_p` | 페어드 t-검정 p-value |
| `paired_n` | 유효 시뮬 쌍 수 (최대 30) |

이 값들이 `simulation_basis` JSONB 안에 묻혀 있어 DB에서 직접 집계·비교·정렬이 불가능하다.

### 불필요한 컬럼

아래 두 컬럼은 파이프라인이 해당 KPI를 계산하지 않으므로 채워질 수 없다:
- `est_delivery_compliance_delta` — 납기 준수율 델타 (시뮬에 없음)
- `est_delay_delta` — 지연 시간 델타 (시뮬에 없음)

### 권고

**기존 컬럼 대체 또는 신규 추가:**

```sql
-- 파이프라인이 실제 계산하는 값에 맞게 컬럼 조정
ALTER TABLE td_action_plan
    RENAME COLUMN est_throughput_delta      TO est_util_delta;        -- 가동률 → 파이프라인 출력값
ALTER TABLE td_action_plan
    RENAME COLUMN est_avg_wait_delta        TO est_q_time_delta;      -- Q-time 델타 (분)
DROP COLUMN est_delivery_compliance_delta;  -- 파이프라인 미계산
DROP COLUMN est_delay_delta;                -- 파이프라인 미계산
ALTER TABLE td_action_plan
    ADD COLUMN est_wip_delta        NUMERIC(8,2),
    ADD COLUMN est_wait_ratio_delta NUMERIC(8,4),
    ADD COLUMN sim_paired_n         SMALLINT,
    ADD COLUMN sim_paired_p_value   NUMERIC(8,6);
```

**`ActionPlanRepository.bulk_insert()` 수정:** `kpi_stats` 딕셔너리에서 값을 추출해 해당 컬럼에 채운다.

---

## 5. `td_cause_analysis` — 컬럼 의미 오용 및 누락 데이터

### 컬럼 의미 불일치

| 컬럼 | DDL 의도 | 현재 저장 값 | 올바른 값 |
|---|---|---|---|
| `bottleneck_cause_type` | 병목 원인 분류 코드 (`"설비_포화"` 등) | `shap_top[0].feature` → `"utilization_avg"` 같은 **KPI 피처명** | `CauseReport.judgment.primary_category` (`"설비_포화"` / `"대기_누적"` / `"WIP_누적"` / `"공급_부족"`) |
| `diffusion_affected_tg_ids` | **다운스트림** 확산 영향 TG UUID 목록 | `upstream_suspects` 기반 UUID → **업스트림** TG | `BottleneckAlert.impact.affected_tgs` (CASCADE_ANALYZER 출력 — 다운스트림) |
| `shap_features` | SHAP 기여도 | `[{feature, importance=abs(shap_value), rank}]` — 부호 소실 | `shap_value` 원본(부호 포함) + `kpi_value` 모두 보존 필요 |

### 파이프라인이 생성하지만 DB에 저장 안 되는 값

| `CauseReport` 필드 | 내용 | 현재 처리 |
|---|---|---|
| `judgment.primary_category` | LLM 판정 주원인 카테고리 | ❌ 미저장 (bottleneck_cause_type에 잘못된 값이 들어감) |
| `judgment.primary_cause` | 대표 피처명 | ❌ 미저장 |
| `judgment.cause_summary` | LLM 자연어 요약 | ❌ 미저장 |
| `consensus` | 4축 합의 결과 (SHAP·트렌드·업스트림·G*) | ❌ 미저장 |
| `trend_top` | 트렌드 분석 결과 (slope, R², 유의성) | ❌ 미저장 |
| `sim_forecast` | 포워드 시뮬 예측 결과 | ❌ 미저장 |
| `cause_categories` | 원인 카테고리별 종합 점수 | ❌ 미저장 |
| `evidence_bundle` | 피처별 4축 증거 상세 | ❌ 미저장 |

### 권고

```sql
-- 기존 컬럼 의미 수정 (코드 변경 포함)
-- bottleneck_cause_type: CauseReport.judgment.primary_category 값으로 채우도록 코드 수정
-- diffusion_affected_tg_ids: CascadeImpact.affected_tgs 기반으로 채우도록 코드 수정

-- 누락 데이터를 위한 컬럼 추가
ALTER TABLE td_cause_analysis
    ADD COLUMN primary_cause_feature  VARCHAR(100),    -- judgment.primary_cause (피처명)
    ADD COLUMN cause_judgment_json    JSONB,            -- judgment 전체 (LLM 판정)
    ADD COLUMN consensus_json         JSONB,            -- consensus 전체 (4축 합의)
    ADD COLUMN trend_json             JSONB;            -- trend_top 결과
```

**`CauseAnalysisRepository.upsert()` 수정 사항:**
1. `bottleneck_cause_type` → `report.judgment.primary_category` 값으로 채우기
2. `diffusion_affected_tg_ids` → `state["alerts"]`에서 해당 TG의 `CascadeImpact.affected_tgs`를 TG UUID로 조회해 저장
3. `shap_features` → `shap_value` 원본 부호 보존 (`importance`를 `abs()` 없이 `shap_value`로 저장)
4. 신규 컬럼에 `judgment`, `consensus`, `trend_top` 저장

---

## 6. `td_response_report` — 일부 컬럼 항상 NULL

### `report_agent` 출력 구조

`report_agent` 노드는 `report_results` 리스트를 반환하는데, 각 항목의 구조는:

```python
{"toolgroup": tg, "output_path": str(md_path), "json_output_path": str(json_path)}
```

`ResponseReportRepository.insert()`가 이 dict에서 읽는 방식:

| 컬럼 | 코드 | 실제 결과 |
|---|---|---|
| `report_html` | `output_path` 파일 읽기 (Markdown 텍스트) | ✅ 적재됨 (컬럼명은 HTML인데 Markdown 내용) |
| `summary` | `meta.summary` 또는 앞 500자 | ✅ 적재됨 |
| `timeline_json` | `json.dumps({"toolgroup":…, "output_path":…, "json_output_path":…})` | ❌ 파일 경로 dict가 들어감 (timeline 데이터 아님) |
| `root_cause_text` | `report_result.get("root_cause_text")` | ❌ **항상 NULL** (key 없음) |
| `action_comparison_text` | `report_result.get("action_comparison_text")` | ❌ **항상 NULL** (key 없음) |

실제 구조화된 리포트 JSON(`ReportV2` 스키마)은 `json_output_path` 파일에 존재하지만, 이 경로가 DB에 저장되지 않는다.

### 권고

**DB 컬럼 수정 (프론트 요청 기준으로 정리):**
```sql
ALTER TABLE td_response_report
    DROP COLUMN IF EXISTS report_html,               -- rendered_markdown으로 역할 대체
    ADD COLUMN IF NOT EXISTS report_schema_version   VARCHAR(30),
    ADD COLUMN IF NOT EXISTS report_json             JSONB,
    ADD COLUMN IF NOT EXISTS rendered_markdown       TEXT;

CREATE INDEX IF NOT EXISTS idx_response_report_schema_version
    ON td_response_report (report_schema_version);
```

> `timeline_json`은 현재 파일 경로 dict가 들어가 있으므로, `report_json` 적재 완료 후 DROP 검토.

**`ResponseReportRepository.insert()` 수정:**
- `report_json` → `json_output_path` 파일을 열어 `ReportV2` JSON 전체를 JSONB로 저장 (리포트 탭·Q&A SoT)
- `rendered_markdown` → `output_path` 파일(Markdown 텍스트)을 읽어 저장 (기존 `report_html` 역할 대체)
- `report_schema_version` → `"report/1.0"` 등 버전 상수 저장
- `root_cause_text` → `ReportV2.cause.summary` 값 추출 후 저장
- `action_comparison_text` → `ReportV2.actions.recommendation.headline` + `primary_reason` 조합 후 저장

---

## 7. `th_agent_step_log` — `model_version_id` 미연결

### 문제

`th_agent_step_log.model_version_id`는 해당 Agent 단계에서 사용한 ML 모델 버전을 `th_ml_model_version`과 연결하는 FK 컬럼이다.  
현재 `AgentStepRepository`의 모든 메서드에서 이 컬럼을 채우는 코드가 없어 **항상 NULL**이다.

### 권고

`AgentStepRepository.mark_done()` 호출 시 선택적으로 `model_version_id`를 전달할 수 있도록 수정:

- `DIFFUSION_ANALYSIS`, `CAUSE_ANALYSIS` 단계: `MlModelRepository.find_active()`로 현재 ACTIVE 모델 버전 ID 조회 후 전달
- 나머지 단계(LLM 전용): NULL 유지

---

## 8. 컬럼 변경 권고 요약

### `td_action_plan` (Vxx 마이그레이션)

| 변경 유형 | 컬럼 | 내용 |
|---|---|---|
| RENAME | `est_throughput_delta` → `est_util_delta` | 가동률 델타로 재사용 |
| RENAME | `est_avg_wait_delta` → `est_q_time_delta` | Q-time 델타 (분) |
| DROP | `est_delivery_compliance_delta` | 파이프라인 미계산 |
| DROP | `est_delay_delta` | 파이프라인 미계산 |
| ADD | `est_wip_delta NUMERIC(8,2)` | WIP 변화량 |
| ADD | `est_wait_ratio_delta NUMERIC(8,4)` | 대기율 변화량 |
| ADD | `sim_paired_n SMALLINT` | 페어드 시뮬 횟수 |
| ADD | `sim_paired_p_value NUMERIC(8,6)` | t-검정 p-value |

### `td_cause_analysis` (Vxx 마이그레이션)

| 변경 유형 | 컬럼 | 내용 |
|---|---|---|
| 코드 수정 | `bottleneck_cause_type` | `judgment.primary_category` 값으로 채우기 |
| 코드 수정 | `diffusion_affected_tg_ids` | 다운스트림 cascade TG UUID로 채우기 |
| 코드 수정 | `shap_features` | `shap_value` 부호 보존 (abs() 제거) |
| ADD | `primary_cause_feature VARCHAR(100)` | `judgment.primary_cause` (피처명) |
| ADD | `cause_judgment_json JSONB` | LLM 판정 전체 |
| ADD | `consensus_json JSONB` | 4축 합의 결과 |
| ADD | `trend_json JSONB` | 트렌드 분석 결과 |

### `td_response_report` (Vxx 마이그레이션)

| 변경 유형 | 컬럼 | 내용 |
|---|---|---|
| DROP | `report_html` | `rendered_markdown`으로 역할 대체 — 불필요 |
| ADD | `report_json JSONB` | 리포트 Q&A·탭 SoT. `ReportV2` JSON 전체 저장 |
| ADD | `rendered_markdown TEXT` | PDF·AI 답변 설명 보조용 Markdown (기존 `report_html` 대체) |
| ADD | `report_schema_version VARCHAR(30)` | 리포트 원본 스키마 버전 (예: `report/1.0`) |
| 코드 수정 | `root_cause_text` | `ReportV2.cause.summary` 추출 |
| 코드 수정 | `action_comparison_text` | `ReportV2.actions.recommendation` 요약 추출 |
| DROP 검토 | `timeline_json` | `report_json`으로 대체되므로 이전 완료 후 제거 |

### `ps_tg_metrics` — UPDATE 로직 추가 (Spring Boot) ✅ 방향 확정

| 변경 유형 | 대상 | 내용 |
|---|---|---|
| 신규 파일 | `com.fabbear.backend.internal.dao.TgMetricsBottleneckWriter` | `batchUpdate(List<PredictionRow>)` — tg_id별 최신 행에 `bottleneck_prob`, `risk_grade` UPDATE |
| 수정 파일 | `BottleneckDetectionService.detectAndTrigger()` | `agentClient.predict()` 직후 전체 TG 예측을 `batchUpdate()`에 전달 (HIGH/CRITICAL 필터 전) |

> `ps_tg_metrics`는 TimescaleDB 하이퍼테이블. DB FK 없음. UPDATE 조건: `tg_id = :tgId AND measured_at = (SELECT MAX(...) WHERE tg_id = :tgId)`  
> **전체 TG를 갱신**해야 하는 이유: HIGH/CRITICAL만 쓰면 나머지 TG의 `risk_grade`가 이전 예측값으로 남아 오염됨

---

## 9. Drop 대상 테이블

### `th_audit_log`

| 항목 | 내용 |
|---|---|
| 원래 목적 | 사용자·역할·메뉴 권한 변경 이력 보안 감사 (`APP_USER`, `ROLE`, `USER_ROLE`, `ROLE_MENU`, `MENU`) |
| 현재 상태 | 미사용 — 적재 코드 없음, 어느 서비스에서도 참조 없음 |
| FK 의존 | `tm_app_user(user_id)` → `ON DELETE SET NULL` (CASCADE 없음, DROP 시 단순 FK 제거면 충분) |
| 조치 | `DROP TABLE` |

**마이그레이션 예시**

```sql
-- Vxx__drop_unused_audit_log.sql
ALTER TABLE th_audit_log DROP CONSTRAINT IF EXISTS fk_th_audit_log_user;
DROP TABLE IF EXISTS th_audit_log;
```

> Flyway 버전 번호는 현재 최신(`V11`) 이후로 지정한다.

### `tm_knowledge_document`

| 항목 | 내용 |
|---|---|
| 원래 목적 | RAG 소스 문서 메타데이터 + Qdrant 임베딩 상태 관리 (`qdrant_doc_id`, `qdrant_indexed` 등) |
| 현재 상태 | 미사용 — Backend Java·AI-Agent Python 어느 쪽도 이 테이블을 참조하지 않음 |
| 실제 Qdrant 접근 방식 | 챗봇 RAG(`app/chatbot/rag.py`, `app/chatbot/tools/knowledge.py`)는 Qdrant collection `knowledge_docs`에 직접 조회 — DB 경유 없음 |
| FK 의존 | `tm_fab.fab_id` ON DELETE SET NULL, `tm_app_user.user_id` (incoming FK 없음) |
| 조치 | `DROP TABLE` |

**마이그레이션 예시**

```sql
-- Vxx__drop_unused_knowledge_document.sql
ALTER TABLE tm_knowledge_document DROP CONSTRAINT IF EXISTS fk_tm_knowledge_document_fab;
ALTER TABLE tm_knowledge_document DROP CONSTRAINT IF EXISTS fk_tm_knowledge_document_user;
DROP TABLE IF EXISTS tm_knowledge_document;
```

---

## 10. ⚠️ th_hitl_decision — 스냅샷 컬럼 누락 및 감사 이력 취약

### 현재 스키마

```sql
th_hitl_decision (
  decision_id      UUID PK,
  case_id          UUID FK→tt_bottleneck_case,
  re_decision_seq  SMALLINT,      -- 재결정 이력 번호 (1부터 증가)
  decided_by       UUID FK→tm_app_user ON DELETE SET NULL,
  decision         VARCHAR(10),   -- 'APPROVED' | 'REJECTED'
  selected_plan_id UUID FK→td_action_plan ON DELETE SET NULL,
  comment          TEXT,
  decided_at       TIMESTAMPTZ,
  created_at       TIMESTAMPTZ DEFAULT NOW()
)
```

### HITL 흐름 요약

```
[프론트] 관리자가 대응안 A/B/C 중 하나 선택
  └─ BncService.decideHitl()
       ├─ HitlDecisionRepository.save()   → th_hitl_decision 적재 (Spring Boot)
       └─ AgentClient.sendHitlResult()    → FastAPI POST /api/agent/hitl-result
            └─ run_post_hitl()
                 ├─ ActionPlanRepository.find_by_id()  (선택 플랜 조회)
                 ├─ build_phase2_pipeline()             (REPORT_AGENT 실행)
                 └─ ResponseReportRepository.insert()  → td_response_report 적재
```

### 컬럼 현황 분석

| 컬럼 | 현재 | 판정 | 근거 |
|---|---|---|---|
| `decision_id` | UUID PK | ✅ 유지 | |
| `case_id` | FK | ✅ 유지 | |
| `re_decision_seq` | 재결정 번호 | ✅ 유지 | `BncService`에서 MAX+1로 증가 처리 |
| `decided_by` | FK ON DELETE SET NULL | ✅ 유지 | 단, 스냅샷 보완 필요 (아래 참조) |
| `decision` | 'APPROVED'/'REJECTED' | ✅ 유지 | |
| `selected_plan_id` | FK ON DELETE SET NULL | ✅ 유지 | 단, 스냅샷 보완 필요 (아래 참조) |
| `comment` | TEXT nullable | ✅ 유지 | |
| `decided_at` | TIMESTAMPTZ | ✅ 유지 | |
| `created_at` | TIMESTAMPTZ DEFAULT NOW() | ❌ **제거** | `decided_at`와 항상 동일값 저장됨 (`HitlDecision.create()`에서 `decidedAt`로 동일 세팅) |

### 누락 컬럼 (추가 필요)

| 컬럼 | 타입 | 이유 |
|---|---|---|
| `decided_by_name` | `VARCHAR(100)` | `decided_by` FK ON DELETE SET NULL 시 결정자 이름 유실. `HistoryReadDao`에서 `JOIN tm_app_user`로 `user_name` 조회하는데 사용자 삭제되면 히스토리에서 결정자 표시 불가 |
| `selected_plan_title` | `VARCHAR(500)` | `selected_plan_id` FK ON DELETE SET NULL 시 어떤 플랜 선택했는지 히스토리 표시 불가. 결정 시점 `plan_title` 스냅샷 필요 |
| `selected_plan_type` | `VARCHAR(50)` | 동일 이유로 `plan_type`('conservative'/'standard'/'aggressive') 스냅샷 필요 |
| `agent_sync_status` | `VARCHAR(10)` | FastAPI 전송 성공/실패가 DB에 기록되지 않음. `BncService`는 `AgentSyncStatus.SYNCED`/`FAILED`를 응답으로만 반환하고 저장 안 함 → 전송 실패 케이스 파악 불가. 값: `'SYNCED'`, `'FAILED'`, `'PENDING'` |

### 코드 변경이 동반되는 사항

| 파일 | 변경 내용 |
|---|---|
| `HitlDecision.java` | `decidedByName`, `selectedPlanTitle`, `selectedPlanType`, `agentSyncStatus` 필드 추가 |
| `HitlDecision.create()` | 위 4개 파라미터 추가 |
| `BncService.decideHitl()` | 결정 저장 전 `tm_app_user`에서 `user_name`, `td_action_plan`에서 `plan_title`/`plan_type` 조회 후 전달. FastAPI 전송 결과에 따라 `agent_sync_status` UPDATE |
| `HitlDecisionRepository` | `findMaxReDecisionSeqByCaseId()` — 변경 없음 |
| `HistoryReadDao` | `decided_by_name`, `selected_plan_title` 컬럼 직접 조회로 전환 (JOIN 불필요) |

### 마이그레이션 예시

```sql
-- Vxx__alter_th_hitl_decision.sql
ALTER TABLE th_hitl_decision
    ADD COLUMN decided_by_name      VARCHAR(100),
    ADD COLUMN selected_plan_title  VARCHAR(500),
    ADD COLUMN selected_plan_type   VARCHAR(50),
    ADD COLUMN agent_sync_status    VARCHAR(10)
        CHECK (agent_sync_status IN ('SYNCED', 'FAILED', 'PENDING')),
    DROP COLUMN created_at;
```

---

## 11. 프론트 요청 신규 DB 항목

> 출처: `db-agent-frontend-sync-20260613.md` 섹션 2  
> 아래 항목은 프론트/챗봇/MLOps 기능 확장을 위해 새로 추가가 필요한 테이블·컬럼이다. 버전 번호는 `V12` 이후 구현 시점에 확정한다.

### `tt_fab_briefing` (신규 테이블)

FAB 현황 브리핑 Agent 실행 이력 저장용.

```sql
CREATE TABLE tt_fab_briefing (
    briefing_id  UUID         NOT NULL DEFAULT gen_random_uuid(),
    user_id      UUID         NOT NULL,
    fab_id       UUID         NOT NULL,
    status       VARCHAR(15)  NOT NULL DEFAULT 'QUEUED',
    result_json  JSONB,
    error_msg    TEXT,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    started_at   TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    CONSTRAINT pk_tt_fab_briefing PRIMARY KEY (briefing_id),
    CONSTRAINT ck_tt_fab_briefing_status CHECK (
        status IN ('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED')
    ),
    CONSTRAINT fk_tt_fab_briefing_user FOREIGN KEY (user_id)
        REFERENCES tm_app_user (user_id),
    CONSTRAINT fk_tt_fab_briefing_fab FOREIGN KEY (fab_id)
        REFERENCES tm_fab (fab_id)
);

CREATE INDEX idx_tt_fab_briefing_user_created
    ON tt_fab_briefing (user_id, created_at DESC);
```

### `tt_period_report` (신규 테이블)

기간 리포트 Agent 실행 이력 저장용.

```sql
CREATE TABLE tt_period_report (
    report_run_id UUID        NOT NULL DEFAULT gen_random_uuid(),
    user_id       UUID        NOT NULL,
    fab_id        UUID        NOT NULL,
    status        VARCHAR(15) NOT NULL DEFAULT 'QUEUED',
    report_intent VARCHAR(20) NOT NULL DEFAULT 'SUMMARY',
    period_from   TIMESTAMPTZ,
    period_to     TIMESTAMPTZ,
    filters       JSONB,
    result_json   JSONB,
    error_msg     TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at    TIMESTAMPTZ,
    completed_at  TIMESTAMPTZ,

    CONSTRAINT pk_tt_period_report PRIMARY KEY (report_run_id),
    CONSTRAINT ck_tt_period_report_status CHECK (
        status IN ('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED')
    ),
    CONSTRAINT ck_tt_period_report_intent CHECK (
        report_intent IN ('MONTHLY', 'SUMMARY', 'PATTERN', 'RETROSPECTIVE')
    ),
    CONSTRAINT fk_tt_period_report_user FOREIGN KEY (user_id)
        REFERENCES tm_app_user (user_id),
    CONSTRAINT fk_tt_period_report_fab FOREIGN KEY (fab_id)
        REFERENCES tm_fab (fab_id)
);

CREATE INDEX idx_tt_period_report_user_created
    ON tt_period_report (user_id, created_at DESC);
```

### `td_chat_message` 컬럼 추가

챗봇 답변 메타 정보 저장. 쓰기 주체는 AI-Agent 응답 → Spring `ChatbotService`가 영속화.

```sql
ALTER TABLE td_chat_message
    ADD COLUMN ui_card    JSONB,   -- generative UI 카드 JSON
    ADD COLUMN warnings   JSONB,   -- 답변 사용 주의사항 (컨텍스트 부족, 도구 실패, 오래된 데이터 등)
    ADD COLUMN tools_used JSONB;   -- 답변 생성에 사용된 도구 목록
```

### `th_drift_alert.detail` 컬럼 추가

F1 drift 상세 리포트. 쓰기 주체: AI-Agent `DriftAlertRepository` + MLOps `evaluate_model_drift.py`.

```sql
ALTER TABLE th_drift_alert
    ADD COLUMN detail JSONB;
```

`detail` 예시 구조:
```json
{
  "f1_baseline": 0.84,
  "f1_current": 0.72,
  "threshold": 0.8,
  "eval_window_hours": 24,
  "sample_count": 1200,
  "top_contributors": [
    { "toolgroup": "WE_FE_8", "fn": 13, "fp": 4 }
  ],
  "active_version": "7",
  "recommendation": "재학습 검토 필요"
}
```

---

## 12. Agent/MLOps 쓰기 경로 참조

> 출처: `db-agent-frontend-sync-20260613.md` 섹션 3  
> Spring은 화면 API·승인 게이트 역할이 크고, 병목 분석 산출물의 실제 쓰기 주체는 AI-Agent다.

| 테이블 | 쓰는 주체 | 파일 | 내용 |
|---|---|---|---|
| `td_cause_analysis` | AI-Agent | `app/repositories/cause_analysis_repository.py` | 원인 분석, SHAP, 확산 영향 TG, RAG 참조 |
| `td_action_plan` | AI-Agent | `app/repositories/action_plan_repository.py` | 대응안 후보, 예상 효과, simulation basis |
| `th_agent_step_log` | AI-Agent | `app/repositories/agent_step_repository.py` | Agent 단계 진행/실패/요약 로그 |
| `td_response_report` | AI-Agent | `app/repositories/response_report_repository.py` | `report_json`, `rendered_markdown`, 리포트 메타 |
| `th_drift_alert` | AI-Agent | `app/repositories/drift_alert_repository.py` | drift alert 저장 |
| `th_drift_alert` | MLOps | `MLOps/simulation/ML/evaluate_model_drift.py` | F1 임계 미달 drift 저장 |
| `th_ml_model_version` | MLOps | `MLOps/simulation/ML/validate_model.py` | 검증 완료 모델 DB mirror |

### MLflow 운영 상태 기준

MLflow 3.x 기준으로 stage API 대신 alias를 사용한다.

- `validate_model.py`: 검증 완료 모델을 DB mirror에 `STAGING`으로 기록
- Admin promote API/Spring: MLflow alias `production`으로 설정 + DB `ACTIVE`→`RETIRED` 처리 후 신규 모델을 `ACTIVE`로 승격
- **Serving SoT**: MLflow alias `production`
- `th_ml_model_version`: Admin UI용 mirror. 모델명 `FabGuard_Bottleneck_Model`로 통일

---

## 13. Frontend/API 읽기 경로 참조

> 출처: `db-agent-frontend-sync-20260613.md` 섹션 4  

| 화면/API 영역 | Frontend API | 주요 테이블 | 프론트 노출/사용 |
|---|---|---|---|
| 메인 대시보드 KPI | `/v1/dashboard/kpi`, `/v1/dashboard/trends` | metric/master 계열 | KPI 카드, 추이 |
| 메인 병목 위험 알림 | `/v1/dashboard/risk-alerts` | `tt_bottleneck_case`, `th_agent_step_log`, `td_cause_analysis` | 위험도, 병목확률, 현재 step, 영향 TG 수, 원인 |
| 공정 상태맵 | `/v1/dashboard/process-map`, `/v1/monitoring/bottleneck/process-map` | `td_bottleneck_snapshot`, metric/master 계열 | area/TG별 risk count, utilization, WIP |
| 병목 모니터링 알림 | `/v1/monitoring/bottleneck/alerts` | `tt_bottleneck_case`, `th_agent_step_log`, `td_cause_analysis` | case list, selector row |
| 병목 대응 센터 case | `/v1/response-center/cases` | `tt_bottleneck_case`, `th_agent_step_log` | case 목록, 진행률 |
| 원인 분석 탭 | `/v1/response-center/cases/{caseId}/cause-analysis` | `td_cause_analysis`, `tt_bottleneck_case`, `tm_tool_group` | 원인 타입, SHAP, 확산 TG |
| 대응안 탭 | `/v1/response-center/cases/{caseId}/action-plans` | `td_action_plan`, `th_hitl_decision`, `ps_fab_metrics` | 후보안, baseline, HITL 상태 |
| 리포트 탭 | `/v1/response-center/cases/{caseId}/report` | `td_response_report` | `report_json` 우선 렌더링, legacy fallback |
| 리포트 Q&A | `/v1/chatbot/messages` | `td_response_report`, `tt_chat_session`, `td_chat_message` | `caseId/reportId` 전달 → Spring이 report context 조회 |
| Admin MLflow | `/v1/admin/mlflow/models`, `/v1/admin/mlflow/drift-alerts` | `th_ml_model_version`, `th_drift_alert` | 모델 상태, drift detail, promote/retrain 요청 |
| Admin 임계값 | `/v1/admin/thresholds` | `tm_threshold_config`, `th_threshold_history` | 임계값 목록/수정 이력 |
| Admin MES 인터페이스 | `/v1/admin/mes/mappings`, `/v1/admin/mes/collect-jobs` | `tm_mes_field_mapping`, `tb_mes_collect_job` | field mapping, 수집 job 상태 |
| Admin 권한 관리 | `/v1/admin/access/users` | `tm_app_user`, `tm_role`, `tm_fab` | 사용자/권한 조회 |
| 챗봇 이력 | `/v1/chatbot/sessions`, `/v1/chatbot/sessions/{sessionId}/messages` | `tt_chat_session`, `td_chat_message` | 대화 목록, 메시지, UI 카드, 경고/도구 메타 |
| 알림 | `/v1/notifications`, `/v1/notifications/stream` | `tt_notification`, `th_notification_read` | 헤더 알림, SSE toast, 읽음 상태 |

### 리포트 Q&A 데이터 흐름

프론트는 리포트 전문 JSON/Markdown을 AI에게 직접 보내지 않는다.

```
1. 프론트 → Spring /v1/chatbot/messages (contextReportId 또는 contextCaseId 전달)
2. Spring ChatbotService → td_response_report 조회 (report_json, rendered_markdown)
3. Spring이 AI-Agent chat context 구성
4. AI-Agent: report_json을 수치·판단의 SoT로 사용, rendered_markdown은 설명 보조
5. 답변 → td_chat_message 저장 (ui_card, warnings, tools_used 포함)
```

---

## 작업 우선순위 제안

> 모든 분석 섹션(1~13)을 읽은 뒤 이 표를 기준으로 작업 순서를 결정한다.
> **최종 갱신: 2026-06-14 — 전체 항목 완료**

| 우선순위 | 항목 | 근거 섹션 | 상태 |
|---|---|---|---|
| 🔴 즉시 | `ps_tg_metrics.bottleneck_prob` / `risk_grade` UPDATE 로직 추가 — 신규 DAO `TgMetricsBottleneckWriter` + `BottleneckDetectionService` 수정 (Spring Boot) | §3 | ✅ 완료 (V12 이전, Spring Boot 코드) |
| 🔴 즉시 | `td_cause_analysis.bottleneck_cause_type` 코드 수정 — 잘못된 값(KPI명) 적재 중 | §5 | ✅ 완료 (`judgment.primary_category` 사용) |
| 🔴 즉시 | `td_cause_analysis.diffusion_affected_tg_ids` 코드 수정 — upstream·downstream 의미 역전 | §5 | ✅ 완료 (`alert.impact.affected_tgs` 파라미터 전달) |
| 🟡 단기 | `td_action_plan` 컬럼 재설계 + `ActionPlanRepository.bulk_insert()` 코드 수정 | §4 | ✅ 완료 (V12 마이그레이션 + bulk_insert kpi_stats 추출) |
| 🟡 단기 | `td_response_report` 컬럼 재설계: `report_html` DROP + `report_json`·`rendered_markdown`·`report_schema_version` ADD + 코드 수정 | §6 | ✅ 완료 (V13 마이그레이션 + ResponseReportRepository 재작성) |
| 🟡 단기 | `th_hitl_decision` 스냅샷 컬럼 추가 (`decided_by_name`, `selected_plan_title`, `selected_plan_type`, `agent_sync_status`) + `created_at` DROP | §10 | ✅ 완료 (V14 마이그레이션 + HitlDecision/BncService 수정) |
| 🟡 단기 | `tt_fab_briefing` / `tt_period_report` 신규 테이블 생성 — 프론트 요청 | §11 | ✅ 완료 (V15 마이그레이션) |
| 🟡 단기 | `td_chat_message` 컬럼 3개 추가 (`ui_card`, `warnings`, `tools_used`) — 프론트 요청 | §11 | ✅ 완료 (V16 마이그레이션) |
| 🟡 단기 | `th_drift_alert.detail JSONB` 추가 — 프론트·MLOps 요청 | §11 | ✅ 완료 (V16 마이그레이션) |
| 🟢 중기 | `td_cause_analysis` JSONB 컬럼 추가 (`cause_judgment_json`, `consensus_json`, `trend_json`, `primary_cause_feature`) | §5 | ✅ 완료 (V17 마이그레이션 + CauseAnalysisRepository upsert 반영) |
| 🔵 낮음 | `th_agent_step_log.model_version_id` 연결 — MLOps 추적용 | §7 | ✅ 완료 (AgentStepRepository.mark_done() 파라미터 추가 + agent_service.py MlModelRepository 조회) |
| 🔵 낮음 | `th_audit_log` DROP 마이그레이션 | §9 | ✅ 완료 (V18 마이그레이션) |
| 🔵 낮음 | `tm_knowledge_document` DROP 마이그레이션 | §9 | ✅ 완료 (V19 마이그레이션) |
