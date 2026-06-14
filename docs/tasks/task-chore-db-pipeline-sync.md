# Task Chore. DB ↔ Agent 파이프라인 스키마 동기화

> 브랜치: `chore/#이슈번호-db-pipeline-sync`
> 상태: 완료
> 작업자: epk4429
> 완료일: 2026-06-14

---

## 구현 목표

`docs/db-pipeline-gap-analysis.md` 분석 기반으로 AI-Agent Python 코드를 Spring Boot DB 스키마 변경(V12~V16)과 동기화하고, 파이프라인 각 노드에서 DB에 올바른 데이터를 저장하도록 버그를 수정한다.

---

## 작업 내용

### repositories/cause_analysis_repository.py

- [x] `cause_type` 오류 수정: `report.shap_top[0].feature` → `report.judgment.primary_category`
- [x] `diffusion_affected_tg_ids` 오류 수정: `upsert()` 파라미터에 `affected_tgs: list[str] | None = None` 추가, `report.upstream_suspects` 대신 `affected_tgs` 사용
- [x] `shap_features importance` 오류 수정: `abs(feature.shap_value)` → `feature.shap_value` (부호 보존)

### repositories/action_plan_repository.py

- [x] `bulk_insert()` kpi_stats 컬럼 추가: est_util_delta, est_q_time_delta, est_wip_delta, est_wait_ratio_delta, sim_paired_n, sim_paired_p_value
- [x] 각 candidate의 `kpi_stats` 딕셔너리에서 값을 추출하여 INSERT

### repositories/response_report_repository.py

- [x] `insert()` 컬럼 교체: report_html/timeline_json 제거, rendered_markdown/report_json/report_schema_version 추가
- [x] `_rendered_markdown()` 헬퍼: output_path 파일 읽기
- [x] `_report_json()` 헬퍼: json_output_path 파일 파싱
- [x] `_root_cause_text()` 헬퍼: `report_json["cause"]["summary"]` 추출
- [x] `_action_comparison_text()` 헬퍼: `report_json["actions"]["recommendation"]["headline"] + primary_reason` 조합

### services/agent_service.py

- [x] `cause` 노드: alerts에서 matching alert 찾아 `alert.impact.affected_tgs` → `cause_repo.upsert(affected_tgs=...)` 전달
- [x] `solution` 노드: `bulk_insert` 호출 제거 (compare_rank로 이동)
- [x] `compare_rank` 노드: `compare_inputs[0]["action_candidates"]`(비 baseline) kpi_stats를 `solution_candidates`에 인덱스별 머지 후 `bulk_insert` 호출
- [x] `_index_report_to_qdrant`: `_report_text()` → `_rendered_markdown()` 참조 수정
