# AI-Agent Dev Log

> Task가 완료될 때마다 아래 형식으로 항목을 추가한다.
> 같은 날 여러 Task를 완료하면 같은 날짜 아래에 이어서 작성한다.

---

### 2026-06-14

**오늘 한 일**
- **[CHORE] DB ↔ Agent 파이프라인 스키마 동기화**
  - `CauseAnalysisRepository.upsert()`: cause_type 소스를 `report.judgment.primary_category`로 수정, diffusion affected_tgs를 upstream_suspects 대신 파라미터로 수신, shap_value abs() 제거
  - `ActionPlanRepository.bulk_insert()`: V12 신규 kpi 컬럼(est_util_delta/est_q_time_delta/est_wip_delta/est_wait_ratio_delta/sim_paired_n/sim_paired_p_value) INSERT/UPDATE 추가
  - `ResponseReportRepository.insert()`: report_html/timeline_json 제거, rendered_markdown/report_json/report_schema_version으로 교체. json_output_path 파일 파싱으로 ReportV2 JSON 저장. cause.summary → root_cause_text, actions.recommendation → action_comparison_text
  - `agent_service.py`: cause 노드에서 alerts→affected_tgs 추출 전달, solution 노드에서 bulk_insert 제거, compare_rank 노드에서 kpi_stats 머지 후 bulk_insert 호출

  - **[CHORE] 중기·낮음 우선순위 DB 동기화 작업**
  - `CauseAnalysisRepository.upsert()`: V17 신규 컬럼 4개(primary_cause_feature/cause_judgment_json/consensus_json/trend_json) INSERT/UPDATE 추가. `report.judgment.primary_cause`→primary_cause_feature, `report.judgment.model_dump()`→cause_judgment_json, `report.consensus.model_dump()`→consensus_json, `[t.model_dump() for t in report.trend_top]`→trend_json
  - `AgentStepRepository.mark_done()`: `model_version_id: UUID | None = None` 파라미터 추가, UPDATE SQL에 `model_version_id = COALESCE($6, model_version_id)` 추가
  - `agent_service.py`: 파이프라인 시작 시 `MlModelRepository.find_active()` 호출해 active_model_version_id 조회. cascade(DIFFUSION_ANALYSIS)/cause(CAUSE_ANALYSIS) 단계의 `_complete_step()` 호출 시 `model_version_id` 전달

**특이사항**
- kpi_stats는 compare_inputs[0]["action_candidates"]의 비 baseline 항목에서 인덱스 순서로 solution_candidates에 병합
- affected_tgs 없으면 빈 리스트로 처리 (기존 _find_tg_ids([]) 동작 유지)
- model_version_id는 ML 모델 사용 단계(cascade/cause)에만 기록; LLM-only 단계(solution/compare/report)는 NULL 유지
