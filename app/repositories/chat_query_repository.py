"""챗 어시스턴트 도구용 읽기 쿼리 — 추세(ps_tg_metrics)·구역 WIP 현황·병목 케이스(tt_bottleneck_case).
개별 Lot 테이블은 스키마에 없어 'lot 현황'은 구역별 WIP·대기·Q-time 집계로 제공한다."""

from uuid import UUID

import asyncpg


class ChatQueryRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def kpi_trend(
        self, fab_id: UUID, area: str, hours: int, bucket_min: int, group_by: str = "area",
    ) -> list[asyncpg.Record]:
        """구역/TG 필터에 대한 KPI 시계열(time_bucket 평균). 기준시각은 테이블 최신 measured_at.
        group_by='tg'면 툴그룹 단위(label=구역/TG코드), 그 외 구역 단위로 집계한다."""
        group_by = group_by if group_by in ("area", "tg") else "area"
        return await self._pool.fetch(
            """
            WITH ref AS (SELECT MAX(measured_at) AS now_ts FROM ps_tg_metrics)
            SELECT
                time_bucket(make_interval(mins => $4), m.measured_at) AS bucket,
                CASE WHEN $5 = 'tg' THEN a.area_name || '/' || tg.tg_code
                     ELSE a.area_name END AS label,
                AVG(m.wip_count)        AS wip,
                AVG(m.utilization_rate) AS util,
                AVG(m.avg_qtime_min)    AS qtime,
                AVG(m.wait_ratio)       AS wait
            FROM ps_tg_metrics m
            JOIN tm_tool_group tg ON tg.tg_id = m.tg_id
            JOIN tm_area a ON a.area_id = tg.area_id
            CROSS JOIN ref
            WHERE a.fab_id = $1
              AND tg.is_active = TRUE
              AND ($2 = '' OR a.area_name ILIKE '%'||$2||'%'
                   OR tg.tg_code ILIKE '%'||$2||'%' OR tg.tg_name ILIKE '%'||$2||'%')
              AND m.measured_at >= ref.now_ts - make_interval(hours => $3)
            GROUP BY bucket, label
            ORDER BY bucket
            """,
            fab_id, area, hours, bucket_min, group_by,
        )

    _TG_RANK_METRICS = {
        "util": "utilization_rate",
        "wip": "wip_count",
        "qtime": "avg_qtime_min",
        "wait": "wait_ratio",
        "bottleneck": "bottleneck_prob",
    }

    async def top_toolgroups(
        self, fab_id: UUID, area: str, metric: str, order: str, limit: int,
    ) -> list[asyncpg.Record]:
        """툴그룹(TG) 단위 최신 스냅샷을 지표 기준으로 정렬해 상위 N개 반환.
        '가동률 가장 높은/낮은 TG' 같은 순위 질문용. metric/order는 화이트리스트로만 치환."""
        column = self._TG_RANK_METRICS.get(metric, "utilization_rate")
        direction = "ASC" if order == "asc" else "DESC"
        return await self._pool.fetch(
            f"""
            WITH ref AS (SELECT MAX(measured_at) AS now_ts FROM ps_tg_metrics),
            latest AS (
                SELECT DISTINCT ON (m.tg_id)
                    a.area_name, tg.tg_code, tg.tg_name,
                    m.utilization_rate, m.wip_count, m.avg_qtime_min, m.wait_ratio,
                    m.available_tool_ratio, m.bottleneck_prob, m.risk_grade, m.measured_at
                FROM ps_tg_metrics m
                JOIN tm_tool_group tg ON tg.tg_id = m.tg_id
                JOIN tm_area a ON a.area_id = tg.area_id
                CROSS JOIN ref
                WHERE a.fab_id = $1
                  AND tg.is_active = TRUE
                  AND m.measured_at >= ref.now_ts - interval '15 minutes'
                  AND ($2 = '' OR a.area_name ILIKE '%'||$2||'%')
                ORDER BY m.tg_id, m.measured_at DESC
            )
            SELECT * FROM latest
            ORDER BY {column} {direction} NULLS LAST
            LIMIT $3
            """,
            fab_id, area, limit,
        )

    async def case_detail(self, fab_id: UUID, ref: str) -> asyncpg.Record | None:
        """병목 케이스 1건 상세(원인 분석·리포트 요약 포함). ref=''이면 fab 최신 케이스,
        아니면 case_id 접두/TG코드·TG명·구역명 부분일치로 가장 최근 1건."""
        return await self._pool.fetchrow(
            """
            SELECT c.case_id, c.detected_at, c.bottleneck_prob, c.risk_grade, c.status, c.resolved_at,
                   a.area_name, tg.tg_code, tg.tg_name,
                   ca.shap_features, ca.bottleneck_cause_type, ca.diffusion_affected_tg_ids,
                   ca.model_accuracy, ca.model_f1,
                   r.summary AS report_summary, r.root_cause_text
            FROM tt_bottleneck_case c
            JOIN tm_tool_group tg ON tg.tg_id = c.tg_id
            JOIN tm_area a ON a.area_id = tg.area_id
            LEFT JOIN td_cause_analysis ca ON ca.case_id = c.case_id
            LEFT JOIN td_response_report r ON r.case_id = c.case_id
            WHERE a.fab_id = $1
              AND ($2 = '' OR c.case_id::text ILIKE $2 || '%'
                   OR tg.tg_code ILIKE '%'||$2||'%' OR tg.tg_name ILIKE '%'||$2||'%'
                   OR a.area_name ILIKE '%'||$2||'%')
            ORDER BY c.detected_at DESC
            LIMIT 1
            """,
            fab_id, ref,
        )

    async def case_plans(self, case_id: UUID) -> list[asyncpg.Record]:
        """케이스의 AI 대응안 목록(예상/실측 KPI 델타) + 최신 HITL 결정에서 선택 여부."""
        return await self._pool.fetch(
            """
            SELECT p.plan_seq, p.plan_type, p.plan_title, p.plan_detail,
                   p.est_throughput_delta, p.est_avg_wait_delta,
                   p.est_delivery_compliance_delta, p.est_delay_delta,
                   p.actual_throughput_delta, p.actual_avg_wait_delta, p.validated_at,
                   COALESCE(p.plan_id = h.selected_plan_id, FALSE) AS selected
            FROM td_action_plan p
            LEFT JOIN LATERAL (
                SELECT selected_plan_id FROM th_hitl_decision d
                WHERE d.case_id = p.case_id ORDER BY d.re_decision_seq DESC LIMIT 1
            ) h ON TRUE
            WHERE p.case_id = $1
            ORDER BY p.plan_seq
            """,
            case_id,
        )

    async def case_hitl(self, case_id: UUID) -> list[asyncpg.Record]:
        """케이스의 HITL 결정 이력(최신순)."""
        return await self._pool.fetch(
            """
            SELECT re_decision_seq, decision, comment, decided_at
            FROM th_hitl_decision
            WHERE case_id = $1
            ORDER BY re_decision_seq DESC
            """,
            case_id,
        )

    async def tool_activity(self, fab_id: UUID, tool: str, hours: int, bucket_min: int) -> list[asyncpg.Record]:
        """특정 툴(설비)의 최근 활동 시계열(가동률·대기 Lot·Q-time·다운율 time_bucket 평균)."""
        return await self._pool.fetch(
            """
            WITH ref AS (SELECT MAX(measured_at) AS now_ts FROM ps_tool_metrics)
            SELECT time_bucket(make_interval(mins => $4), m.measured_at) AS bucket,
                   t.tool_code,
                   AVG(m.utilization_rate) AS util,
                   AVG(m.queue_lot_count)  AS queue,
                   AVG(m.avg_qtime_min)    AS qtime,
                   AVG(m.down_ratio)       AS down
            FROM ps_tool_metrics m
            JOIN tm_tool t ON t.tool_id = m.tool_id
            JOIN tm_tool_group tg ON tg.tg_id = t.tg_id
            JOIN tm_area a ON a.area_id = tg.area_id
            CROSS JOIN ref
            WHERE a.fab_id = $1
              AND t.is_active = TRUE
              AND ($2 = '' OR t.tool_code ILIKE '%'||$2||'%' OR t.tool_name ILIKE '%'||$2||'%')
              AND m.measured_at >= ref.now_ts - make_interval(hours => $3)
            GROUP BY bucket, t.tool_code
            ORDER BY bucket
            """,
            fab_id, tool, hours, bucket_min,
        )

    async def tool_status(self, fab_id: UUID, tg: str) -> list[asyncpg.Record]:
        """특정 툴그룹(또는 구역/툴코드)에 속한 개별 툴들의 최신 현황(ps_tool_metrics)."""
        return await self._pool.fetch(
            """
            WITH ref AS (SELECT MAX(measured_at) AS now_ts FROM ps_tool_metrics)
            SELECT DISTINCT ON (m.tool_id)
                t.tool_code, t.tool_name, tg.tg_code, a.area_name,
                m.utilization_rate, m.oee_estimate, m.avg_qtime_min,
                m.queue_lot_count, m.setup_ratio, m.down_ratio, m.measured_at
            FROM ps_tool_metrics m
            JOIN tm_tool t ON t.tool_id = m.tool_id
            JOIN tm_tool_group tg ON tg.tg_id = t.tg_id
            JOIN tm_area a ON a.area_id = tg.area_id
            CROSS JOIN ref
            WHERE a.fab_id = $1
              AND t.is_active = TRUE
              AND m.measured_at >= ref.now_ts - interval '15 minutes'
              AND ($2 = '' OR tg.tg_code ILIKE '%'||$2||'%' OR tg.tg_name ILIKE '%'||$2||'%'
                   OR t.tool_code ILIKE '%'||$2||'%' OR a.area_name ILIKE '%'||$2||'%')
            ORDER BY m.tool_id, m.measured_at DESC
            """,
            fab_id, tg,
        )

    async def lot_status(self, fab_id: UUID, area: str) -> list[asyncpg.Record]:
        """구역별 최신 WIP·대기·Q-time (개별 Lot 대신 가용한 실데이터 집계)."""
        return await self._pool.fetch(
            """
            SELECT DISTINCT ON (m.tg_id)
                a.area_name, tg.tg_code, tg.tg_name,
                m.wip_count, m.wait_ratio, m.avg_qtime_min, m.utilization_rate, m.measured_at
            FROM ps_tg_metrics m
            JOIN tm_tool_group tg ON tg.tg_id = m.tg_id
            JOIN tm_area a ON a.area_id = tg.area_id
            WHERE a.fab_id = $1
              AND tg.is_active = TRUE
              AND ($2 = '' OR a.area_name ILIKE '%'||$2||'%'
                   OR tg.tg_code ILIKE '%'||$2||'%' OR tg.tg_name ILIKE '%'||$2||'%')
            ORDER BY m.tg_id, m.measured_at DESC
            """,
            fab_id, area,
        )

    async def compare_periods(self, fab_id: UUID, area: str, hours: int) -> list[asyncpg.Record]:
        """최근 N시간 vs 그 직전 N시간 — 구역별 WIP·가동률·Q-time 평균 비교(델타 계산용)."""
        return await self._pool.fetch(
            """
            WITH ref AS (SELECT MAX(measured_at) AS now_ts FROM ps_tg_metrics),
            base AS (
                SELECT a.area_name,
                       CASE WHEN m.measured_at >= ref.now_ts - make_interval(hours => $3)
                            THEN 'recent' ELSE 'previous' END AS period,
                       m.wip_count, m.utilization_rate, m.avg_qtime_min
                FROM ps_tg_metrics m
                JOIN tm_tool_group tg ON tg.tg_id = m.tg_id
                JOIN tm_area a ON a.area_id = tg.area_id
                CROSS JOIN ref
                WHERE a.fab_id = $1
                  AND tg.is_active = TRUE
                  AND ($2 = '' OR a.area_name ILIKE '%'||$2||'%'
                       OR tg.tg_code ILIKE '%'||$2||'%' OR tg.tg_name ILIKE '%'||$2||'%')
                  AND m.measured_at >= ref.now_ts - make_interval(hours => $3 * 2)
            )
            SELECT area_name, period,
                   AVG(wip_count)        AS wip,
                   AVG(utilization_rate) AS util,
                   AVG(avg_qtime_min)    AS qtime
            FROM base
            GROUP BY area_name, period
            ORDER BY area_name, period
            """,
            fab_id, area, hours,
        )

    async def bottleneck_cases(self, fab_id: UUID, area: str, status: str, limit: int) -> list[asyncpg.Record]:
        """최근 병목 케이스(감지확률·위험등급·상태). 구역/상태 필터 가능."""
        return await self._pool.fetch(
            """
            SELECT
                c.case_id, a.area_name, tg.tg_code, tg.tg_name,
                c.detected_at, c.bottleneck_prob, c.risk_grade, c.status, c.resolved_at
            FROM tt_bottleneck_case c
            JOIN tm_tool_group tg ON tg.tg_id = c.tg_id
            JOIN tm_area a ON a.area_id = tg.area_id
            WHERE a.fab_id = $1
              AND ($2 = '' OR a.area_name ILIKE '%'||$2||'%'
                   OR tg.tg_code ILIKE '%'||$2||'%' OR tg.tg_name ILIKE '%'||$2||'%')
              AND ($3 = '' OR c.status = $3)
            ORDER BY c.detected_at DESC
            LIMIT $4
            """,
            fab_id, area, status, limit,
        )
