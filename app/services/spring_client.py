import logging
from datetime import UTC, datetime
from uuid import UUID

import httpx

from app.config import Settings, get_settings
from app.repositories.agent_step_repository import STEP_NAME_MAP, STEP_ORDER_MAP

logger = logging.getLogger(__name__)
_spring_client: "SpringClient | None" = None


class SpringClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = httpx.AsyncClient(
            base_url=self._settings.spring_base_url,
            headers={
                "X-Internal-Token": self._settings.internal_api_token,
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self._settings.agent_step_timeout_sec),
        )

    async def notify_agent_step(
        self,
        case_id: UUID,
        node_name: str,
        summary: str | None,
        status: str = "DONE",
    ) -> None:
        step_name = STEP_NAME_MAP.get(node_name, node_name)
        payload = {
            "caseId": str(case_id),
            "stepName": step_name,
            "stepOrder": STEP_ORDER_MAP.get(step_name),
            "status": status,
            "outputSummary": summary,
            "completedAt": datetime.now(UTC).isoformat(),
        }
        try:
            await self._post("/api/internal/agent-step", payload)
        except Exception as exc:
            logger.warning("Agent step 알림 전송 실패: %s", exc)

    async def request_snapshot(
        self,
        case_id: UUID,
        tg_id: UUID,
        bottleneck_prob: float,
        risk_grade: str,
        detected_at: datetime,
        simulation_tick: int | None = None,
    ) -> None:
        await self._post(
            "/api/internal/snapshot",
            {
                "caseId": str(case_id),
                "tgId": str(tg_id),
                "bottleneckProb": bottleneck_prob,
                "riskGrade": risk_grade,
                "detectedAt": detected_at.isoformat(),
                "simulationTick": simulation_tick,
            },
        )

    async def send_notification(
        self,
        notification_type: str,
        title: str,
        message: str,
    ) -> None:
        try:
            await self._post(
                "/api/internal/notification",
                {"type": notification_type, "title": title, "message": message},
            )
        except Exception as exc:
            logger.warning("내부 알림 전송 실패: %s", exc)

    async def _post(self, path: str, payload: dict) -> None:
        response = await self._client.post(
            path,
            json=payload,
            headers={"X-Event-Timestamp": datetime.now(UTC).isoformat()},
        )
        response.raise_for_status()

    async def close(self) -> None:
        await self._client.aclose()


def get_spring_client(settings: Settings | None = None) -> SpringClient:
    global _spring_client
    if _spring_client is None:
        _spring_client = SpringClient(settings)
    return _spring_client


async def close_spring_client() -> None:
    global _spring_client
    if _spring_client is not None:
        await _spring_client.close()
        _spring_client = None
