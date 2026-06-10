from agents.schemas.kpi import ToolGroupKPI
from app.api.deps import get_predict_service
from app.services.predict_service import PredictService


def test_predict_service_loads_model_and_returns_shap() -> None:
    kpi = ToolGroupKPI(
        toolgroup="Diffusion_FE_127",
        snapshot_time=120.0,
        available_tool_ratio=1.0,
        q_time_min=20.0,
        wait_ratio=0.5,
        wip=10.0,
        setup_ratio_avg=0.1,
        utilization_avg=0.8,
        max_avg_q_time=30.0,
        max_util=0.9,
    )

    results = PredictService().predict_all([kpi])

    assert len(results) == 1
    assert 0 <= results[0].probability <= 1
    assert len(results[0].shap_top) == 3


def test_predict_service_dependency_reuses_instance() -> None:
    assert get_predict_service() is get_predict_service()
