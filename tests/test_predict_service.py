from concurrent.futures import ThreadPoolExecutor

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
        max_util=0.9,
    )

    results = PredictService().predict_all([kpi])

    assert len(results) == 1
    assert 0 <= results[0].probability <= 1
    assert len(results[0].shap_top) == 3


def test_predict_service_dependency_reuses_instance() -> None:
    assert get_predict_service() is get_predict_service()


def test_predict_service_loads_model_once_across_threads(monkeypatch) -> None:
    load_count = 0

    class FakeBooster:
        def load_model(self, _: object) -> None:
            nonlocal load_count
            load_count += 1

    monkeypatch.setattr("app.services.predict_service.xgb.Booster", FakeBooster)
    service = PredictService()

    with ThreadPoolExecutor(max_workers=4) as executor:
        boosters = list(executor.map(lambda _: service._load_model(), range(8)))

    assert load_count == 1
    assert all(booster is boosters[0] for booster in boosters)
