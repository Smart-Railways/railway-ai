from src.services.ml_engine import RailwayMLEngine


def test_ml_engine_health():
    engine = RailwayMLEngine()

    result = engine.health()

    assert result["status"] == "ok"
    assert result["service"] == "railway-ml-engine"
    assert result["version"] == "0.1.0"
