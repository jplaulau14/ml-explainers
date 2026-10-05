from mb_explainer.notes import cache_fit, pace


def test_pace_bands() -> None:
    assert pace(50, 100) == "slower"
    assert pace(100, 100) == "close"
    assert pace(200, 100) == "faster"


def test_cache_fit() -> None:
    assert cache_fit(10, 32) == "fits"
    assert cache_fit(64, 32) == "larger"
