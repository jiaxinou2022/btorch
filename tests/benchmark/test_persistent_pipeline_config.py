"""Test the persistent pipeline's environment configuration contract."""

import pytest

from btorch.backend.persistent.plain_version import loader


PIPELINE_ENVIRONMENT_VARIABLES = (
    "BTORCH_PIPELINE_ROLE_RATIO",
    "BTORCH_PIPELINE_DEDICATED_WARPS",
    "BTORCH_PIPELINE_HELPER_WARPS",
    "BTORCH_PIPELINE_TICKET_CHUNK",
    "BTORCH_PIPELINE_STATIC_WAVES",
    "BTORCH_PIPELINE_BINNED_THRESHOLD",
    "BTORCH_PIPELINE_LOW_SUBWARP_SIZE",
    "BTORCH_PIPELINE_HIGH_LOW_RATIO",
    "BTORCH_PIPELINE_TIMING",
    "BTORCH_PIPELINE_COMPONENT_TIMING",
)


def test_pipeline_scan_winners_are_defaults(monkeypatch):
    """Keep the 5090 FlyBrain scan winners aligned across loader defaults."""

    # Clearing every tuning variable exercises the settings users receive
    # when they enable the pipeline without carrying over a prior experiment.
    for name in PIPELINE_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    assert loader._pipeline_role_ratio() == "7:1"
    assert loader._pipeline_consumer_warps(
        "BTORCH_PIPELINE_DEDICATED_WARPS", 1
    ) == 1
    assert loader._pipeline_consumer_warps(
        "BTORCH_PIPELINE_HELPER_WARPS", 1
    ) == 1
    assert loader._pipeline_ticket_chunk() == 1
    assert loader._pipeline_static_waves() == 0
    assert loader._pipeline_binned_threshold() == 512
    assert loader._pipeline_low_subwarp_size() == 8
    assert loader._pipeline_high_low_ratio() == 2
    assert loader._pipeline_timing() is False
    assert loader._pipeline_component_timing() is False


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("BTORCH_PIPELINE_DEDICATED_WARPS", "3"),
        ("BTORCH_PIPELINE_HELPER_WARPS", "16"),
    ],
)
def test_pipeline_rejects_unsupported_consumer_warps(
    monkeypatch, name, value
):
    """Reject warp counts outside the kernel's supported tuning points."""

    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        loader._pipeline_consumer_warps(name, 1)


@pytest.mark.parametrize("waves", (0, 1, 2, 4))
def test_pipeline_accepts_static_wave_scan_points(monkeypatch, waves):
    """Accept every static prefix length exercised by the tuning scan."""

    monkeypatch.setenv("BTORCH_PIPELINE_STATIC_WAVES", str(waves))
    assert loader._pipeline_static_waves() == waves


@pytest.mark.parametrize("waves", (-1, 3, 8))
def test_pipeline_rejects_unsupported_static_waves(monkeypatch, waves):
    """Reject static prefixes that do not have a validated scheduling path."""

    monkeypatch.setenv("BTORCH_PIPELINE_STATIC_WAVES", str(waves))
    with pytest.raises(ValueError, match="BTORCH_PIPELINE_STATIC_WAVES"):
        loader._pipeline_static_waves()


@pytest.mark.parametrize("threshold", (0, 128, 256, 512))
def test_pipeline_accepts_binned_thresholds(monkeypatch, threshold):
    monkeypatch.setenv("BTORCH_PIPELINE_BINNED_THRESHOLD", str(threshold))
    assert loader._pipeline_binned_threshold() == threshold


@pytest.mark.parametrize("threshold", (-1, 64, 1024))
def test_pipeline_rejects_unsupported_binned_thresholds(
    monkeypatch, threshold
):
    monkeypatch.setenv("BTORCH_PIPELINE_BINNED_THRESHOLD", str(threshold))
    with pytest.raises(ValueError, match="BTORCH_PIPELINE_BINNED_THRESHOLD"):
        loader._pipeline_binned_threshold()


@pytest.mark.parametrize("size", (4, 8, 16))
def test_pipeline_accepts_low_subwarp_sizes(monkeypatch, size):
    monkeypatch.setenv("BTORCH_PIPELINE_LOW_SUBWARP_SIZE", str(size))
    assert loader._pipeline_low_subwarp_size() == size


@pytest.mark.parametrize("size", (2, 12, 32))
def test_pipeline_rejects_unsupported_low_subwarp_sizes(monkeypatch, size):
    monkeypatch.setenv("BTORCH_PIPELINE_LOW_SUBWARP_SIZE", str(size))
    with pytest.raises(ValueError, match="BTORCH_PIPELINE_LOW_SUBWARP_SIZE"):
        loader._pipeline_low_subwarp_size()


@pytest.mark.parametrize("ratio", (1, 2, 4))
def test_pipeline_accepts_high_low_ratios(monkeypatch, ratio):
    monkeypatch.setenv("BTORCH_PIPELINE_HIGH_LOW_RATIO", str(ratio))
    assert loader._pipeline_high_low_ratio() == ratio


@pytest.mark.parametrize("ratio", (0, 3, 8))
def test_pipeline_rejects_unsupported_high_low_ratios(monkeypatch, ratio):
    monkeypatch.setenv("BTORCH_PIPELINE_HIGH_LOW_RATIO", str(ratio))
    with pytest.raises(ValueError, match="BTORCH_PIPELINE_HIGH_LOW_RATIO"):
        loader._pipeline_high_low_ratio()
