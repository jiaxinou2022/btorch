"""Tests for the firing-rate activity-sweep driver."""

import csv

import pytest

from benchmark.benchmark_activity_sweep import (
    external_event_rate_for_target,
    firing_rate_hz_to_activity,
    merge_provider_parts,
    valid_provider_part,
)


def test_firing_rate_hz_to_activity_uses_millisecond_timestep():
    """A 1 ms step should map 50 Hz to 0.05 spikes per neuron per step."""

    assert firing_rate_hz_to_activity(50.0, 1.0) == pytest.approx(0.05)
    assert firing_rate_hz_to_activity(0.1, 1.0) == pytest.approx(0.0001)
    assert firing_rate_hz_to_activity(0.0, 1.0) == 0.0


@pytest.mark.parametrize("rate_hz, dt_ms", [(-0.1, 1.0), (1.0, 0.0)])
def test_firing_rate_hz_to_activity_rejects_invalid_values(
    rate_hz: float,
    dt_ms: float,
):
    """Invalid physical values must not silently enter calibration."""

    with pytest.raises(ValueError):
        firing_rate_hz_to_activity(rate_hz, dt_ms)


def test_external_event_rate_is_reduced_only_for_low_targets():
    """Low firing targets need finer input sparsity than the 1% default."""

    assert external_event_rate_for_target(0.0, 0.01) == 0.0
    assert external_event_rate_for_target(0.0001, 0.01) == pytest.approx(0.0001)
    assert external_event_rate_for_target(0.005, 0.01) == pytest.approx(0.005)
    assert external_event_rate_for_target(0.02, 0.01) == pytest.approx(0.01)


def test_merge_provider_parts_preserves_one_row_per_provider(tmp_path):
    """Provider isolation must reconstruct one homogeneous workload CSV."""

    parts = []
    for provider in ("baseline", "method"):
        path = tmp_path / f"{provider}.csv"
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "provider",
                    "latency_ms",
                    "correctness_status",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "provider": provider,
                    "latency_ms": "1.0",
                    "correctness_status": "passed",
                }
            )
        parts.append(path)

    output = tmp_path / "merged.csv"
    merge_provider_parts(parts, output)

    with output.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["provider"] for row in rows] == ["baseline", "method"]
    assert valid_provider_part(parts[0], "baseline")
    assert not valid_provider_part(parts[0], "method")


def test_failed_provider_part_is_not_resumed(tmp_path):
    """A provider error row must be rerun instead of treated as complete."""

    path = tmp_path / "provider.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["provider", "correctness_status"],
        )
        writer.writeheader()
        writer.writerow(
            {"provider": "method", "correctness_status": "error:RuntimeError"}
        )

    assert not valid_provider_part(path, "method")
