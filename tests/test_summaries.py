"""Summaries are computed by code, so anomaly detection has one correct answer."""

from __future__ import annotations

from datetime import timedelta

from incident_agent.config import Detection, parse_iso
from incident_agent.tools.schemas import MetricPoint
from incident_agent.tools.summaries import empty_summary, summarize_metrics

DETECTION = Detection()
START = parse_iso("2026-09-22T14:00:00Z")
END = parse_iso("2026-09-22T16:00:00Z")


def points(values: list[float]) -> list[MetricPoint]:
    return [MetricPoint(timestamp=START + timedelta(minutes=i), value=v) for i, v in enumerate(values)]


def summary(values: list[float], metric: str = "error_rate", limit: int = 20,
            detection: Detection = DETECTION) -> str:
    return summarize_metrics("checkout-api", metric, START, END, points(values), limit, detection)[0]


def test_a_spike_reports_its_start_its_peak_and_how_many_points_are_high():
    text = summary([0.008] * 100 + [0.17] * 20)
    assert "spike starts 2026-09-22T15:40:00Z" in text
    assert "peak 0.1700 (17.00%)" in text
    assert "20 of 120 points above threshold" in text


def test_a_flat_series_reports_no_spike():
    assert "no spike detected" in summary([0.008] * 120)


def test_too_few_points_makes_no_spike_claim():
    text = summary([0.008, 0.9, 0.008])
    assert "insufficient metric data" in text
    assert "spike starts" not in text


def test_a_window_that_is_mostly_elevated_refuses_to_report_a_baseline():
    text = summary([0.008] * 20 + [0.17] * 100)
    assert "elevated for most of the window" in text
    assert "not a usable baseline" in text
    assert "wider window" in text
    assert "spike starts" not in text
    assert "no spike detected" not in text


def test_a_window_entirely_inside_an_incident_cannot_be_detected():
    # A documented limitation: with no quiet points to compare against, a constant
    # 17% error rate is indistinguishable from a service whose normal rate is 17%.
    assert "no spike detected" in summary([0.17] * 120)


def test_a_tiny_baseline_does_not_turn_noise_into_a_spike():
    # 0.0005 is five times the baseline but far below the 0.01 minimum difference.
    assert "no spike detected" in summary([0.0001] * 100 + [0.0005] * 20)


def test_latency_uses_its_own_minimum_difference():
    assert "no spike detected" in summary([10] * 100 + [40] * 20, metric="latency_p95_ms")
    assert "spike starts" in summary([10] * 100 + [400] * 20, metric="latency_p95_ms")


def test_truncation_is_disclosed_and_the_summary_still_uses_every_point():
    text, rows = summarize_metrics("checkout-api", "error_rate", START, END,
                                     points([0.008] * 100 + [0.17] * 20), 20, DETECTION)
    assert len(rows) == 20
    assert "Showing 20 of 120 points" in text
    assert "20 of 120 points above threshold" in text  # computed from all 120


def test_no_truncation_notice_when_everything_fits():
    text, rows = summarize_metrics("checkout-api", "error_rate", START, END, points([0.008] * 10), 20, DETECTION)
    assert len(rows) == 10
    assert "Showing" not in text


def test_an_empty_result_is_worded_as_a_fact_about_the_query():
    text = empty_summary("get_deployments", "checkout-api", START, END)
    assert "returned no matching records" in text
    assert "not proof that nothing happened" in text


def test_detection_thresholds_are_configurable():
    values = [0.008] * 100 + [0.02] * 20  # 2.5x the baseline, below the 0.01 minimum difference
    assert "no spike detected" in summary(values)
    sensitive = Detection(spike_multiplier=1.5, min_metric_points=5)
    assert "spike starts" in summary(values, detection=sensitive)


def test_the_minimum_point_count_is_configurable():
    assert "insufficient metric data" in summary([0.008] * 4)
    assert "insufficient" not in summary([0.008] * 4, detection=Detection(min_metric_points=3))
