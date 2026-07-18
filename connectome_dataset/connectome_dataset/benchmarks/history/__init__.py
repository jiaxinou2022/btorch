"""Trackable history: normalize native benchmark output, store it, and report.

A thin layer over pytest-benchmark (Python) and Google Benchmark (C++) — it adds
only cross-language unification, a leaderboard, and static reports, never a
re-implementation of timing/stats/storage. See docs/benchmark_design/history_and_reporting.md.

Import submodules explicitly, e.g. ``from connectome_dataset.benchmarks.history import ingest``.
"""
