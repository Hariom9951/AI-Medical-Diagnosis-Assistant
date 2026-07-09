"""Pytest configuration for integration tests."""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Automatically skip real-network integration tests in CI mock mode."""
    ci_mock = os.getenv("CI_MOCK_DOWNLOADER", "false").lower() == "true"
    if ci_mock:
        skip_real = pytest.mark.skip(reason="CI_MOCK_DOWNLOADER=true: skipping real network tests")
        for item in items:
            if "integration" in item.keywords and "slow" in item.keywords:
                item.add_marker(skip_real)
