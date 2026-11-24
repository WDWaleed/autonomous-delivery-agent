"""Unit tests for serial-first parallel execution helper (M2.6)."""

import pytest

from delivery.experiments.parallel import map_tasks


def square(x: int) -> int:
    return x * x


def test_map_tasks_serial_execution():
    items = [1, 2, 3, 4, 5]
    results = map_tasks(square, items, workers=1)
    assert results == [1, 4, 9, 16, 25]


def test_map_tasks_empty():
    assert map_tasks(square, [], workers=1) == []
