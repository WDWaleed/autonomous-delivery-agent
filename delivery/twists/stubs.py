"""Stub implementations for secondary/unassigned twists (§7.1 Table 2).

Provides full interface conformance and serialization support for all 8 standard
twists from oel_instance.py, allowing dynamic toggling in live/viva scenarios.
"""

from __future__ import annotations

from typing import Any, Dict

from delivery.twists.base import BaseTwist


class FuelCapacityTwist(BaseTwist):
    """Twist 1: fuel_limit_with_charging_stations."""

    def __init__(self, capacity: int = 50, enabled: bool = True):
        super().__init__(
            name="fuel_limit_with_charging_stations",
            title="Fuel Limit with Charging Stations",
            enabled=enabled,
        )
        self.capacity = capacity
        self.charging_stations = []

    def initialize(self, instance: Any, world: Any) -> None:
        self.capacity = 50
        # Deterministically place 2-3 charging stations
        n = world.n
        self.charging_stations = [(n // 4, n // 4), (3 * n // 4, 3 * n // 4)]

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({"capacity": self.capacity, "charging_stations": self.charging_stations})
        return data


class RushHourTwist(BaseTwist):
    """Twist 3: rush_hour_time_varying_costs."""

    def __init__(self, peak_multiplier: float = 2.0, enabled: bool = True):
        super().__init__(
            name="rush_hour_time_varying_costs",
            title="Rush Hour Time-Varying Costs",
            enabled=enabled,
        )
        self.peak_multiplier = peak_multiplier

    def initialize(self, instance: Any, world: Any) -> None:
        pass

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["peak_multiplier"] = self.peak_multiplier
        return data


class TollGatesTwist(BaseTwist):
    """Twist 5: toll_gates_budget."""

    def __init__(self, budget: int = 20, enabled: bool = True):
        super().__init__(
            name="toll_gates_budget",
            title="Toll Gates Budget",
            enabled=enabled,
        )
        self.budget = budget
        self.toll_cells = []

    def initialize(self, instance: Any, world: Any) -> None:
        self.budget = 20

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({"budget": self.budget, "toll_cells": self.toll_cells})
        return data


class TeleportTunnelsTwist(BaseTwist):
    """Twist 6: teleport_tunnels."""

    def __init__(self, enabled: bool = True):
        super().__init__(
            name="teleport_tunnels",
            title="Teleport Tunnels",
            enabled=enabled,
        )
        self.tunnels = []

    def initialize(self, instance: Any, world: Any) -> None:
        pass

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["tunnels"] = self.tunnels
        return data


class WindDriftTwist(BaseTwist):
    """Twist 7: wind_drift."""

    def __init__(self, drift_probability: float = 0.15, enabled: bool = True):
        super().__init__(
            name="wind_drift",
            title="Wind Drift",
            enabled=enabled,
        )
        self.drift_probability = drift_probability

    def initialize(self, instance: Any, world: Any) -> None:
        pass

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["drift_probability"] = self.drift_probability
        return data


class RivalStealsParcelsTwist(BaseTwist):
    """Twist 8: rival_steals_parcels."""

    def __init__(self, enabled: bool = True):
        super().__init__(
            name="rival_steals_parcels",
            title="Rival Steals Parcels",
            enabled=enabled,
        )

    def initialize(self, instance: Any, world: Any) -> None:
        pass
