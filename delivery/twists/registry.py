"""Central registry and factory for all environment twists (§7.1, §7.2).

Registers all 8 standard twists from oel_instance.py plus the custom third twist
(fragile_parcel), and provides the deterministic build_twist_stack factory.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Union

from delivery.twists.base import BaseTwist, TwistStack
from delivery.twists.fog import FogTwist
from delivery.twists.fragile import FragileParcelTwist
from delivery.twists.one_way import OneWayTwist
from delivery.twists.stubs import (
    FuelCapacityTwist,
    RivalStealsParcelsTwist,
    RushHourTwist,
    TeleportTunnelsTwist,
    TollGatesTwist,
    WindDriftTwist,
)

TWIST_REGISTRY: Dict[str, Type[BaseTwist]] = {
    "fuel_limit_with_charging_stations": FuelCapacityTwist,
    "one_way_streets": OneWayTwist,
    "rush_hour_time_varying_costs": RushHourTwist,
    "fog_partial_observability": FogTwist,
    "toll_gates_budget": TollGatesTwist,
    "teleport_tunnels": TeleportTunnelsTwist,
    "wind_drift": WindDriftTwist,
    "rival_steals_parcels": RivalStealsParcelsTwist,
    "fragile_parcel": FragileParcelTwist,
}


def register_twist(name: str, cls: Type[BaseTwist]) -> None:
    """Registers or overrides a twist class in the global registry."""
    TWIST_REGISTRY[name] = cls


def get_twist_class(name: str) -> Type[BaseTwist]:
    """Retrieves twist class by canonical identifier."""
    if name not in TWIST_REGISTRY:
        raise KeyError(
            f"Unknown twist '{name}'. Available twists: {sorted(TWIST_REGISTRY.keys())}"
        )
    return TWIST_REGISTRY[name]


def list_available_twists() -> List[str]:
    """Returns sorted list of registered twist names."""
    return sorted(TWIST_REGISTRY.keys())


def build_twist_stack(
    instance: Any,
    world: Any,
    enable_third_twist: bool = False,
    extra_twists: Optional[List[str]] = None,
) -> TwistStack:
    """Builds and initializes the composed TwistStack for an instance and world.

    Parameters:
    - instance: Instance object or dictionary with 'twists' attribute/key.
    - world: GridWorld instance.
    - enable_third_twist: If True, includes 'fragile_parcel' in active twists.
    - extra_twists: Optional list of additional twist names to enable.

    Returns:
    - Initialized TwistStack.
    """
    raw_twists = getattr(instance, "twists", None)
    if raw_twists is None and isinstance(instance, dict):
        raw_twists = instance.get("twists", [])
    elif raw_twists is None:
        raw_twists = []

    names_to_load = list(raw_twists)

    if enable_third_twist and "fragile_parcel" not in names_to_load:
        names_to_load.append("fragile_parcel")

    if extra_twists:
        for extra in extra_twists:
            if extra not in names_to_load:
                names_to_load.append(extra)

    twists: List[BaseTwist] = []
    for name in names_to_load:
        cls = get_twist_class(name)
        twist_obj = cls()
        twist_obj.initialize(instance, world)
        twists.append(twist_obj)

    return TwistStack(twists)
