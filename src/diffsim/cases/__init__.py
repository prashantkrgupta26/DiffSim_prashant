"""Case-input loaders (config.txt -> typed dataclasses) for immersed-body
flow drivers. Currently: the ThinShell.pdf TRUCK case (truck_config)."""
from .truck_config import load_truck_config, TruckConfig, BodySpec, RegionBox

__all__ = ["load_truck_config", "TruckConfig", "BodySpec", "RegionBox"]
