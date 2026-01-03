"""
src/generalized - Domain-Agnostic Physics Modules

Generalized trajectory physics and atmospheric models for any unpowered
flight regime on any celestial body.

CLAUDEME v3.1 Compliant: All functions emit receipts.

Flight Regimes:
    - Ballistic: No lift, pure drag (Dragon, Soyuz)
    - Lifting: L/D control, bank angle modulation (Starship, Shuttle)
    - Orbital: Kepler dynamics + perturbations (satellites)
    - Hypersonic glide: High L/D, trajectory shaping (X-37B)

Celestial Bodies:
    - Earth: US Standard Atmosphere 1976
    - Mars: Mars-GRAM simplified
    - Custom: User-defined profiles
"""

from .atmospheric_models import (
    atmospheric_density,
    atmospheric_temperature,
    atmospheric_pressure,
    scale_height,
    AtmosphereModel,
    EARTH_ATMOSPHERE,
    MARS_ATMOSPHERE,
)
from .trajectory_laws import (
    TrajectoryLaw,
    BallisticLaw,
    LiftingLaw,
    OrbitalLaw,
    predict_trajectory,
    compress_with_law,
    decompress_with_law,
)
from .validation import (
    compute_rmse,
    compute_metrics,
    validate_against_slo,
    ValidationResult,
)

__all__ = [
    # Atmospheric models
    "atmospheric_density",
    "atmospheric_temperature",
    "atmospheric_pressure",
    "scale_height",
    "AtmosphereModel",
    "EARTH_ATMOSPHERE",
    "MARS_ATMOSPHERE",
    # Trajectory laws
    "TrajectoryLaw",
    "BallisticLaw",
    "LiftingLaw",
    "OrbitalLaw",
    "predict_trajectory",
    "compress_with_law",
    "decompress_with_law",
    # Validation
    "compute_rmse",
    "compute_metrics",
    "validate_against_slo",
    "ValidationResult",
]
