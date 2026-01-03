"""
src/spaceflight - Spaceflight Telemetry Compression Modules

Physics-based telemetry compression with gap reconstruction for spaceflight.
CLAUDEME v3.1 Compliant: All functions emit receipts.

Modules:
    telemetry_ingest: Standardized ingest from CSV/JSON sources
    physics_discovery: Discover physics laws from telemetry data
    gap_reconstruct: Reconstruct missing telemetry via discovered physics
    orbital_laws: Orbital mechanics specific physics models
"""

from .telemetry_ingest import (
    ingest_telemetry,
    normalize_units,
    TelemetryStream,
)
from .physics_discovery import (
    discover_law,
    compress_trajectory,
    PhysicsLaw,
)
from .gap_reconstruct import (
    reconstruct_gap,
    validate_reconstruction,
    ReconstructionResult,
)

__all__ = [
    # Ingest
    "ingest_telemetry",
    "normalize_units",
    "TelemetryStream",
    # Physics discovery
    "discover_law",
    "compress_trajectory",
    "PhysicsLaw",
    # Gap reconstruction
    "reconstruct_gap",
    "validate_reconstruction",
    "ReconstructionResult",
]
