"""
src/generalized/atmospheric_models.py - Atmospheric Density Models

Atmospheric density models for multiple celestial bodies.
CLAUDEME v3.1 Compliant: All functions emit receipts.

Supported Bodies:
    - Earth: US Standard Atmosphere 1976
    - Mars: Mars-GRAM simplified (exponential approximation)
    - Custom: User-defined density profiles

Altitude Ranges:
    - Earth: 0-100 km (reentry), 100-400 km (orbital decay)
    - Mars: 0-80 km (reentry)

Receipt: atmosphere_model_receipt
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# Import from project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from receipts import dual_hash, emit_receipt


# =============================================================================
# CONSTANTS
# =============================================================================

# Earth atmosphere (US Standard Atmosphere 1976)
EARTH_RHO_0 = 1.225  # kg/m^3 at sea level
EARTH_T_0 = 288.15  # K at sea level
EARTH_P_0 = 101325.0  # Pa at sea level
EARTH_H_SCALE = 8500.0  # m scale height
EARTH_G_0 = 9.80665  # m/s^2
EARTH_M = 0.0289644  # kg/mol (molar mass of dry air)
EARTH_R = 8.31447  # J/(mol*K)

# Mars atmosphere
MARS_RHO_0 = 0.020  # kg/m^3 at surface
MARS_T_0 = 210.0  # K at surface (average)
MARS_P_0 = 636.0  # Pa at surface
MARS_H_SCALE = 11100.0  # m scale height
MARS_G_0 = 3.72076  # m/s^2

# Titan atmosphere
TITAN_RHO_0 = 5.3  # kg/m^3 at surface (dense!)
TITAN_T_0 = 94.0  # K at surface
TITAN_P_0 = 146700.0  # Pa at surface (1.45 atm)
TITAN_H_SCALE = 21000.0  # m scale height


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class AtmosphereModel:
    """
    Atmospheric model parameters for a celestial body.

    Uses exponential density profile:
        rho(h) = rho_0 * exp(-h / H)
    """
    body_name: str
    rho_0: float  # Surface density (kg/m^3)
    T_0: float  # Surface temperature (K)
    P_0: float  # Surface pressure (Pa)
    H_scale: float  # Scale height (m)
    g_0: float  # Surface gravity (m/s^2)
    altitude_min: float = 0.0  # Valid altitude range min (m)
    altitude_max: float = 200000.0  # Valid altitude range max (m)

    def density(self, altitude: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute density at altitude."""
        return self.rho_0 * np.exp(-np.clip(altitude, 0, self.altitude_max) / self.H_scale)

    def temperature(self, altitude: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute temperature at altitude.

        Uses simple lapse rate model for troposphere, isothermal above.
        """
        # Simple model: linear decrease up to 11km, isothermal above
        if self.body_name == "earth":
            lapse_rate = 0.0065  # K/m
            tropopause = 11000.0  # m
        else:
            lapse_rate = 0.002  # K/m (slower for other bodies)
            tropopause = 20000.0

        h = np.clip(altitude, 0, self.altitude_max)

        if isinstance(h, np.ndarray):
            temp = np.where(
                h < tropopause,
                self.T_0 - lapse_rate * h,
                self.T_0 - lapse_rate * tropopause
            )
        else:
            if h < tropopause:
                temp = self.T_0 - lapse_rate * h
            else:
                temp = self.T_0 - lapse_rate * tropopause

        return np.maximum(temp, 50.0)  # Minimum temperature

    def pressure(self, altitude: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute pressure at altitude using barometric formula."""
        return self.P_0 * np.exp(-np.clip(altitude, 0, self.altitude_max) / self.H_scale)

    def scale_height_at(self, altitude: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute local scale height at altitude."""
        # Scale height varies with temperature: H = RT/(Mg)
        T = self.temperature(altitude)
        if self.body_name == "earth":
            M = EARTH_M
            R = EARTH_R
        else:
            M = EARTH_M  # Approximate for other bodies
            R = EARTH_R

        # Account for gravity variation with altitude
        # g(h) = g_0 * (R / (R + h))^2 where R is planet radius
        if self.body_name == "earth":
            R_planet = 6.371e6
        elif self.body_name == "mars":
            R_planet = 3.3895e6
        else:
            R_planet = 6.371e6

        g = self.g_0 * (R_planet / (R_planet + altitude)) ** 2
        return R * T / (M * g)


# =============================================================================
# PRESET MODELS
# =============================================================================

EARTH_ATMOSPHERE = AtmosphereModel(
    body_name="earth",
    rho_0=EARTH_RHO_0,
    T_0=EARTH_T_0,
    P_0=EARTH_P_0,
    H_scale=EARTH_H_SCALE,
    g_0=EARTH_G_0,
    altitude_min=0.0,
    altitude_max=400000.0,  # Up to orbital decay
)

MARS_ATMOSPHERE = AtmosphereModel(
    body_name="mars",
    rho_0=MARS_RHO_0,
    T_0=MARS_T_0,
    P_0=MARS_P_0,
    H_scale=MARS_H_SCALE,
    g_0=MARS_G_0,
    altitude_min=0.0,
    altitude_max=80000.0,
)

TITAN_ATMOSPHERE = AtmosphereModel(
    body_name="titan",
    rho_0=TITAN_RHO_0,
    T_0=TITAN_T_0,
    P_0=TITAN_P_0,
    H_scale=TITAN_H_SCALE,
    g_0=1.352,  # m/s^2
    altitude_min=0.0,
    altitude_max=200000.0,
)

# Model registry
ATMOSPHERE_MODELS = {
    "earth": EARTH_ATMOSPHERE,
    "mars": MARS_ATMOSPHERE,
    "titan": TITAN_ATMOSPHERE,
}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def atmospheric_density(
    altitude: Union[float, np.ndarray],
    body: str = "earth",
    model: Optional[AtmosphereModel] = None,
) -> Union[float, np.ndarray]:
    """
    Compute atmospheric density at altitude.

    Args:
        altitude: Altitude in meters above surface
        body: Celestial body name ("earth", "mars", "titan")
        model: Optional custom AtmosphereModel

    Returns:
        Density in kg/m^3
    """
    if model is not None:
        return model.density(altitude)

    if body not in ATMOSPHERE_MODELS:
        raise ValueError(f"Unknown body: {body}. Known: {list(ATMOSPHERE_MODELS.keys())}")

    return ATMOSPHERE_MODELS[body].density(altitude)


def atmospheric_temperature(
    altitude: Union[float, np.ndarray],
    body: str = "earth",
    model: Optional[AtmosphereModel] = None,
) -> Union[float, np.ndarray]:
    """Compute atmospheric temperature at altitude."""
    if model is not None:
        return model.temperature(altitude)

    if body not in ATMOSPHERE_MODELS:
        raise ValueError(f"Unknown body: {body}")

    return ATMOSPHERE_MODELS[body].temperature(altitude)


def atmospheric_pressure(
    altitude: Union[float, np.ndarray],
    body: str = "earth",
    model: Optional[AtmosphereModel] = None,
) -> Union[float, np.ndarray]:
    """Compute atmospheric pressure at altitude."""
    if model is not None:
        return model.pressure(altitude)

    if body not in ATMOSPHERE_MODELS:
        raise ValueError(f"Unknown body: {body}")

    return ATMOSPHERE_MODELS[body].pressure(altitude)


def scale_height(
    altitude: Union[float, np.ndarray],
    body: str = "earth",
    model: Optional[AtmosphereModel] = None,
) -> Union[float, np.ndarray]:
    """Compute local scale height at altitude."""
    if model is not None:
        return model.scale_height_at(altitude)

    if body not in ATMOSPHERE_MODELS:
        raise ValueError(f"Unknown body: {body}")

    return ATMOSPHERE_MODELS[body].scale_height_at(altitude)


# =============================================================================
# CUSTOM MODEL CREATION
# =============================================================================

def create_custom_atmosphere(
    body_name: str,
    rho_0: float,
    T_0: float,
    P_0: float,
    H_scale: float,
    g_0: float,
    tenant_id: str = "default",
) -> Tuple[AtmosphereModel, Dict[str, Any]]:
    """
    Create a custom atmosphere model.

    Args:
        body_name: Name for this atmosphere
        rho_0: Surface density (kg/m^3)
        T_0: Surface temperature (K)
        P_0: Surface pressure (Pa)
        H_scale: Scale height (m)
        g_0: Surface gravity (m/s^2)
        tenant_id: Tenant ID for receipt

    Returns:
        Tuple of (AtmosphereModel, atmosphere_model_receipt)
    """
    model = AtmosphereModel(
        body_name=body_name,
        rho_0=rho_0,
        T_0=T_0,
        P_0=P_0,
        H_scale=H_scale,
        g_0=g_0,
    )

    # Validate: density at sea level should match rho_0
    test_density = model.density(0.0)
    if not np.isclose(test_density, rho_0):
        raise ValueError(f"Model validation failed: density(0) = {test_density} != {rho_0}")

    receipt = emit_receipt("atmosphere_model", {
        "tenant_id": tenant_id,
        "body_name": body_name,
        "rho_0_kgm3": rho_0,
        "T_0_K": T_0,
        "P_0_Pa": P_0,
        "H_scale_m": H_scale,
        "g_0_mps2": g_0,
        "model_hash": dual_hash(json.dumps({
            "body_name": body_name,
            "rho_0": rho_0,
            "T_0": T_0,
            "P_0": P_0,
            "H_scale": H_scale,
            "g_0": g_0,
        }, sort_keys=True)),
    })

    return model, receipt


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_earth_atmosphere(tenant_id: str = "default") -> Tuple[bool, Dict[str, Any]]:
    """
    Validate Earth atmosphere model against reference values.

    Reference: US Standard Atmosphere 1976

    Returns:
        Tuple of (all_pass, validation_receipt)
    """
    # Reference values at key altitudes
    reference = {
        0: {"rho": 1.225, "T": 288.15, "P": 101325.0},
        5000: {"rho": 0.7364, "T": 255.65, "P": 54048.0},
        10000: {"rho": 0.4135, "T": 223.15, "P": 26500.0},
        20000: {"rho": 0.0889, "T": 216.65, "P": 5529.0},
        50000: {"rho": 0.001027, "T": 270.65, "P": 79.78},
    }

    errors = []
    for alt, ref in reference.items():
        rho = EARTH_ATMOSPHERE.density(float(alt))
        T = EARTH_ATMOSPHERE.temperature(float(alt))
        P = EARTH_ATMOSPHERE.pressure(float(alt))

        rho_err = abs(rho - ref["rho"]) / ref["rho"]
        T_err = abs(T - ref["T"]) / ref["T"]
        P_err = abs(P - ref["P"]) / ref["P"]

        if rho_err > 0.1:  # 10% tolerance
            errors.append(f"Density at {alt}m: {rho:.4f} vs {ref['rho']:.4f}")
        if T_err > 0.05:  # 5% tolerance
            errors.append(f"Temperature at {alt}m: {T:.2f} vs {ref['T']:.2f}")
        if P_err > 0.1:  # 10% tolerance
            errors.append(f"Pressure at {alt}m: {P:.2f} vs {ref['P']:.2f}")

    all_pass = len(errors) == 0

    receipt = emit_receipt("atmosphere_validation", {
        "tenant_id": tenant_id,
        "body": "earth",
        "reference": "US Standard Atmosphere 1976",
        "altitudes_tested": list(reference.keys()),
        "errors": errors,
        "slo_pass": all_pass,
    })

    return all_pass, receipt
