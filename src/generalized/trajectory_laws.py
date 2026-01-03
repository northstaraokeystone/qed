"""
src/generalized/trajectory_laws.py - Generalized Trajectory Physics Laws

Generalized trajectory physics models for any unpowered flight regime.
CLAUDEME v3.1 Compliant: All functions emit receipts.

Flight Regimes:
    - Ballistic: No lift, pure drag (Dragon capsule)
    - Lifting: L/D ratio, bank angle control (Starship, Shuttle)
    - Orbital: Kepler dynamics + perturbations (satellites)
    - Hypersonic glide: L/D modulation (X-37B, Dream Chaser)

Receipt: trajectory_law_receipt
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import integrate

# Import from project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from receipts import dual_hash, emit_receipt, StopRule

from .atmospheric_models import (
    AtmosphereModel,
    EARTH_ATMOSPHERE,
    MARS_ATMOSPHERE,
    atmospheric_density,
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Gravitational parameters (m^3/s^2)
MU_EARTH = 3.986004418e14
MU_MARS = 4.282837e13

# Planet radii (m)
R_EARTH = 6.371e6
R_MARS = 3.3895e6


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class VehicleConfig:
    """Vehicle configuration for trajectory prediction."""
    name: str
    mass_kg: float
    area_m2: float
    drag_coefficient: float
    lift_coefficient: float = 0.0
    flight_regime: str = "ballistic"  # ballistic, lifting, orbital, hypersonic

    @property
    def lift_to_drag(self) -> float:
        """Compute L/D ratio."""
        if self.drag_coefficient == 0:
            return 0.0
        return self.lift_coefficient / self.drag_coefficient

    @property
    def ballistic_coefficient(self) -> float:
        """Compute ballistic coefficient (kg/m^2)."""
        if self.drag_coefficient == 0 or self.area_m2 == 0:
            return float("inf")
        return self.mass_kg / (self.drag_coefficient * self.area_m2)


# Preset vehicle configurations
STARSHIP_CONFIG = VehicleConfig(
    name="starship",
    mass_kg=100000.0,
    area_m2=300.0,
    drag_coefficient=1.5,
    lift_coefficient=0.5,
    flight_regime="lifting",
)

DRAGON_CONFIG = VehicleConfig(
    name="dragon_capsule",
    mass_kg=12000.0,
    area_m2=10.0,
    drag_coefficient=1.2,
    lift_coefficient=0.3,
    flight_regime="ballistic",
)

MARS_SAMPLE_RETURN_CONFIG = VehicleConfig(
    name="mars_sample_return",
    mass_kg=500.0,
    area_m2=3.0,
    drag_coefficient=1.4,
    lift_coefficient=0.0,
    flight_regime="ballistic",
)


# =============================================================================
# ABSTRACT TRAJECTORY LAW
# =============================================================================

class TrajectoryLaw(ABC):
    """Abstract base class for trajectory physics laws."""

    @abstractmethod
    def predict(
        self,
        initial_state: Dict[str, float],
        time_span: np.ndarray,
        vehicle: VehicleConfig,
    ) -> Dict[str, np.ndarray]:
        """
        Predict trajectory given initial state.

        Args:
            initial_state: Dict with 'altitude', 'velocity', 'flight_path_angle'
            time_span: Array of time points
            vehicle: Vehicle configuration

        Returns:
            Dict with 'altitude', 'velocity', 'acceleration' arrays
        """
        pass

    @abstractmethod
    def compress(
        self,
        observed: Dict[str, np.ndarray],
        predicted: Dict[str, np.ndarray],
    ) -> bytes:
        """Compress observed data using this law's predictions."""
        pass

    @abstractmethod
    def decompress(
        self,
        compressed: bytes,
        initial_state: Dict[str, float],
        time_span: np.ndarray,
        vehicle: VehicleConfig,
    ) -> Dict[str, np.ndarray]:
        """Decompress data using this law."""
        pass


# =============================================================================
# BALLISTIC TRAJECTORY LAW
# =============================================================================

class BallisticLaw(TrajectoryLaw):
    """
    Ballistic trajectory law for unpowered descent.

    Equations of motion:
        dv/dt = -D/m - g*sin(gamma)
        dgamma/dt = (g - v^2/r)*cos(gamma) / v
        dh/dt = v*sin(gamma)

    Where D = 0.5 * rho * v^2 * Cd * A
    """

    def __init__(
        self,
        atmosphere: AtmosphereModel = EARTH_ATMOSPHERE,
        planet_mu: float = MU_EARTH,
        planet_radius: float = R_EARTH,
    ):
        self.atmosphere = atmosphere
        self.mu = planet_mu
        self.R = planet_radius

    def _gravity(self, altitude: float) -> float:
        """Compute local gravity."""
        r = self.R + altitude
        return self.mu / (r ** 2)

    def _dynamics(
        self,
        t: float,
        state: np.ndarray,
        vehicle: VehicleConfig,
    ) -> np.ndarray:
        """
        Compute state derivatives.

        State: [altitude, velocity, flight_path_angle]
        """
        h, v, gamma = state

        # Handle ground impact
        if h <= 0:
            return np.array([0.0, 0.0, 0.0])

        # Atmospheric properties
        rho = self.atmosphere.density(h)

        # Gravity
        g = self._gravity(h)

        # Drag
        D = 0.5 * rho * v ** 2 * vehicle.drag_coefficient * vehicle.area_m2

        # Derivatives
        dv_dt = -D / vehicle.mass_kg - g * np.sin(gamma)
        r = self.R + h

        # Avoid division by zero
        if v < 1.0:
            dgamma_dt = 0.0
        else:
            dgamma_dt = (g - v ** 2 / r) * np.cos(gamma) / v

        dh_dt = v * np.sin(gamma)

        return np.array([dh_dt, dv_dt, dgamma_dt])

    def predict(
        self,
        initial_state: Dict[str, float],
        time_span: np.ndarray,
        vehicle: VehicleConfig,
    ) -> Dict[str, np.ndarray]:
        """Predict ballistic trajectory."""
        h0 = initial_state["altitude"]
        v0 = initial_state["velocity"]
        gamma0 = initial_state.get("flight_path_angle", -np.pi / 2)  # Default: straight down

        y0 = np.array([h0, v0, gamma0])

        # Use simple RK4 integration for speed
        dt = time_span[1] - time_span[0] if len(time_span) > 1 else 1.0
        n = len(time_span)

        altitude = np.zeros(n)
        velocity = np.zeros(n)
        gamma = np.zeros(n)

        altitude[0] = h0
        velocity[0] = v0
        gamma[0] = gamma0

        for i in range(1, n):
            y = np.array([altitude[i-1], velocity[i-1], gamma[i-1]])

            # RK4 step
            k1 = self._dynamics(time_span[i-1], y, vehicle)
            k2 = self._dynamics(time_span[i-1] + dt/2, y + dt/2 * k1, vehicle)
            k3 = self._dynamics(time_span[i-1] + dt/2, y + dt/2 * k2, vehicle)
            k4 = self._dynamics(time_span[i], y + dt * k3, vehicle)

            y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

            altitude[i] = max(0, y_new[0])
            velocity[i] = max(0, y_new[1])
            gamma[i] = y_new[2]

            # Stop if we hit the ground
            if altitude[i] <= 0:
                altitude[i:] = 0
                velocity[i:] = velocity[i]
                gamma[i:] = gamma[i]
                break

        # Compute acceleration from velocity derivative
        acceleration = np.gradient(velocity, time_span)

        return {
            "altitude": altitude,
            "velocity": velocity,
            "flight_path_angle": gamma,
            "acceleration": acceleration,
        }

    def compress(
        self,
        observed: Dict[str, np.ndarray],
        predicted: Dict[str, np.ndarray],
    ) -> bytes:
        """Compress by storing residuals."""
        alt_residual = observed["altitude"] - predicted["altitude"]
        vel_residual = observed["velocity"] - predicted["velocity"]

        # Quantize to 16-bit
        alt_quant = np.clip(alt_residual, -32767, 32767).astype(np.int16)
        vel_quant = np.clip(vel_residual, -32767, 32767).astype(np.int16)

        data = {
            "alt_residual": alt_quant.tobytes().hex(),
            "vel_residual": vel_quant.tobytes().hex(),
            "n_samples": len(alt_residual),
        }

        return json.dumps(data).encode()

    def decompress(
        self,
        compressed: bytes,
        initial_state: Dict[str, float],
        time_span: np.ndarray,
        vehicle: VehicleConfig,
    ) -> Dict[str, np.ndarray]:
        """Decompress using predicted + residuals."""
        data = json.loads(compressed.decode())

        # Get prediction
        predicted = self.predict(initial_state, time_span, vehicle)

        # Decode residuals
        alt_residual = np.frombuffer(
            bytes.fromhex(data["alt_residual"]), dtype=np.int16
        ).astype(np.float64)
        vel_residual = np.frombuffer(
            bytes.fromhex(data["vel_residual"]), dtype=np.int16
        ).astype(np.float64)

        return {
            "altitude": predicted["altitude"] + alt_residual,
            "velocity": predicted["velocity"] + vel_residual,
            "acceleration": np.gradient(predicted["velocity"] + vel_residual, time_span),
        }


# =============================================================================
# LIFTING TRAJECTORY LAW
# =============================================================================

class LiftingLaw(TrajectoryLaw):
    """
    Lifting body trajectory law for controlled descent.

    Extends ballistic with lift force:
        L = 0.5 * rho * v^2 * Cl * A
        dv/dt = -D/m - g*sin(gamma)
        dgamma/dt = L/(m*v) * cos(bank) + (g - v^2/r)*cos(gamma) / v

    Bank angle modulates cross-range and descent rate.
    """

    def __init__(
        self,
        atmosphere: AtmosphereModel = EARTH_ATMOSPHERE,
        planet_mu: float = MU_EARTH,
        planet_radius: float = R_EARTH,
        bank_angle: float = 0.0,  # radians
    ):
        self.atmosphere = atmosphere
        self.mu = planet_mu
        self.R = planet_radius
        self.bank_angle = bank_angle

    def _gravity(self, altitude: float) -> float:
        r = self.R + altitude
        return self.mu / (r ** 2)

    def _dynamics(
        self,
        t: float,
        state: np.ndarray,
        vehicle: VehicleConfig,
    ) -> np.ndarray:
        """Compute state derivatives with lift."""
        h, v, gamma = state

        if h <= 0:
            return np.array([0.0, 0.0, 0.0])

        rho = self.atmosphere.density(h)
        g = self._gravity(h)

        # Dynamic pressure
        q = 0.5 * rho * v ** 2

        # Drag and Lift
        D = q * vehicle.drag_coefficient * vehicle.area_m2
        L = q * vehicle.lift_coefficient * vehicle.area_m2

        # Derivatives
        dv_dt = -D / vehicle.mass_kg - g * np.sin(gamma)

        r = self.R + h
        if v < 1.0:
            dgamma_dt = 0.0
        else:
            lift_term = (L / (vehicle.mass_kg * v)) * np.cos(self.bank_angle)
            gravity_term = (g - v ** 2 / r) * np.cos(gamma) / v
            dgamma_dt = lift_term + gravity_term

        dh_dt = v * np.sin(gamma)

        return np.array([dh_dt, dv_dt, dgamma_dt])

    def predict(
        self,
        initial_state: Dict[str, float],
        time_span: np.ndarray,
        vehicle: VehicleConfig,
    ) -> Dict[str, np.ndarray]:
        """Predict lifting trajectory."""
        h0 = initial_state["altitude"]
        v0 = initial_state["velocity"]
        gamma0 = initial_state.get("flight_path_angle", -np.pi / 6)  # Shallower for lifting

        dt = time_span[1] - time_span[0] if len(time_span) > 1 else 1.0
        n = len(time_span)

        altitude = np.zeros(n)
        velocity = np.zeros(n)
        gamma = np.zeros(n)

        altitude[0] = h0
        velocity[0] = v0
        gamma[0] = gamma0

        for i in range(1, n):
            y = np.array([altitude[i-1], velocity[i-1], gamma[i-1]])

            k1 = self._dynamics(time_span[i-1], y, vehicle)
            k2 = self._dynamics(time_span[i-1] + dt/2, y + dt/2 * k1, vehicle)
            k3 = self._dynamics(time_span[i-1] + dt/2, y + dt/2 * k2, vehicle)
            k4 = self._dynamics(time_span[i], y + dt * k3, vehicle)

            y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

            altitude[i] = max(0, y_new[0])
            velocity[i] = max(0, y_new[1])
            gamma[i] = y_new[2]

            if altitude[i] <= 0:
                altitude[i:] = 0
                velocity[i:] = velocity[i]
                gamma[i:] = gamma[i]
                break

        acceleration = np.gradient(velocity, time_span)

        return {
            "altitude": altitude,
            "velocity": velocity,
            "flight_path_angle": gamma,
            "acceleration": acceleration,
        }

    def compress(
        self,
        observed: Dict[str, np.ndarray],
        predicted: Dict[str, np.ndarray],
    ) -> bytes:
        alt_residual = observed["altitude"] - predicted["altitude"]
        vel_residual = observed["velocity"] - predicted["velocity"]

        alt_quant = np.clip(alt_residual, -32767, 32767).astype(np.int16)
        vel_quant = np.clip(vel_residual, -32767, 32767).astype(np.int16)

        data = {
            "alt_residual": alt_quant.tobytes().hex(),
            "vel_residual": vel_quant.tobytes().hex(),
            "n_samples": len(alt_residual),
            "bank_angle": self.bank_angle,
        }

        return json.dumps(data).encode()

    def decompress(
        self,
        compressed: bytes,
        initial_state: Dict[str, float],
        time_span: np.ndarray,
        vehicle: VehicleConfig,
    ) -> Dict[str, np.ndarray]:
        data = json.loads(compressed.decode())
        predicted = self.predict(initial_state, time_span, vehicle)

        alt_residual = np.frombuffer(
            bytes.fromhex(data["alt_residual"]), dtype=np.int16
        ).astype(np.float64)
        vel_residual = np.frombuffer(
            bytes.fromhex(data["vel_residual"]), dtype=np.int16
        ).astype(np.float64)

        return {
            "altitude": predicted["altitude"] + alt_residual,
            "velocity": predicted["velocity"] + vel_residual,
            "acceleration": np.gradient(predicted["velocity"] + vel_residual, time_span),
        }


# =============================================================================
# ORBITAL TRAJECTORY LAW
# =============================================================================

class OrbitalLaw(TrajectoryLaw):
    """
    Keplerian orbital trajectory law.

    For orbits with minimal drag (high altitude satellites).
    Uses orbital elements: a, e, i, Omega, omega, nu

    Energy and angular momentum are conserved:
        E = -mu / (2a)
        h = sqrt(mu * a * (1 - e^2))
    """

    def __init__(
        self,
        planet_mu: float = MU_EARTH,
        planet_radius: float = R_EARTH,
        include_j2: bool = False,
    ):
        self.mu = planet_mu
        self.R = planet_radius
        self.include_j2 = include_j2
        self.J2 = 1.08263e-3 if include_j2 else 0.0  # Earth J2

    def predict(
        self,
        initial_state: Dict[str, float],
        time_span: np.ndarray,
        vehicle: VehicleConfig,
    ) -> Dict[str, np.ndarray]:
        """
        Predict orbital trajectory.

        Initial state needs: altitude, velocity, flight_path_angle
        Converts to orbital elements and propagates.
        """
        h0 = initial_state["altitude"]
        v0 = initial_state["velocity"]
        gamma0 = initial_state.get("flight_path_angle", 0.0)

        r0 = self.R + h0

        # Compute orbital elements from state
        # Specific energy
        energy = v0 ** 2 / 2 - self.mu / r0

        # Semi-major axis
        if energy >= 0:
            # Hyperbolic or parabolic - not truly orbital
            a = r0  # Approximate
        else:
            a = -self.mu / (2 * energy)

        # Angular momentum magnitude
        h_ang = r0 * v0 * np.cos(gamma0)

        # Eccentricity
        e_sq = 1 + 2 * energy * h_ang ** 2 / (self.mu ** 2)
        e = np.sqrt(max(0, e_sq))

        # Orbital period (if elliptical)
        if a > 0:
            T = 2 * np.pi * np.sqrt(a ** 3 / self.mu)
        else:
            T = float("inf")

        # Propagate using Kepler's equation
        n = len(time_span)
        altitude = np.zeros(n)
        velocity = np.zeros(n)

        # Mean motion
        if a > 0:
            n_motion = np.sqrt(self.mu / a ** 3)
        else:
            n_motion = 0

        # Initial true anomaly
        if e > 0:
            cos_nu0 = (a * (1 - e ** 2) / r0 - 1) / e
            cos_nu0 = np.clip(cos_nu0, -1, 1)
            nu0 = np.arccos(cos_nu0)
            if gamma0 < 0:
                nu0 = -nu0
        else:
            nu0 = 0

        # Initial eccentric anomaly
        if e < 1:
            tan_E0_2 = np.tan(nu0 / 2) * np.sqrt((1 - e) / (1 + e))
            E0 = 2 * np.arctan(tan_E0_2)
            M0 = E0 - e * np.sin(E0)
        else:
            M0 = 0
            E0 = 0

        for i, t in enumerate(time_span):
            dt = t - time_span[0]
            M = M0 + n_motion * dt

            # Solve Kepler's equation: M = E - e*sin(E)
            E = M
            for _ in range(10):  # Newton iteration
                E_new = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
                if abs(E_new - E) < 1e-10:
                    break
                E = E_new

            # True anomaly
            tan_nu_2 = np.tan(E / 2) * np.sqrt((1 + e) / (1 - e))
            nu = 2 * np.arctan(tan_nu_2)

            # Radius
            r = a * (1 - e ** 2) / (1 + e * np.cos(nu))
            altitude[i] = r - self.R

            # Velocity (vis-viva)
            if r > 0:
                v = np.sqrt(self.mu * (2 / r - 1 / a))
            else:
                v = 0
            velocity[i] = v

        acceleration = np.gradient(velocity, time_span)

        return {
            "altitude": altitude,
            "velocity": velocity,
            "acceleration": acceleration,
            "semi_major_axis": a,
            "eccentricity": e,
            "period": T,
        }

    def compress(
        self,
        observed: Dict[str, np.ndarray],
        predicted: Dict[str, np.ndarray],
    ) -> bytes:
        alt_residual = observed["altitude"] - predicted["altitude"]
        vel_residual = observed["velocity"] - predicted["velocity"]

        alt_quant = np.clip(alt_residual, -32767, 32767).astype(np.int16)
        vel_quant = np.clip(vel_residual, -32767, 32767).astype(np.int16)

        data = {
            "alt_residual": alt_quant.tobytes().hex(),
            "vel_residual": vel_quant.tobytes().hex(),
            "n_samples": len(alt_residual),
            "semi_major_axis": float(predicted.get("semi_major_axis", 0)),
            "eccentricity": float(predicted.get("eccentricity", 0)),
        }

        return json.dumps(data).encode()

    def decompress(
        self,
        compressed: bytes,
        initial_state: Dict[str, float],
        time_span: np.ndarray,
        vehicle: VehicleConfig,
    ) -> Dict[str, np.ndarray]:
        data = json.loads(compressed.decode())
        predicted = self.predict(initial_state, time_span, vehicle)

        alt_residual = np.frombuffer(
            bytes.fromhex(data["alt_residual"]), dtype=np.int16
        ).astype(np.float64)
        vel_residual = np.frombuffer(
            bytes.fromhex(data["vel_residual"]), dtype=np.int16
        ).astype(np.float64)

        return {
            "altitude": predicted["altitude"] + alt_residual,
            "velocity": predicted["velocity"] + vel_residual,
            "acceleration": np.gradient(predicted["velocity"] + vel_residual, time_span),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def predict_trajectory(
    initial_state: Dict[str, float],
    time_span: np.ndarray,
    vehicle: VehicleConfig,
    body: str = "earth",
    tenant_id: str = "default",
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Predict trajectory using appropriate physics law.

    Automatically selects law based on vehicle flight regime.

    Args:
        initial_state: Dict with altitude, velocity, flight_path_angle
        time_span: Array of time points
        vehicle: Vehicle configuration
        body: Celestial body ("earth", "mars")
        tenant_id: Tenant ID for receipt

    Returns:
        Tuple of (trajectory_dict, trajectory_law_receipt)
    """
    # Select atmosphere
    if body == "earth":
        atmosphere = EARTH_ATMOSPHERE
        mu = MU_EARTH
        R = R_EARTH
    elif body == "mars":
        atmosphere = MARS_ATMOSPHERE
        mu = MU_MARS
        R = R_MARS
    else:
        raise ValueError(f"Unknown body: {body}")

    # Select law based on flight regime
    if vehicle.flight_regime == "ballistic":
        law = BallisticLaw(atmosphere, mu, R)
    elif vehicle.flight_regime == "lifting":
        law = LiftingLaw(atmosphere, mu, R)
    elif vehicle.flight_regime == "orbital":
        law = OrbitalLaw(mu, R)
    else:
        law = BallisticLaw(atmosphere, mu, R)  # Default

    # Predict
    trajectory = law.predict(initial_state, time_span, vehicle)

    # Emit receipt
    receipt = emit_receipt("trajectory_law", {
        "tenant_id": tenant_id,
        "vehicle_name": vehicle.name,
        "flight_regime": vehicle.flight_regime,
        "body": body,
        "n_samples": len(time_span),
        "duration_s": float(time_span[-1] - time_span[0]) if len(time_span) > 1 else 0,
        "initial_altitude_m": initial_state["altitude"],
        "final_altitude_m": float(trajectory["altitude"][-1]),
        "initial_velocity_mps": initial_state["velocity"],
        "final_velocity_mps": float(trajectory["velocity"][-1]),
    })

    return trajectory, receipt


def compress_with_law(
    observed: Dict[str, np.ndarray],
    initial_state: Dict[str, float],
    time_span: np.ndarray,
    vehicle: VehicleConfig,
    body: str = "earth",
    tenant_id: str = "default",
) -> Tuple[bytes, float, Dict[str, Any]]:
    """
    Compress observed trajectory using physics law.

    Returns:
        Tuple of (compressed_bytes, compression_ratio, compression_receipt)
    """
    # Select appropriate law
    if body == "earth":
        atmosphere = EARTH_ATMOSPHERE
        mu = MU_EARTH
        R = R_EARTH
    else:
        atmosphere = MARS_ATMOSPHERE
        mu = MU_MARS
        R = R_MARS

    if vehicle.flight_regime == "ballistic":
        law = BallisticLaw(atmosphere, mu, R)
    elif vehicle.flight_regime == "lifting":
        law = LiftingLaw(atmosphere, mu, R)
    else:
        law = BallisticLaw(atmosphere, mu, R)

    # Predict
    predicted = law.predict(initial_state, time_span, vehicle)

    # Compress
    compressed = law.compress(observed, predicted)

    # Calculate ratio
    raw_bytes = len(observed["altitude"]) * 8 * 2  # 2 channels * 8 bytes
    ratio = raw_bytes / len(compressed) if len(compressed) > 0 else 1.0

    receipt = emit_receipt("trajectory_compression", {
        "tenant_id": tenant_id,
        "vehicle_name": vehicle.name,
        "raw_bytes": raw_bytes,
        "compressed_bytes": len(compressed),
        "compression_ratio": ratio,
        "compressed_hash": dual_hash(compressed),
    })

    return compressed, ratio, receipt


def decompress_with_law(
    compressed: bytes,
    initial_state: Dict[str, float],
    time_span: np.ndarray,
    vehicle: VehicleConfig,
    body: str = "earth",
    tenant_id: str = "default",
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Decompress trajectory using physics law.

    Returns:
        Tuple of (trajectory_dict, decompression_receipt)
    """
    if body == "earth":
        atmosphere = EARTH_ATMOSPHERE
        mu = MU_EARTH
        R = R_EARTH
    else:
        atmosphere = MARS_ATMOSPHERE
        mu = MU_MARS
        R = R_MARS

    if vehicle.flight_regime == "ballistic":
        law = BallisticLaw(atmosphere, mu, R)
    elif vehicle.flight_regime == "lifting":
        law = LiftingLaw(atmosphere, mu, R)
    else:
        law = BallisticLaw(atmosphere, mu, R)

    trajectory = law.decompress(compressed, initial_state, time_span, vehicle)

    receipt = emit_receipt("trajectory_decompression", {
        "tenant_id": tenant_id,
        "vehicle_name": vehicle.name,
        "n_samples": len(time_span),
        "decompressed_hash": dual_hash(json.dumps({
            k: v.tolist() for k, v in trajectory.items()
        }, sort_keys=True)),
    })

    return trajectory, receipt
