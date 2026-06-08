"""
EpicyclicGearSystem  (Version 4 – solver only, no GUI)
======================================================

PURPOSE
-------
This module is the numerical backbone of the planetary / epicyclic
gear tool.  It contains only the mathematics and data handling.

The model implemented here follows this structure:

  1) Build geometry and stiffness
  2) Generate STATIC manufacturing / assembly errors
  3) Build PERIODIC effects over one phase sweep:
        - rotating eccentricity
        - time-varying mesh stiffness
  4) Solve the Velex-style equilibrium problem:
        q = [w ; phi_x ; phi_y]
     where:
        w      = settlement / vertical approach [m]
        phi_x  = tilt about x-axis [rad]
        phi_y  = tilt about y-axis [rad]
  5) Recover planet forces and load sharing from the SAME solve
  6) Optionally repeat the whole process in Monte Carlo mode
  7) Optionally compute sensitivity sweep data (graphs etc.)

IMPORTANT MODEL INTERPRETATION
------------------------------
This is NOT a full rigid-body 3D dynamic model.
It is a quasi-static "translational analogy" / "tilting plate on
springs" style model, using generalized coordinates:

      q = [w ; phi_x ; phi_y]

It is therefore a very useful REDUCED-ORDER model of floating-sun
settlement / tilt and load sharing, but not a full ODE dynamics model.

INTERNAL UNIT CONVENTION
------------------------
Internal calculations use:
  length     -> meters [m]
  force      -> Newtons [N]
  stiffness  -> N/m
  angle      -> radians [rad]

User inputs may be in mm / microns / degrees; they are converted.

Usage
-----
    inputs = {}
    inputs['N'] = 5
    inputs['T_input_Nm'] = 1000
    inputs['R_sun_mm'] = 50
    # ...

    sys = EpicyclicGearSystem(inputs)
    sys.run()
    sys.print_summary()

    res = sys.results

OUTPUT HIGHLIGHTS
-----------------
    results['phase_rad']               -> phase sweep vector
    results['force_phase_N']           -> planet forces vs phase
    results['LSF_phase']               -> planet LSF vs phase
    results['K_gamma_phase']           -> system load-sharing factor vs phase
    results['sun_q_phase']             -> [w; phi_x; phi_y] vs phase
    results['LSF_final']               -> planet LSF at worst phase
    results['sun_displacement_final']  -> [w; phi_x; phi_y] at worst phase
    results['sensitivity']             -> data for future plots
    results['monte_carlo']             -> summary of MC runs if enabled

Translated from MATLAB (EpicyclicGearSystem.m) with zero numerical loss.
"""

from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# =========================================================================
# SAFE MATH HELPERS
# =========================================================================

def acos_safe(x: float) -> float:
    """Safe acos with argument clamp to avoid small floating-point violations."""
    return math.acos(max(min(x, 1.0), -1.0))


def asin_safe(x: float) -> float:
    """Safe asin with argument clamp to avoid small floating-point violations."""
    return math.asin(max(min(x, 1.0), -1.0))


def sqrt_safe(x: float) -> float:
    """Safe sqrt for geometry expressions that might hit tiny negative values from roundoff."""
    return math.sqrt(max(x, 0.0))


# =========================================================================
# TOLERANCE PRESETS
# =========================================================================

def get_tolerance_preset(preset: str) -> dict:
    """
    Engineering default error presets, in MICRONS.
    These are project-friendly defaults rather than a strict standards table.
    """
    preset_lower = preset.lower().strip()

    if preset_lower == 'fine':
        return dict(
            pin_rad_um=8,
            pin_tan_um=8,
            runout_um=5,
            profile_um=4,
            commonProfile_um=2,
            commonXY_um=2,
        )
    elif preset_lower == 'medium':
        return dict(
            pin_rad_um=15,
            pin_tan_um=15,
            runout_um=10,
            profile_um=8,
            commonProfile_um=3,
            commonXY_um=3,
        )
    elif preset_lower == 'coarse':
        return dict(
            pin_rad_um=25,
            pin_tan_um=25,
            runout_um=18,
            profile_um=12,
            commonProfile_um=5,
            commonXY_um=5,
        )
    else:
        raise ValueError(f"Unknown error_preset '{preset}'. Use fine, medium, or coarse.")


# =========================================================================
# BEARING STIFFNESS LOOKUP
# =========================================================================

def get_bearing_stiffness_constant(bearing_type: str) -> float:
    """
    Returns one of the constant bearing stiffness magnitudes used in the
    reduced-order model.

    IMPORTANT:
    These are engineering preset values used to represent the support-path
    compliance in a compact way. They are not intended to identify a specific
    commercial bearing catalog value; rather, they provide representative
    stiffness levels for comparative load-sharing studies.

    Presets used:
      1e6  -> ball bearing with clearance
      1e7  -> ball bearing without clearance
      5e7  -> "standard" intermediate support stiffness
      1e8  -> cylindrical / roller with clearance
      1e9  -> cylindrical / roller without clearance

    All values are in N/m.
    """
    bt = bearing_type.lower().strip()

    lookup = {
        'ball_clearance': 1e6,
        'ball_with_clearance': 1e6,
        'ball_noclearance': 1e7,
        'ball_without_clearance': 1e7,
        'ball': 1e7,
        'standard': 5e7,
        'standard_5e7': 5e7,
        'intermediate': 5e7,
        'mid': 5e7,
        'roller_clearance': 1e8,
        'cylindrical_clearance': 1e8,
        'roller_with_clearance': 1e8,
        'cylindrical_with_clearance': 1e8,
        'roller_noclearance': 1e9,
        'cylindrical_noclearance': 1e9,
        'roller_without_clearance': 1e9,
        'cylindrical_without_clearance': 1e9,
        'roller': 1e9,
        'cylindrical': 1e9,
    }

    if bt not in lookup:
        raise ValueError(
            f"Unknown bearing_type '{bearing_type}'. Use one of: "
            "ball_clearance, ball_noclearance, standard, "
            "roller_clearance, roller_noclearance"
        )

    return lookup[bt]


# =========================================================================
# STOCHASTIC ERROR SAMPLING
# =========================================================================

def sample_err_microns_to_meters(
    tol_um: float,
    shape: tuple,
    params: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Samples a random error and returns it in METERS.

    tolMeaning options:
      'sigma' : tol_um is interpreted as 1-sigma value
      'pm'    : tol_um is interpreted as a +/- tolerance band

    dist options:
      'normal'
      'uniform'
    """
    if tol_um is None or tol_um == 0:
        return np.zeros(shape)

    um_to_m = 1e-6
    tol_meaning = params.get('tolMeaning', 'pm').lower()
    dist = params.get('dist', 'normal').lower()
    sigma_rule = params.get('sigmaRule', 3)
    truncate = params.get('truncate', True)

    if tol_meaning == 'sigma':
        sigma = tol_um * um_to_m
        T = None
    elif tol_meaning == 'pm':
        T = tol_um * um_to_m
        sigma = T / sigma_rule
    else:
        raise ValueError("tolMeaning must be 'pm' or 'sigma'.")

    if dist == 'normal':
        x_m = sigma * rng.standard_normal(shape)

        if tol_meaning == 'pm' and truncate and T is not None:
            lb = -T
            ub = T
            max_iter = 50

            for _ in range(max_iter):
                mask = (x_m < lb) | (x_m > ub)
                if not np.any(mask):
                    break
                n_resample = int(np.sum(mask))
                x_m[mask] = sigma * rng.standard_normal(n_resample)

            x_m = np.clip(x_m, lb, ub)

    elif dist == 'uniform':
        if tol_meaning == 'sigma':
            T_uni = 3 * sigma
        else:
            T_uni = T
        x_m = (2 * rng.random(shape) - 1) * T_uni

    else:
        raise ValueError("dist must be 'normal' or 'uniform'.")

    return x_m


# =========================================================================
# MESH STIFFNESS FROM GEOMETRY
# =========================================================================

def compute_mesh_stiffness_from_geometry(params: dict) -> dict:
    """
    Computes:
      - sun-planet mesh stiffness mean/amplitude
      - ring-planet mesh stiffness mean/amplitude

    The formulas are adapted from Filip's script.

    Output units:
      all stiffness values -> N/m

    IMPORTANT:
      The geometry formulas below are only for the REDUCED stiffness model.
      They are not a full tooth contact mechanics solver.
    """
    # Read geometry inputs
    z1 = params['z1']       # sun teeth
    z2 = params['z2']       # planet teeth
    z3 = params['z3']       # ring teeth

    x1 = params['x1']       # profile shift coefficient, sun
    x2 = params['x2']       # profile shift coefficient, planet

    m0 = params['m0_mm']                         # normal module [mm]
    alpha_n = np.deg2rad(params['alpha_n_deg'])   # normal pressure angle [rad]
    beta = np.deg2rad(params['beta_deg'])         # helix angle [rad]
    aw = params['aw_mm']                          # center / working distance [mm]
    b_m = params['b_face_mm'] / 1000.0            # face width [m]

    # ----- Sun-planet geometry -------------------------------------------
    r1 = (m0 * z1) / (2 * math.cos(beta))     # [mm]
    r2 = (m0 * z2) / (2 * math.cos(beta))     # [mm]

    ha1 = (1 + x1) * m0                        # [mm]
    ha2 = (1 + x2) * m0                        # [mm]

    ra1 = r1 + ha1                              # [mm]
    ra2 = r2 + ha2                              # [mm]

    beta_b = asin_safe(math.sin(beta) * math.cos(alpha_n))

    # This follows Filip's script directly
    alpha_t0 = acos_safe(
        (math.cos(alpha_n) * math.cos(beta)) / math.cos(beta_b)
    )

    rb1 = r1 * math.cos(alpha_t0)              # [mm]
    rb2 = r2 * math.cos(alpha_t0)              # [mm]

    rw1 = aw * z1 / (z1 + z2)                  # [mm]
    alpha_w = acos_safe(rb1 / rw1)

    S1S2_ksp = (sqrt_safe(ra1**2 - rb1**2)
                - aw * math.sin(alpha_w)
                + sqrt_safe(ra2**2 - rb2**2))
    Pbt = 2 * math.pi * rb1 / z1

    epsilon_alpha_sp = S1S2_ksp / Pbt

    # Max / min reduced stiffness following Filip's convention
    ksp_max = (epsilon_alpha_sp * b_m / math.cos(beta_b)) * 1.3e10   # [N/m]
    ksp_min = (epsilon_alpha_sp * b_m / math.cos(beta_b)) * 1.2e10   # [N/m]

    # ----- Ring-planet geometry ------------------------------------------
    mt = m0 / math.cos(beta)                    # transverse module [mm]
    r3 = mt * z3 / 2                            # [mm]

    alpha_t = math.atan(math.tan(alpha_n) / math.cos(beta))
    rb3 = r3 * math.cos(alpha_t)                # [mm]
    ra3 = r3 - m0                               # [mm] internal gear outside radius model

    rw2 = aw * z2 / (z3 - z2)                   # [mm]
    alpha_w_krp = acos_safe(rb2 / rw2)

    S1S2_krp = (sqrt_safe(ra2**2 - rb2**2)
                + aw * math.sin(alpha_w_krp)
                - sqrt_safe(ra3**2 - rb3**2))
    Pbt_krp = 2 * math.pi * rb2 / z2

    epsilon_alpha_rp = S1S2_krp / Pbt_krp

    krp_max = (epsilon_alpha_rp * b_m / math.cos(beta_b)) * 1.3e10   # [N/m]
    krp_min = (epsilon_alpha_rp * b_m / math.cos(beta_b)) * 1.2e10   # [N/m]

    # ----- Convert max/min to mean/amplitude for periodic use ------------
    ksp_mean = 0.5 * (ksp_max + ksp_min)
    ksp_amp = 0.5 * (ksp_max - ksp_min)

    krp_mean = 0.5 * (krp_max + krp_min)
    krp_amp = 0.5 * (krp_max - krp_min)

    return dict(
        ksp_min_Nm=ksp_min,
        ksp_max_Nm=ksp_max,
        ksp_mean_Nm=ksp_mean,
        ksp_amp_Nm=ksp_amp,
        krp_min_Nm=krp_min,
        krp_max_Nm=krp_max,
        krp_mean_Nm=krp_mean,
        krp_amp_Nm=krp_amp,
        epsilon_alpha_sp=epsilon_alpha_sp,
        epsilon_alpha_rp=epsilon_alpha_rp,
        beta_b_rad=beta_b,
    )


# =========================================================================
# THERMAL MESH MODIFIER
# =========================================================================

def compute_thermal_mesh_modifier(params: dict) -> dict:
    """
    Lumped thermal correction for the reduced-order mesh stiffness model.

    MODELING RATIONALE
    -----------------
    This project does not use a full thermo-elastic tooth-contact model.
    Instead, temperature is introduced through a first-order engineering
    correction applied to the reduced mesh stiffness.

    The correction is based on two simplified effects:

      1) Material softening with temperature
         E(T) / E(T_ref) ~= 1 - c_E * DeltaT

      2) Face-width thermal expansion
         b(T) / b(T_ref) ~= 1 + alpha * DeltaT

    Combined:
         k(T) / k(T_ref) ~= (E(T)/E(T_ref)) * (b(T)/b(T_ref))
    """
    thermal = dict(
        deltaT_C=params['temperature_C'] - params['temperature_ref_C'],
        ksp_scale=1.0,
        krp_scale=1.0,
        E_ratio=1.0,
        b_ratio=1.0,
    )

    if not params['enable_temperature_effects']:
        return thermal

    dT = thermal['deltaT_C']

    # 1) Retained modulus / stiffness ratio
    E_ratio = 1 - params['thermal_E_drop_per_C'] * dT
    E_ratio = min(max(E_ratio, params['thermal_min_scale']), params['thermal_max_scale'])

    # 2) Face-width thermal expansion ratio
    b_ratio = 1 + params['lambda_thermal_1C'] * dT

    # 3) Combined mesh stiffness scale
    scale = E_ratio * b_ratio
    scale = min(max(scale, params['thermal_min_scale']), params['thermal_max_scale'])

    thermal['E_ratio'] = E_ratio
    thermal['b_ratio'] = b_ratio
    thermal['ksp_scale'] = scale
    thermal['krp_scale'] = scale

    return thermal


# =========================================================================
# THERMAL GEOMETRY ERROR
# =========================================================================

def compute_thermal_geometry_error(params: dict, geom: dict) -> dict:
    """
    Creates a differential THERMAL GEOMETRY ERROR per planet, expressed in
    the same equivalent line-of-action (LOA) sense as the other geometric
    error terms.

    Only retains the relative thermal mismatch between planets.
    """
    N = geom['N']

    thermal_geom = dict(
        e_th_m=np.zeros(N),
        dT_local_C=np.zeros(N),
        e_pin_th_m=np.zeros(N),
        e_planet_th_m=np.zeros(N),
        dT_mean_C=0.0,
        dT_grad_C=0.0,
    )

    if (not params['enable_temperature_effects'] or
            not params['enable_thermal_geometry_error']):
        return thermal_geom

    # 1) Mean temperature rise
    dT_mean = params['temperature_C'] - params['temperature_ref_C']

    # 2) Small circumferential thermal gradient
    if params['use_auto_thermal_gradient']:
        dT_grad = params['thermal_gradient_ratio'] * abs(dT_mean)
    else:
        dT_grad = params['thermal_gradient_C']

    psi_hot = np.deg2rad(params['thermal_hotspot_deg'])

    # Local planet temperature around the circumference
    psi_rad = geom['psi_rad']
    dT_local = dT_mean + dT_grad * np.cos(psi_rad - psi_hot)

    # 3) Carrier / pin-circle thermal expansion term
    aw_m = params['aw_mm'] / 1000.0
    alpha_lin = params['lambda_thermal_1C']

    # Radial thermal shift vector at each planet location [N x 2]
    eR = geom['eR']   # [N x 2]
    u_pin_th_m = (alpha_lin * aw_m * dT_local).reshape(-1, 1) * eR

    # Project onto LOA-equivalent scalar error
    nLOA = geom['nLOA']   # [N x 2]
    e_pin_th_m = -np.sum(u_pin_th_m * nLOA, axis=1)

    # 4) Planet effective size-growth term
    r_planet_pitch_m = ((params['m0_mm'] * params['z2']) /
                        (2 * math.cos(np.deg2rad(params['beta_deg']))) / 1000.0)

    e_planet_th_m = (alpha_lin * r_planet_pitch_m * dT_local *
                     math.sin(geom['alpha_rad']))

    # 5) Combined thermal geometry error
    e_th_m = (params['thermal_pin_weight'] * e_pin_th_m +
              params['thermal_planet_weight'] * e_planet_th_m)

    # Remove the common-mode component explicitly
    e_th_m = e_th_m - np.mean(e_th_m)

    thermal_geom['e_th_m'] = e_th_m
    thermal_geom['dT_local_C'] = dT_local
    thermal_geom['e_pin_th_m'] = e_pin_th_m
    thermal_geom['e_planet_th_m'] = e_planet_th_m
    thermal_geom['dT_mean_C'] = dT_mean
    thermal_geom['dT_grad_C'] = dT_grad

    return thermal_geom


# =========================================================================
# BUILD GEOMETRY
# =========================================================================

def build_geometry(params: dict) -> dict:
    """
    Builds:
      - planet angular positions
      - line-of-action directions used in the error projection model
      - mesh stiffness baseline (sun-planet and ring-planet)
      - bearing stiffness
      - effective stiffness via the TUTOR EQUATION:

          k_eff = [ 1/(kSP + kRP) + 1/kB ]^(-1)
    """
    N = params['N']

    # ----- Planet angular positions around the sun -----------------------
    psi_angles = params.get('psi_angles_deg')
    if psi_angles is None or len(psi_angles) == 0:
        psi_deg = np.arange(N) * (360.0 / N)
    else:
        psi_deg = np.asarray(psi_angles, dtype=float).flatten()
        if len(psi_deg) != N:
            raise ValueError('params.psi_angles_deg must have exactly N elements.')

    psi_rad = np.deg2rad(psi_deg)

    # ----- Analysis pressure angle for error projection ------------------
    alpha_rad = np.deg2rad(params['alpha_deg'])

    # LOA (line-of-action) direction convention
    theta_LOA = psi_rad - alpha_rad + (math.pi / 2)
    nLOA = np.column_stack([np.cos(theta_LOA), np.sin(theta_LOA)])   # [N x 2]

    # Radial and tangential local directions at each planet position
    eR = np.column_stack([np.cos(psi_rad), np.sin(psi_rad)])
    eT = np.column_stack([-np.sin(psi_rad), np.cos(psi_rad)])

    # ----- Support radius ------------------------------------------------
    r_support_m = params['r_support_mm'] / 1000.0

    # ----- Mesh stiffness model ------------------------------------------
    mesh = compute_mesh_stiffness_from_geometry(params)

    thermal = compute_thermal_mesh_modifier(params)

    mesh['ksp_mean_Nm'] *= thermal['ksp_scale']
    mesh['ksp_amp_Nm'] *= thermal['ksp_scale']
    mesh['krp_mean_Nm'] *= thermal['krp_scale']
    mesh['krp_amp_Nm'] *= thermal['krp_scale']

    # Apply optional global mesh scale factor for sensitivity studies
    mesh['ksp_mean_Nm'] *= params['mesh_scale_factor']
    mesh['ksp_amp_Nm'] *= params['mesh_scale_factor']
    mesh['krp_mean_Nm'] *= params['mesh_scale_factor']
    mesh['krp_amp_Nm'] *= params['mesh_scale_factor']

    # ----- Bearing stiffness selection -----------------------------------
    kB_base_Nm = get_bearing_stiffness_constant(params['bearing_type'])
    kB_base_Nm *= params['bearing_scale_factor']

    kB_vec_Nm = kB_base_Nm * np.ones(N)

    # ----- Baseline effective stiffness per planet -----------------------
    # IMPORTANT: This uses the tutor's equation exactly.
    kSP0_vec = mesh['ksp_mean_Nm'] * np.ones(N)
    kRP0_vec = mesh['krp_mean_Nm'] * np.ones(N)

    kEff0_vec = 1.0 / (1.0 / (kSP0_vec + kRP0_vec) + 1.0 / kB_vec_Nm)

    return dict(
        N=N,
        psi_rad=psi_rad,
        alpha_rad=alpha_rad,
        theta_LOA_rad=theta_LOA,
        nLOA=nLOA,
        eR=eR,
        eT=eT,
        r_support_m=r_support_m,
        mesh=mesh,
        kB_base_Nm=kB_base_Nm,
        kB_vec_Nm=kB_vec_Nm,
        kEff0_vec_Nm=kEff0_vec,
        thermal=thermal,
    )


# =========================================================================
# GENERATE STATIC ERRORS
# =========================================================================

def generate_static_errors(params: dict, geom: dict) -> dict:
    """
    Generates ONE STATIC error realization (manufacturing / assembly errors).

    Main output:
      e_static_m   [m]  (N,)

    Optional special mode:
      If params.single_planet_error_override = True, then:
        - all stochastic manufacturing / profile / runout errors are zero
        - common stochastic errors are zero
        - user rigid offsets still remain active
        - planet 1 receives the user-specified scalar error
    """
    # Set RNG
    seed = params.get('seed')
    if seed is not None and seed != '':
        rng = np.random.default_rng(int(seed))
    else:
        rng = np.random.default_rng()

    N = params['N']
    nLOA = geom['nLOA']      # [N x 2]
    eR = geom['eR']          # [N x 2]
    eT = geom['eT']          # [N x 2]

    # Deterministic offsets from user input
    sun_xy_m = np.asarray(params['sunOffset_xy_mm'], dtype=float).flatten() / 1000.0      # (2,)
    carrier_xy_m = np.asarray(params['carrierOffset_xy_mm'], dtype=float).flatten() / 1000.0  # (2,)

    extra_rel = params.get('extraRel_xy_mm')
    if extra_rel is None or (hasattr(extra_rel, '__len__') and len(extra_rel) == 0):
        extraRel_xy_m = np.zeros((N, 2))
    else:
        extraRel_xy_m = np.asarray(extra_rel, dtype=float) / 1000.0
        if extraRel_xy_m.shape != (N, 2):
            raise ValueError('params.extraRel_xy_mm must be an N x 2 array.')

    # Tolerance preset (microns)
    tol = get_tolerance_preset(params['error_preset'])

    # Manual overrides
    tol_override = params.get('tol_override')
    if tol_override is not None and isinstance(tol_override, dict):
        tol.update(tol_override)

    # Zero-error override
    if params.get('zero_error_override', False):
        tol = dict(
            pin_rad_um=0, pin_tan_um=0, runout_um=0,
            profile_um=0, commonProfile_um=0, commonXY_um=0,
        )

    # Dedicated one-planet-only error mode
    single_planet_mode = params.get('single_planet_error_override', False)
    single_planet_error_m = params.get('single_planet_error_um', 0) * 1e-6

    # Sample / assign error contributors
    if single_planet_mode:
        pin_rad_m = np.zeros(N)
        pin_tan_m = np.zeros(N)
        pin_xy_m = np.zeros((N, 2))

        runout_amp_m = np.zeros(N)
        runout_xy_m = np.zeros((N, 2))

        profile_m = np.zeros(N)
        commonProfile_m = np.zeros(N)
        commonXY_m = np.zeros((N, 2))
    else:
        # 1) Pinhole position errors
        pin_rad_m = sample_err_microns_to_meters(tol['pin_rad_um'], (N,), params, rng)
        pin_tan_m = sample_err_microns_to_meters(tol['pin_tan_um'], (N,), params, rng)

        pin_xy_m = (pin_rad_m.reshape(-1, 1) * eR +
                    pin_tan_m.reshape(-1, 1) * eT)   # [N x 2]

        # 2) Planet runout / eccentricity
        runout_amp_m = np.abs(
            sample_err_microns_to_meters(tol['runout_um'], (N,), params, rng)
        )

        mesh_phase_mode = params.get('meshPhaseMode', 'random').lower()
        if mesh_phase_mode == 'fixed':
            phi0 = np.deg2rad(params.get('meshPhase_deg', 0))
            phi_runout = phi0 * np.ones(N)
        else:
            phi_runout = 2 * math.pi * rng.random(N)

        runout_xy_m = np.column_stack([
            runout_amp_m * np.cos(phi_runout),
            runout_amp_m * np.sin(phi_runout),
        ])

        # 3) Tooth/profile error
        profile_m = sample_err_microns_to_meters(tol['profile_um'], (N,), params, rng)

        # 4) Common-mode errors
        cp_scalar = sample_err_microns_to_meters(tol['commonProfile_um'], (1,), params, rng)
        commonProfile_m = cp_scalar[0] * np.ones(N)

        cxy = sample_err_microns_to_meters(tol['commonXY_um'], (1, 2), params, rng)
        commonXY_m = np.tile(cxy, (N, 1))

    # ----- Collapse all geometric effects onto LOA scalar error -----------
    # Contribution from pin errors
    e_pin_m = -np.sum(pin_xy_m * nLOA, axis=1)

    # Contribution from runout
    e_runout_m = -np.sum(runout_xy_m * nLOA, axis=1)

    # Contribution from rigid assembly offsets
    rigid_xy_m = (commonXY_m
                  + np.tile(carrier_xy_m, (N, 1))
                  - np.tile(sun_xy_m, (N, 1))
                  + extraRel_xy_m)
    e_rigid_m = -np.sum(rigid_xy_m * nLOA, axis=1)

    # Contribution from profile error
    e_profile_m = profile_m + commonProfile_m

    # In dedicated planet-1 mode, inject the user-specified scalar error
    if single_planet_mode:
        e_profile_m = np.zeros(N)
        e_profile_m[0] = single_planet_error_m   # planet 1 = index 0

    # Final static equivalent error
    e_static_m = e_pin_m + e_runout_m + e_rigid_m + e_profile_m

    return dict(
        e_static_m=e_static_m,
        tol_used=tol,
        single_planet_error_override=single_planet_mode,
        single_planet_error_um=single_planet_error_m * 1e6,
        components=dict(
            pin_rad_m=pin_rad_m,
            pin_tan_m=pin_tan_m,
            pin_xy_m=pin_xy_m,
            runout_amp_m=runout_amp_m,
            runout_xy_m=runout_xy_m,
            profile_m=profile_m,
            commonProfile_m=commonProfile_m,
            commonXY_m=commonXY_m,
            e_pin_m=e_pin_m,
            e_runout_m=e_runout_m,
            e_rigid_m=e_rigid_m,
            e_profile_m=e_profile_m,
        ),
    )


# =========================================================================
# SOLVE TUTOR LOAD SHARING (HEART OF THE SOLVER)
# =========================================================================

def solve_tutor_load_sharing(
    theta_rad: np.ndarray,
    k_vec_Nm: np.ndarray,
    r_support_m: float,
    Qext: np.ndarray,
    s_gap_m: np.ndarray,
    k_support_w_Nm: float,
    k_support_phi_NmRad: float,
) -> dict:
    """
    This is the CONNECTED equilibrium solver — the heart of the final model.

    Generalized coordinates:
      q = [w ; phi_x ; phi_y]

    Influence vector for planet i:
      v_i = [1 ;
             r*sin(theta_i) ;
            -r*cos(theta_i)]

    Equilibrium:
      K*q = g

    where:
      K = sum_i [ k_i * v_i * v_i' ] + support terms
      g = Qext + sum_i [ k_i * s_i * v_i ]

    Compression-only / lift-off condition:
      if f_i <= 0, that planet is removed and the system is re-solved.

    Parameters
    ----------
    theta_rad : (N,) planet angular positions
    k_vec_Nm : (N,) effective stiffnesses, N/m
    r_support_m : lever arm radius [m]
    Qext : (3,) generalized external load = [W; Mx; My]
    s_gap_m : (N,) non-negative gap vector [m]
    k_support_w_Nm : optional support stiffness on settlement DOF [N/m]
    k_support_phi_NmRad : optional support stiffness on each tilt DOF [N*m/rad]

    Returns
    -------
    dict with keys: q, K, g, deflection_m, force_N, active
    """
    N = len(theta_rad)

    theta = theta_rad.flatten()
    s_gap = s_gap_m.flatten()
    Qext = Qext.flatten()

    # Working copy of active stiffnesses for lift-off iteration
    k_active = k_vec_Nm.flatten().copy()

    MAX_ITER = N + 2
    converged = False

    q = np.zeros(3)
    Delta = np.zeros(N)
    f = np.zeros(N)
    K = np.zeros((3, 3))
    g = np.zeros(3)

    for _iter in range(MAX_ITER):
        # Build generalized stiffness matrix K and force vector g
        K = np.zeros((3, 3))
        g = np.zeros(3)

        # Add optional support stiffnesses for the generalized DOFs
        K += np.diag([k_support_w_Nm, k_support_phi_NmRad, k_support_phi_NmRad])

        # Assemble contributions from all still-active planets
        for i in range(N):
            if k_active[i] <= 0:
                continue

            v_i = np.array([
                1.0,
                r_support_m * math.sin(theta[i]),
                -r_support_m * math.cos(theta[i]),
            ])

            K += k_active[i] * np.outer(v_i, v_i)
            g += k_active[i] * s_gap[i] * v_i

        # Add external generalized load
        g += Qext

        # Solve K*q = g
        cond = np.linalg.cond(K)
        if cond > 1e14:
            raise RuntimeError(
                'Tutor load-sharing stiffness matrix became singular. '
                'Check geometry, stiffness, support stiffness, or input loads.'
            )
        q = np.linalg.solve(K, g)

        # Recover spring deflections and planet forces
        Delta = np.zeros(N)
        f = np.zeros(N)

        for i in range(N):
            v_i = np.array([
                1.0,
                r_support_m * math.sin(theta[i]),
                -r_support_m * math.cos(theta[i]),
            ])

            # Spring compression / deflection
            Delta[i] = v_i @ q - s_gap[i]

            # Compression-only force
            f[i] = k_active[i] * Delta[i]

        # Lift-off / compression-only logic
        still_active = k_active > 0
        force_tol = 1e-10
        in_compression = f > force_tol

        if np.all(in_compression[still_active]):
            converged = True
            break

        lift_off = still_active & (~in_compression)
        k_active[lift_off] = 0

    if not converged:
        warnings.warn(
            'solveTutorLoadSharing: Compression-only iteration did not fully converge.',
            RuntimeWarning,
        )

    # Clean final forces (never allow negative contact force in output)
    f = np.maximum(f, 0.0)

    return dict(
        q=q,
        K=K,
        g=g,
        deflection_m=Delta,
        force_N=f,
        active=k_active > 0,
    )


# =========================================================================
# PHASE SWEEP
# =========================================================================

def run_phase_sweep(params: dict, geom: dict, err_static: dict) -> dict:
    """
    Runs the quasi-static phase sweep:

      gamma from 0 to 2*pi

    At each phase step:
      1) build periodic eccentricity
      2) build periodic mesh stiffness
      3) combine with static errors
      4) convert equivalent errors into tutor-style gap vector s_i
      5) solve Velex-style load-sharing problem
    """
    N = params['N']
    nPhase = params['phase_steps']

    # Phase vector over one full cycle
    phase_rad = np.linspace(0, 2 * math.pi, nPhase + 1)[:-1]

    # Preallocate history arrays
    force_phase_N = np.zeros((N, nPhase))
    load_share_phase = np.zeros((N, nPhase))
    LSF_phase = np.zeros((N, nPhase))
    active_phase = np.zeros((N, nPhase), dtype=bool)

    sun_q_phase = np.zeros((3, nPhase))
    K_gamma_phase = np.zeros(nPhase)

    e_total_phase_m = np.zeros((N, nPhase))
    gap_phase_m = np.zeros((N, nPhase))

    ksp_phase_Nm = np.zeros((N, nPhase))
    krp_phase_Nm = np.zeros((N, nPhase))
    keff_phase_Nm = np.zeros((N, nPhase))

    ecc_xy_phase_m = np.zeros((2, nPhase))

    # Thermal geometry error (static over one phase sweep)
    thermal_geom = compute_thermal_geometry_error(params, geom)
    e_thermal_phase_m = np.zeros((N, nPhase))

    # Nominal transmitted force
    W_nominal_N = params['T_input_Nm'] / (params['R_sun_mm'] / 1000.0)

    # Generalized external force vector in tutor-style model:
    #   Qext = [W ; Mx ; My]
    Qext = np.array([
        W_nominal_N,
        params['external_Mx_Nm'],
        params['external_My_Nm'],
    ])

    # ----- Loop over all phase steps -------------------------------------
    for j in range(nPhase):
        gamma = phase_rad[j]

        # 1) PERIODIC SUN ECCENTRICITY
        if params['enable_periodic_ecc'] and params['ecc_amp_um'] != 0:
            ecc_amp_m = params['ecc_amp_um'] * 1e-6
            ecc_phase0 = np.deg2rad(params['ecc_phase_deg'])

            ecc_xy = ecc_amp_m * np.array([
                math.cos(params['ecc_order'] * gamma + ecc_phase0),
                math.sin(params['ecc_order'] * gamma + ecc_phase0),
            ])
        else:
            ecc_xy = np.zeros(2)

        ecc_xy_phase_m[:, j] = ecc_xy

        # Project rotating eccentricity onto each LOA
        e_ecc_m = geom['nLOA'] @ ecc_xy   # [N]

        # 2) PERIODIC MESH STIFFNESS
        psi = geom['psi_rad']

        if params['enable_periodic_stiffness']:
            phase_shift = params['tvms_planet_phase_scale'] * psi

            kSP_i = (geom['mesh']['ksp_mean_Nm'] +
                     params['tvms_amp_scale'] * geom['mesh']['ksp_amp_Nm'] *
                     np.cos(params['tvms_order'] * gamma + phase_shift))

            kRP_i = (geom['mesh']['krp_mean_Nm'] +
                     params['tvms_amp_scale'] * geom['mesh']['krp_amp_Nm'] *
                     np.cos(params['tvms_order'] * gamma + phase_shift))
        else:
            kSP_i = geom['mesh']['ksp_mean_Nm'] * np.ones(N)
            kRP_i = geom['mesh']['krp_mean_Nm'] * np.ones(N)

        # Guard against negative / zero stiffness
        kSP_i = np.maximum(kSP_i, params['min_stiffness_Nm'])
        kRP_i = np.maximum(kRP_i, params['min_stiffness_Nm'])

        kB_i = geom['kB_vec_Nm']

        # Final k_eff equation from literature
        kEff_i = 1.0 / (1.0 / (kSP_i + kRP_i) + 1.0 / kB_i)

        # 3) TOTAL EQUIVALENT ERROR
        e_total_i = (err_static['e_static_m'] + e_ecc_m +
                     thermal_geom['e_th_m'])

        # 4) CONVERT EQUIVALENT ERRORS TO GAP VECTOR s_i
        #    s_i = max(e) - e_i
        s_i = np.max(e_total_i) - e_total_i

        # 5) SOLVE TUTOR-STYLE LOAD SHARING
        step = solve_tutor_load_sharing(
            geom['psi_rad'],
            kEff_i,
            geom['r_support_m'],
            Qext,
            s_i,
            params['k_support_w_Nm'],
            params['k_support_phi_NmRad'],
        )

        # 6) POST-PROCESS LSF
        f_i = step['force_N']
        total_f = np.sum(f_i)

        if total_f > 0:
            share_i = f_i / total_f
            mean_f = total_f / N
            LSF_i = f_i / mean_f
            K_gamma = np.max(f_i) / mean_f
        else:
            share_i = np.zeros(N)
            LSF_i = np.zeros(N)
            K_gamma = 0.0

        # Store histories
        force_phase_N[:, j] = f_i
        load_share_phase[:, j] = share_i
        LSF_phase[:, j] = LSF_i
        active_phase[:, j] = step['active']
        sun_q_phase[:, j] = step['q']
        K_gamma_phase[j] = K_gamma

        e_total_phase_m[:, j] = e_total_i
        gap_phase_m[:, j] = s_i

        ksp_phase_Nm[:, j] = kSP_i
        krp_phase_Nm[:, j] = kRP_i
        keff_phase_Nm[:, j] = kEff_i
        e_thermal_phase_m[:, j] = thermal_geom['e_th_m']

    return dict(
        phase_rad=phase_rad,
        W_nominal_N=W_nominal_N,
        force_phase_N=force_phase_N,
        load_share_phase=load_share_phase,
        LSF_phase=LSF_phase,
        active_phase=active_phase,
        sun_q_phase=sun_q_phase,
        K_gamma_phase=K_gamma_phase,
        e_total_phase_m=e_total_phase_m,
        gap_phase_m=gap_phase_m,
        ksp_phase_Nm=ksp_phase_Nm,
        krp_phase_Nm=krp_phase_Nm,
        keff_phase_Nm=keff_phase_Nm,
        ecc_xy_phase_m=ecc_xy_phase_m,
        e_thermal_phase_m=e_thermal_phase_m,
        thermal_geometry=thermal_geom,
    )


# =========================================================================
# SOLVE SINGLE CASE
# =========================================================================

def solve_single_case(params: dict) -> dict:
    """
    Runs ONE complete analysis:
      1) geometry + stiffness
      2) one static error realization
      3) phase sweep with periodic eccentricity / periodic stiffness
      4) extract worst phase and summary outputs
    """
    # Geometry and stiffness baseline
    geom = build_geometry(params)

    # One static error realization
    err_static = generate_static_errors(params, geom)

    # Solve the phase sweep
    hist = run_phase_sweep(params, geom, err_static)

    # Find worst phase using system load-sharing factor
    idx_worst = int(np.argmax(hist['K_gamma_phase']))
    K_gamma_max = hist['K_gamma_phase'][idx_worst]

    # Pack output structure
    single = dict(
        N=params['N'],
        phase_rad=hist['phase_rad'],
        worst_phase_index=idx_worst,
        worst_phase_rad=hist['phase_rad'][idx_worst],
        W_nominal_N=hist['W_nominal_N'],

        # Full histories over one phase
        force_phase_N=hist['force_phase_N'],
        load_share_phase=hist['load_share_phase'],
        LSF_phase=hist['LSF_phase'],
        K_gamma_phase=hist['K_gamma_phase'],
        sun_q_phase=hist['sun_q_phase'],
        active_phase=hist['active_phase'],
        e_total_phase_m=hist['e_total_phase_m'],
        gap_phase_m=hist['gap_phase_m'],
        ksp_phase_Nm=hist['ksp_phase_Nm'],
        krp_phase_Nm=hist['krp_phase_Nm'],
        keff_phase_Nm=hist['keff_phase_Nm'],
        ecc_xy_phase_m=hist['ecc_xy_phase_m'],

        # Final / worst-phase values
        F_final_N=hist['force_phase_N'][:, idx_worst],
        load_share_final=hist['load_share_phase'][:, idx_worst],
        LSF_final=hist['LSF_phase'][:, idx_worst],
        active_final=hist['active_phase'][:, idx_worst],
        sun_displacement_final=hist['sun_q_phase'][:, idx_worst],

        # System summary metrics
        K_gamma_max=K_gamma_max,
        K_gamma_mean=float(np.mean(hist['K_gamma_phase'])),

        # Static ingredients
        geometry=geom,
        static_errors=err_static,

        # Location of each gear around a circle
        planet_angles_rad=geom['psi_rad'],
        planet_angles_deg=np.rad2deg(geom['psi_rad']),
        load_share_percent_final=100.0 * hist['load_share_phase'][:, idx_worst],
        K_gamma_span=float(np.max(hist['K_gamma_phase']) - np.min(hist['K_gamma_phase'])),
        zero_error_override=params.get('zero_error_override', False),
        periodic_ecc_enabled=params.get('enable_periodic_ecc', False),
        periodic_stiffness_enabled=params.get('enable_periodic_stiffness', False),

        sunOffset_xy_mm=np.asarray(params['sunOffset_xy_mm']).flatten(),
        carrierOffset_xy_mm=np.asarray(params['carrierOffset_xy_mm']).flatten(),
        single_planet_error_override=params.get('single_planet_error_override', False),
        single_planet_error_um=params.get('single_planet_error_um', 0),
    )

    if 'thermal_geometry' in hist:
        single['thermal_geometry'] = hist['thermal_geometry']

    return single


# =========================================================================
# EXTRACT SWEEP METRICS
# =========================================================================

def extract_sweep_metrics(case_result: dict) -> dict:
    """
    Compresses a full case result into a few scalar metrics that are useful
    for sensitivity plots later.
    """
    q = case_result['sun_q_phase']   # [3 x nPhase]

    return dict(
        K_gamma_max=float(np.max(case_result['K_gamma_phase'])),
        max_LSF_any=float(np.max(case_result['LSF_phase'])),
        max_abs_w_um=float(np.max(np.abs(q[0, :]))) * 1e6,
        max_abs_phi_x_mrad=float(np.max(np.abs(q[1, :]))) * 1e3,
        max_abs_phi_y_mrad=float(np.max(np.abs(q[2, :]))) * 1e3,
        max_force_N=float(np.max(case_result['force_phase_N'])),
    )


# =========================================================================
# SENSITIVITY SWEEPS
# =========================================================================

def run_sensitivity_sweeps(params: dict) -> dict:
    """
    Generates DATA for future plotting.

    Three sweep families:
      1) periodic eccentricity amplitude
      2) mesh stiffness scale factor
      3) bearing stiffness scale factor
    """
    base_seed = params['sensitivity_seed']

    p_base = copy.deepcopy(params)
    p_base['seed'] = base_seed

    # Baseline
    base = solve_single_case(p_base)

    sensitivity = dict(baseline=extract_sweep_metrics(base))

    # 1) Eccentricity amplitude sweep
    ecc_vals = np.asarray(params['sensitivity_ecc_values_um']).flatten()
    ecc_metrics = []
    for val in ecc_vals:
        p = copy.deepcopy(p_base)
        p['ecc_amp_um'] = val
        out = solve_single_case(p)
        ecc_metrics.append(extract_sweep_metrics(out))

    sensitivity['eccentricity'] = dict(values_um=ecc_vals, metrics=ecc_metrics)

    # 2) Mesh stiffness scale sweep
    mesh_vals = np.asarray(params['sensitivity_mesh_scale_values']).flatten()
    mesh_metrics = []
    for val in mesh_vals:
        p = copy.deepcopy(p_base)
        p['mesh_scale_factor'] = val
        out = solve_single_case(p)
        mesh_metrics.append(extract_sweep_metrics(out))

    sensitivity['mesh_scale'] = dict(values=mesh_vals, metrics=mesh_metrics)

    # 3) Bearing stiffness scale sweep
    bearing_vals = np.asarray(params['sensitivity_bearing_scale_values']).flatten()
    bearing_metrics = []
    for val in bearing_vals:
        p = copy.deepcopy(p_base)
        p['bearing_scale_factor'] = val
        out = solve_single_case(p)
        bearing_metrics.append(extract_sweep_metrics(out))

    sensitivity['bearing_scale'] = dict(values=bearing_vals, metrics=bearing_metrics)

    return sensitivity


# =========================================================================
# MONTE CARLO
# =========================================================================

def run_monte_carlo(params: dict) -> dict:
    """
    Optional Monte Carlo wrapper.
    Repeats the full single-case analysis several times, each time with a
    different stochastic static error realization.
    """
    n_runs = params['nMonteCarlo']
    N = params['N']

    K_gamma_max_all = np.zeros(n_runs)
    worst_phase_all = np.zeros(n_runs)
    max_LSF_any_all = np.zeros(n_runs)

    sun_q_worst_all = np.zeros((3, n_runs))
    LSF_worst_all = np.zeros((N, n_runs))

    for r in range(n_runs):
        p = copy.deepcopy(params)
        # Use a deterministic sequence of seeds for reproducibility
        p['seed'] = params['monte_carlo_base_seed'] + r

        out = solve_single_case(p)

        K_gamma_max_all[r] = out['K_gamma_max']
        worst_phase_all[r] = out['worst_phase_rad']
        max_LSF_any_all[r] = float(np.max(out['LSF_phase']))

        sun_q_worst_all[:, r] = out['sun_displacement_final']
        LSF_worst_all[:, r] = out['LSF_final']

    return dict(
        nRuns=n_runs,
        K_gamma_max_all=K_gamma_max_all,
        worst_phase_rad_all=worst_phase_all,
        max_LSF_any_all=max_LSF_any_all,
        sun_q_worst_all=sun_q_worst_all,
        LSF_worst_all=LSF_worst_all,
    )


# =========================================================================
# INPUT VALIDATION
# =========================================================================

def validate_inputs(params: dict) -> None:
    """
    Basic input validation.
    Intentionally simple, focuses only on essential correctness.
    """
    if params['N'] < 3 or int(params['N']) != params['N']:
        raise ValueError('params.N must be an integer >= 3.')

    if params['T_input_Nm'] <= 0:
        raise ValueError('params.T_input_Nm must be > 0.')

    if params['R_sun_mm'] <= 0:
        raise ValueError('params.R_sun_mm must be > 0.')

    if params['phase_steps'] < 3:
        raise ValueError('params.phase_steps must be >= 3.')

    if params['z1'] <= 0 or params['z2'] <= 0 or params['z3'] <= 0:
        raise ValueError('z1, z2, z3 must all be positive.')

    if params['m0_mm'] <= 0 or params['aw_mm'] <= 0 or params['b_face_mm'] <= 0:
        raise ValueError('m0_mm, aw_mm, and b_face_mm must all be > 0.')

    # Reduced-order planetary geometry consistency checks
    beta = np.deg2rad(params['beta_deg'])
    mt = params['m0_mm'] / math.cos(beta)

    aw_nom_sp = mt * (params['z1'] + params['z2']) / 2
    aw_nom_rp = mt * (params['z3'] - params['z2']) / 2

    if abs(aw_nom_sp - aw_nom_rp) > 1e-9:
        raise ValueError(
            'Inconsistent tooth numbers for the simplified planetary geometry: '
            'expected sun-planet and ring-planet center distances to match.'
        )

    if abs(params['aw_mm'] - aw_nom_sp) > 1e-6:
        warnings.warn(
            'aw_mm is not consistent with z1, z2, z3, m0_mm, and beta_deg '
            'for the simplified geometry model. Results may be non-physical.',
            RuntimeWarning,
        )

    if params['r_support_mm'] <= 0:
        raise ValueError('params.r_support_mm must be > 0.')

    if params['thermal_gradient_ratio'] < 0:
        raise ValueError('params.thermal_gradient_ratio must be >= 0.')

    if params['thermal_pin_weight'] < 0 or params['thermal_planet_weight'] < 0:
        raise ValueError('Thermal geometry weights must be >= 0.')


# =========================================================================
# NORMALIZE INPUTS (FILL DEFAULTS)
# =========================================================================

def normalize_inputs(inputs: dict) -> dict:
    """
    Fills in all defaults so that the solver can run with a compact user
    input structure.

    This function is deliberately verbose so that the variable names are easy
    to inspect and change later.
    """
    p = dict(inputs)  # shallow copy

    def d(key, default):
        """Return p[key] if it exists and is not None/empty, else default."""
        val = p.get(key)
        if val is None:
            return default
        # Handle empty sequences
        if isinstance(val, (list, np.ndarray)) and len(val) == 0:
            return default
        if isinstance(val, str) and val.strip() == '':
            return default
        return val

    # BASIC SYSTEM INPUTS
    p['N'] = int(d('N', 5))
    p['T_input_Nm'] = float(d('T_input_Nm', 1000))
    p['R_sun_mm'] = float(d('R_sun_mm', 50))
    p['alpha_deg'] = float(d('alpha_deg', 20))
    p['psi_angles_deg'] = d('psi_angles_deg', [])

    # STIFFNESS / GEAR GEOMETRY INPUTS
    p['z1'] = int(d('z1', 73))
    p['z2'] = int(d('z2', 26))
    p['z3'] = int(d('z3', 125))

    p['m0_mm'] = float(d('m0_mm', 2))
    p['alpha_n_deg'] = float(d('alpha_n_deg', p['alpha_deg']))
    p['beta_deg'] = float(d('beta_deg', 0))
    p['aw_mm'] = float(d('aw_mm', 99))
    p['b_face_mm'] = float(d('b_face_mm', 25))

    p['x1'] = float(d('x1', 0))
    p['x2'] = float(d('x2', 0))

    p['r_support_mm'] = float(d('r_support_mm', p['R_sun_mm']))

    # BEARING MODEL INPUTS
    p['bearing_type'] = d('bearing_type', 'ball_noclearance')
    p['mesh_scale_factor'] = float(d('mesh_scale_factor', 1.0))
    p['bearing_scale_factor'] = float(d('bearing_scale_factor', 1.0))
    p['min_stiffness_Nm'] = float(d('min_stiffness_Nm', 1.0e3))

    # TEMPERATURE / DYNAMIC EFFECTS
    p['enable_temperature_effects'] = bool(d('enable_temperature_effects', True))
    p['temperature_C'] = float(d('temperature_C', 20))
    p['temperature_ref_C'] = float(d('temperature_ref_C', 20))

    p['lambda_thermal_1C'] = float(d('lambda_thermal_1C', 11.5e-6))
    p['E_Pa'] = float(d('E_Pa', 206e9))
    p['nu'] = float(d('nu', 0.3))
    p['r0_mm'] = float(d('r0_mm', 20))

    p['thermal_E_drop_per_C'] = float(d('thermal_E_drop_per_C', 4e-4))
    p['thermal_min_scale'] = float(d('thermal_min_scale', 0.75))
    p['thermal_max_scale'] = float(d('thermal_max_scale', 1.05))

    # THERMAL GEOMETRY ERROR MODEL
    p['enable_thermal_geometry_error'] = bool(d('enable_thermal_geometry_error', True))
    p['use_auto_thermal_gradient'] = bool(d('use_auto_thermal_gradient', True))
    p['thermal_gradient_ratio'] = float(d('thermal_gradient_ratio', 0.10))
    p['thermal_gradient_C'] = float(d('thermal_gradient_C', 0))
    p['thermal_hotspot_deg'] = float(d('thermal_hotspot_deg', 0))
    p['thermal_pin_weight'] = float(d('thermal_pin_weight', 1.0))
    p['thermal_planet_weight'] = float(d('thermal_planet_weight', 0.7))

    # STATIC ASSEMBLY / OFFSET INPUTS
    p['sunOffset_xy_mm'] = np.asarray(d('sunOffset_xy_mm', [0.0, 0.0]), dtype=float).flatten()
    p['carrierOffset_xy_mm'] = np.asarray(d('carrierOffset_xy_mm', [0.0, 0.0]), dtype=float).flatten()
    p['extraRel_xy_mm'] = d('extraRel_xy_mm', [])

    # STATIC ERROR MODEL INPUTS
    p['error_preset'] = d('error_preset', 'medium')
    p['zero_error_override'] = bool(d('zero_error_override', False))
    p['single_planet_error_override'] = bool(d('single_planet_error_override', False))
    p['single_planet_error_um'] = float(d('single_planet_error_um', 0))

    p['seed'] = d('seed', None)
    p['tolMeaning'] = d('tolMeaning', 'pm')
    p['sigmaRule'] = float(d('sigmaRule', 3))
    p['dist'] = d('dist', 'normal')
    p['truncate'] = bool(d('truncate', True))

    p['meshPhaseMode'] = d('meshPhaseMode', 'random')
    p['meshPhase_deg'] = float(d('meshPhase_deg', 0))
    p['tol_override'] = d('tol_override', {})

    # PERIODIC ECCENTRICITY INPUTS
    p['enable_periodic_ecc'] = bool(d('enable_periodic_ecc', False))
    p['ecc_amp_um'] = float(d('ecc_amp_um', 0))
    p['ecc_phase_deg'] = float(d('ecc_phase_deg', 0))
    p['ecc_order'] = float(d('ecc_order', 1))

    # PERIODIC MESH STIFFNESS INPUTS
    p['enable_periodic_stiffness'] = bool(d('enable_periodic_stiffness', False))
    p['tvms_amp_scale'] = float(d('tvms_amp_scale', 1.0))
    p['tvms_order'] = float(d('tvms_order', 1))
    p['tvms_planet_phase_scale'] = float(d('tvms_planet_phase_scale', 1.0))

    # PHASE SWEEP INPUTS
    p['phase_steps'] = int(d('phase_steps', 91))

    # GENERALIZED EXTERNAL LOADS
    p['external_Mx_Nm'] = float(d('external_Mx_Nm', 0))
    p['external_My_Nm'] = float(d('external_My_Nm', 0))
    p['k_support_w_Nm'] = float(d('k_support_w_Nm', 0))
    p['k_support_phi_NmRad'] = float(d('k_support_phi_NmRad', 0))

    # MONTE CARLO OPTIONS
    p['run_monte_carlo'] = bool(d('run_monte_carlo', False))
    p['nMonteCarlo'] = int(d('nMonteCarlo', 50))
    p['monte_carlo_base_seed'] = int(d('monte_carlo_base_seed', 1001))

    # SENSITIVITY OPTIONS
    p['run_sensitivity'] = bool(d('run_sensitivity', False))
    p['sensitivity_seed'] = int(d('sensitivity_seed', 4242))
    p['sensitivity_ecc_values_um'] = d('sensitivity_ecc_values_um', [0, 5, 10, 20, 40, 80])
    p['sensitivity_mesh_scale_values'] = d('sensitivity_mesh_scale_values',
                                           [0.70, 0.85, 1.00, 1.15, 1.30])
    p['sensitivity_bearing_scale_values'] = d('sensitivity_bearing_scale_values',
                                              [0.70, 0.85, 1.00, 1.15, 1.30])

    return p


# =========================================================================
# MAIN ANALYSIS DRIVER
# =========================================================================

def run_analysis_internal(params: dict) -> dict:
    """
    Master internal driver:
      - validates inputs
      - runs one baseline single-case analysis
      - optionally runs sensitivity sweeps
      - optionally runs Monte Carlo
    """
    validate_inputs(params)

    # 1) Run the main / baseline case
    baseline = solve_single_case(params)
    results = dict(baseline)

    # 2) Optional sensitivity study
    if params['run_sensitivity']:
        results['sensitivity'] = run_sensitivity_sweeps(params)
    else:
        results['sensitivity'] = None

    # 3) Optional Monte Carlo study
    if params['run_monte_carlo'] and params['nMonteCarlo'] > 0:
        results['monte_carlo'] = run_monte_carlo(params)
    else:
        results['monte_carlo'] = None

    return results


# =========================================================================
# PUBLIC CLASS
# =========================================================================

class EpicyclicGearSystem:
    """
    Epicyclic Gear System solver (Version 4).

    Usage:
        sys = EpicyclicGearSystem({'N': 5, 'T_input_Nm': 1000, ...})
        sys.run()
        sys.print_summary()
        res = sys.results
    """

    def __init__(self, inputs: Optional[dict] = None):
        if inputs is None:
            inputs = {}
        self.params = normalize_inputs(inputs)
        self.results: Optional[dict] = None

    def run(self) -> None:
        """Main entry point: runs the complete numerical workflow."""
        self.params = normalize_inputs(self.params)
        self.results = run_analysis_internal(self.params)

    def print_summary(self) -> None:
        """Simple console summary for debugging / validation."""
        if self.results is None:
            print('\nNo results available. Run the system first.\n')
            return

        res = self.results

        print('\n' + '=' * 52)
        print(' E P I C Y C L I C   G E A R   S Y S T E M   V4')
        print('=' * 52)

        print(f"Number of planets, N                 = {res['N']}")
        print(f"Nominal transmitted force, W [N]     = {res['W_nominal_N']:.6g}")
        print(f"Worst phase index                    = {res['worst_phase_index']}")
        print(f"Worst phase angle [deg]              = {np.rad2deg(res['worst_phase_rad']):.3f}")
        print(f"Maximum K_gamma                      = {res['K_gamma_max']:.6f}")
        print(f"K_gamma span over phase            = {res['K_gamma_span']:.12g}")

        print('\nSun generalized displacement at worst phase:')
        print(f"  w      [um]                        = {res['sun_displacement_final'][0] * 1e6:.6f}")
        print(f"  phi_x  [mrad]                      = {res['sun_displacement_final'][1] * 1e3:.6f}")
        print(f"  phi_y  [mrad]                      = {res['sun_displacement_final'][2] * 1e3:.6f}")

        print('\nPlanet-by-planet results at worst phase:')
        print('Planet   Angle [deg]   Force [N]   Share [%]   LSF [-]   Active')
        print('---------------------------------------------------------------')
        for i in range(res['N']):
            print(f"{i+1:4d}   {np.rad2deg(res['geometry']['psi_rad'][i]):11.2f}"
                  f"   {res['F_final_N'][i]:11.4f}"
                  f"   {100*res['load_share_final'][i]:9.4f}"
                  f"   {res['LSF_final'][i]:8.6f}"
                  f"   {int(res['active_final'][i])}")
        print('-------------------------------------------------------')

        mc = res.get('monte_carlo')
        if mc is not None and isinstance(mc, dict) and 'nRuns' in mc:
            print('\nMonte Carlo summary:')
            print(f"  Number of MC runs                 = {mc['nRuns']}")
            print(f"  Mean max K_gamma                  = {np.mean(mc['K_gamma_max_all']):.6f}")
            print(f"  Std  max K_gamma                  = {np.std(mc['K_gamma_max_all']):.6f}")

        print('=' * 52 + '\n')
