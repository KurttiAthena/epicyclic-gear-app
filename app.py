"""
Streamlit UI for the Epicyclic Gear System solver (V4).
Replaces the MATLAB EpicyclicGearAppV4.m GUI with a web-based interface.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from epicyclic_gear_system import EpicyclicGearSystem

st.set_page_config(page_title="Epicyclic Gear System Analysis - V4", layout="wide")

# =========================================================================
# HELPER: Plotly figure defaults
# =========================================================================
def _fig_layout(fig: go.Figure, title: str = "", xaxis_title: str = "",
                yaxis_title: str = "", height: int = 350, **kwargs) -> go.Figure:
    
    layout_args = dict(
        title=dict(text=title, font=dict(size=15, color="black"), x=0.5, y=0.95, xanchor='center', yanchor='top'),
        xaxis_title=dict(text=xaxis_title, font=dict(size=12, color="black")),
        yaxis_title=dict(text=yaxis_title, font=dict(size=12, color="black")),
        height=height,
        template="plotly_white",
        margin=dict(l=50, r=30, t=60, b=60),
        font=dict(size=11, color="black"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h", yanchor="top", y=-0.25, x=0.5, xanchor="center",
            bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1
        )
    )
    layout_args.update(kwargs)
    
    fig.update_layout(**layout_args)
    fig.update_xaxes(gridcolor="#e0e0e0", zeroline=True, zerolinecolor="#888", showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(gridcolor="#e0e0e0", zeroline=True, zerolinecolor="#888", showline=True, linewidth=1, linecolor='black', mirror=True)
    return fig

def _auto_y_range(y_data: np.ndarray, pad_frac: float = 0.15):
    y_clean = y_data[np.isfinite(y_data)]
    if len(y_clean) == 0: return None
    y_min, y_max = float(np.min(y_clean)), float(np.max(y_clean))
    span = y_max - y_min
    if span < 1e-4: pad = max(1e-4, 0.02 * max(abs(y_min), 1))
    else: pad = pad_frac * span
    return [y_min - pad, y_max + pad]

_PHASE_TICKS = dict(
    tickmode='array',
    tickvals=[0, 90, 180, 270, 360],
    ticktext=["0° / 0", "90° / π/2", "180° / π", "270° / 3π/2", "360° / 2π"],
)

# =========================================================================
# PLOTTING FUNCTIONS
# =========================================================================
def plot_lsf_bar(res: dict) -> go.Figure:
    N = res['N']
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1, N + 1)), y=res['LSF_final'], marker_color="#2e86c1", showlegend=False))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#e74c3c", annotation_text="Ideal = 1.0", annotation_position="top right")
    y_range = _auto_y_range(np.array(res['LSF_final']))
    _fig_layout(fig, "Final Load Sharing Factor", "Planet Index", "LSF [-]", yaxis_range=y_range)
    fig.update_xaxes(dtick=1)
    return fig

def plot_error_components(res: dict) -> go.Figure:
    if ('static_errors' not in res or res['static_errors'] is None or 'components' not in res['static_errors']):
        return go.Figure().add_annotation(text="No error decomposition available", x=0.5, y=0.5, showarrow=False)
    N = res['N']
    c = res['static_errors']['components']
    e_pin = c['e_pin_m'] * 1e6
    e_runout = c['e_runout_m'] * 1e6
    e_rigid = c['e_rigid_m'] * 1e6
    e_profile = c['e_profile_m'] * 1e6
    idx_worst = res['worst_phase_index']
    e_total_worst = res['e_total_phase_m'][:, idx_worst] * 1e6
    e_static = res['static_errors']['e_static_m'] * 1e6
    e_thermal = res['thermal_geometry']['e_th_m'] * 1e6 if res.get('thermal_geometry') else np.zeros(N)
    e_periodic = e_total_worst - e_static - e_thermal

    colors = ['#4285F4', '#F4B400', '#0F9D58', '#AB47BC', '#DB4437', '#00ACC1']
    names = ['Pin', 'Runout', 'Rigid/Offset', 'Profile/User', 'Periodic ecc.', 'Thermal']
    data = [e_pin, e_runout, e_rigid, e_profile, e_periodic, e_thermal]

    fig = go.Figure()
    for y_vals, name, color in zip(data, names, colors):
        fig.add_trace(go.Bar(x=list(range(1, N + 1)), y=y_vals, name=name, marker_color=color))
    fig.add_trace(go.Scatter(x=list(range(1, N + 1)), y=e_total_worst, mode='lines+markers', name='Total', line=dict(color='black', width=2), marker=dict(size=6, color='black')))
    fig.update_layout(barmode='relative')
    max_abs = max(np.max(np.abs(e_total_worst)), 1)
    _fig_layout(fig, "Error Component Breakdown (Worst Phase)", "Planet Index", "Equivalent error [μm]", yaxis_range=[-1.15 * max_abs, 1.15 * max_abs], margin=dict(t=60, b=90, l=50, r=20))
    fig.update_xaxes(dtick=1)
    return fig

def plot_schematic(res: dict, R_sun_mm: float) -> go.Figure:
    N = res['N']
    r_plot = 1.8 * R_sun_mm
    fig = go.Figure()
    th = np.linspace(0, 2 * math.pi, 100)
    
    # Sun
    fig.add_trace(go.Scatter(x=R_sun_mm * np.cos(th), y=R_sun_mm * np.sin(th), mode='lines', line=dict(color='red', width=2), name='Sun', showlegend=False))
    
    # Planets (Draw circles to look more like the MATLAB gears)
    psi = res['planet_angles_rad']
    lsf = res['LSF_final']
    
    for i in range(N):
        px, py = r_plot * math.cos(psi[i]), r_plot * math.sin(psi[i])
        r_planet = 0.4 * R_sun_mm
        if np.max(lsf) - np.min(lsf) > 1e-4:
            if abs(lsf[i] - np.max(lsf)) < 1e-4:
                color = '#333333'
                r_planet = 0.55 * R_sun_mm
            elif abs(lsf[i] - np.min(lsf)) < 1e-4:
                color = '#cccccc'
                r_planet = 0.25 * R_sun_mm
            else:
                color = '#888888'
        else:
            color = '#888888'
            
        fig.add_trace(go.Scatter(x=px + r_planet * np.cos(th), y=py + r_planet * np.sin(th), fill="toself", mode='lines', line=dict(color='black', width=1), fillcolor=color, showlegend=False))
        fig.add_annotation(x=px, y=py, text=f"P{i+1}", showarrow=False, font=dict(color="white" if color == '#333333' else "black", size=9))
        
    # Tilt
    phi_x, phi_y = res['sun_displacement_final'][1], res['sun_displacement_final'][2]
    tilt_norm = math.sqrt(phi_x**2 + phi_y**2)
    if tilt_norm > 1e-15:
        tilt_len = 0.9 * R_sun_mm
        fig.add_annotation(x=tilt_len*phi_x/tilt_norm, y=tilt_len*phi_y/tilt_norm, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.5, arrowcolor="blue", arrowwidth=2)
    
    _fig_layout(fig, "System Schematic (qualitative)", "X [mm]", "Y [mm]")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

def plot_phase_kgamma(res: dict, mode: str = 'Raw K_gamma') -> go.Figure:
    phase_deg, worst_deg = np.rad2deg(res['phase_rad']), np.rad2deg(res['worst_phase_rad'])
    K_raw = res['K_gamma_phase'].flatten()
    k_mean = np.mean(K_raw)
    
    if mode == "Normalized K_gamma":
        y_data = K_raw / k_mean
        y_label = "Normalized K_γ [-]"
    elif mode == "Delta from mean":
        y_data = K_raw - k_mean
        y_label = "ΔK_γ [-]"
    elif mode == "Percent variation":
        y_data = ((K_raw - k_mean) / k_mean) * 100.0
        y_label = "% Variation K_γ"
    else: # Raw
        y_data = K_raw
        y_label = "K_γ [-]"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=phase_deg, y=y_data, mode='lines', line=dict(width=2, color="#2e86c1"), showlegend=False))
    fig.add_vline(x=worst_deg, line_dash="dash", line_color="red", annotation_text="Worst phase", annotation_textangle=-90)
    
    _fig_layout(fig, f"Phase Response: K_γ vs Phase ({mode})", "Phase angle [deg / rad]", y_label, yaxis_range=_auto_y_range(y_data))
    fig.update_xaxes(range=[0, 360], **_PHASE_TICKS)
    return fig

def plot_phase_planet(res: dict, key: str, title: str, ylabel: str) -> go.Figure:
    phase_deg, worst_deg = np.rad2deg(res['phase_rad']), np.rad2deg(res['worst_phase_rad'])
    fig = go.Figure()
    colors = px.colors.qualitative.D3
    for i in range(res['N']):
        y = res[key][i, :]
        if key == 'e_total_phase_m': y = y * 1e6
        fig.add_trace(go.Scatter(x=phase_deg, y=y, mode='lines', line=dict(width=1.5, color=colors[i%len(colors)]), name=f"P{i+1}"))
    fig.add_vline(x=worst_deg, line_dash="dash", line_color="red", annotation_text="Worst", annotation_textangle=-90)
    if key == 'LSF_phase': fig.add_hline(y=1.0, line_dash="dot", line_color="black")
    if key == 'e_total_phase_m': fig.add_hline(y=0.0, line_dash="dot", line_color="black")
    data = res[key] if key != 'e_total_phase_m' else res[key]*1e6
    _fig_layout(fig, title, "Phase angle [deg / rad]", ylabel, yaxis_range=_auto_y_range(data.flatten()))
    fig.update_xaxes(range=[0, 360], **_PHASE_TICKS)
    return fig
  
def plot_mc_hist(mc: dict) -> go.Figure:
    # Changed nbinsx from 16 to 25 for more detailed bars
    fig = go.Figure(data=[go.Histogram(x=mc['K_gamma_max_all'], nbinsx=25, marker_color="#4285F4", marker_line=dict(color="white", width=1))])
    fig = _fig_layout(fig, "Monte Carlo: K_γ,max distribution", "K_γ,max [-]", "Count")
    
    # Force Plotly to draw a lot more ticks (e.g., ~15 ticks) on the X axis
    fig.update_xaxes(nticks=15)
    
    return fig

def plot_mc_box(mc: dict, N: int) -> go.Figure:
    fig = go.Figure()
    for i in range(N):
        fig.add_trace(go.Box(y=mc['LSF_worst_all'][i, :], name=f"{i+1}", marker_color=px.colors.qualitative.D3[i % 10], boxmean=True))
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Ideal=1")
    return _fig_layout(fig, "Monte Carlo: worst-phase LSF by planet", "Planet index", "Worst-phase LSF [-]")

def plot_trend(res: dict, k_w: float, is_stiff: bool) -> go.Figure:
    geom = res.get('geometry')
    if not geom or 'kEff0_vec_Nm' not in geom: return go.Figure().add_annotation(text="No data", showarrow=False)
    k_eff_mean = float(np.mean(geom['kEff0_vec_Nm']))
    N_cur = res['N']
    N_rng = np.arange(3, max(8, N_cur + 3) + 1)
    K = (k_w + N_rng * k_eff_mean)
    K_cur = (k_w + N_cur * k_eff_mean)
    
    if is_stiff:
        y, y_cur, ylab, tit = K / 1e6, K_cur / 1e6, "Nominal settlement stiffness [MN/m]", "Nominal stiffness trend vs number of planets"
    else:
        y, y_cur, ylab, tit = (res['W_nominal_N'] / np.maximum(K, 1e-9))*1e6, (res['W_nominal_N'] / max(K_cur, 1e-9))*1e6, "Nominal deflection [μm]", "Nominal deflection trend vs number of planets"
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=N_rng, y=y, mode='lines+markers', marker=dict(size=7, color="#2e86c1"), showlegend=False))
    fig.add_trace(go.Scatter(x=[N_cur], y=[y_cur], mode='markers', marker=dict(size=12, color="red"), name=f"Current run: N={N_cur}"))
    
    fig = _fig_layout(fig, tit, "Number of planets", ylab)
    
    # ADDED THIS LINE: Forces the X-axis to only show integer steps (1 by 1)
    fig.update_xaxes(dtick=1)
    
    return fig

def plot_sens(sens: dict, key: str, xlab: str, tit: str, sym: str) -> go.Figure:
    if not sens or key not in sens: return go.Figure().add_annotation(text="Disabled", showarrow=False)
    s = sens[key]
    x, y = s.get('values_um', s.get('values', [])), [m['K_gamma_max'] for m in s['metrics']]
    fig = go.Figure(data=[go.Scatter(x=x, y=y, mode='lines+markers', marker=dict(size=8, color="#2e86c1", symbol=sym), line=dict(width=2), showlegend=False)])
    return _fig_layout(fig, tit, xlab, "Max K_γ [-]", margin=dict(l=40, r=20, t=50, b=50))

# =========================================================================
# UI LAYOUT & INPUTS
# =========================================================================

def collect_inputs():
    inputs = {}
    with st.expander("**USER INPUTS CONFIGURATION** (Click to Expand / Collapse)", expanded=True):
        st.markdown("---")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Basic Inputs**")
            inputs['N'] = st.number_input("Number of planets", 3, 12, 5)
            inputs['T_input_Nm'] = st.number_input("Input torque [Nm]", value=1000.0)
            inputs['R_sun_mm'] = st.number_input("Sun radius [mm]", value=50.0)
            inputs['alpha_n_deg'] = float(st.selectbox("Normal pressure angle", ["14", "20", "25"], index=1))
            inputs['alpha_deg'] = inputs['alpha_n_deg']
            inputs['bearing_type'] = st.selectbox("Bearing type", ['ball_clearance', 'ball_noclearance', 'standard', 'roller_clearance', 'roller_noclearance'], index=1)
            st.markdown("---")
            mod_geom = st.checkbox("Modify geometrical constraints?", False)
            show_adv = st.checkbox("Show advanced effects?", False)
            
        with c2:
            st.markdown("**Geometry & Offsets**")
            if mod_geom:
                g1, g2 = st.columns(2)
                with g1:
                    inputs['z1'] = st.number_input("z1 (sun teeth)", value=73)
                    inputs['z2'] = st.number_input("z2 (planet teeth)", value=26)
                    inputs['z3'] = st.number_input("z3 (ring teeth)", value=125)
                    inputs['x1'] = st.number_input("Profile shift x1", value=0.0)
                    inputs['aw_mm'] = st.number_input("Center distance aw [mm]", value=99.0)
                with g2:
                    inputs['m0_mm'] = st.number_input("Module m0 [mm]", value=2.0)
                    inputs['beta_deg'] = st.number_input("Helix angle beta [deg]", value=0.0)
                    inputs['b_face_mm'] = st.number_input("Face width b [mm]", value=25.0)
                    inputs['x2'] = st.number_input("Profile shift x2", value=0.0)
                    ns_pa = st.checkbox("Non-standard pressure angle?", False)
                    inputs['alpha_deg'] = st.number_input("Custom alpha [deg]", value=23.04, disabled=not ns_pa)
                ox1, ox2 = st.columns(2)
                with ox1: sx = st.number_input("Sun offset X [mm]", value=0.0)
                with ox2: sy = st.number_input("Sun offset Y [mm]", value=0.0)
                inputs['sunOffset_xy_mm'] = [sx, sy]
                
                cx1, cx2 = st.columns(2)
                with cx1: cx = st.number_input("Carrier offset X [mm]", value=0.0)
                with cx2: cy = st.number_input("Carrier offset Y [mm]", value=0.0)
                inputs['carrierOffset_xy_mm'] = [cx, cy]
            else:
                inputs.update(dict(z1=73, z2=26, z3=125, m0_mm=2.0, beta_deg=0.0, aw_mm=99.0, b_face_mm=25.0, x1=0.0, x2=0.0, sunOffset_xy_mm=[0,0], carrierOffset_xy_mm=[0,0]))
                
        with c3:
            st.markdown("**Advanced Effects**")
            if show_adv:
                ae1, ae2 = st.columns(2)
                with ae1:
                    inputs['zero_error_override'] = st.checkbox("Zero error override", False)
                    inputs['error_preset'] = st.selectbox("Error level", ['fine', 'medium', 'coarse'], index=1, disabled=inputs['zero_error_override'])
                    
                    sp = st.checkbox("Error on planet 1 only", False)
                    inputs['single_planet_error_override'] = sp
                    inputs['single_planet_error_um'] = st.number_input("Single planet error [um]", value=0.0, disabled=not sp)
                    
                    # 1. Periodic Eccentricity
                    pe = st.checkbox("Enable periodic eccentricity", False)
                    inputs['enable_periodic_ecc'] = pe
                    elvl = st.selectbox("Excitation level", ['Low (5 µm)', 'Medium (10 µm)', 'High (20 µm)'], index=1, disabled=not pe)
                    inputs['ecc_amp_um'] = {'low (5 µm)': 5, 'medium (10 µm)': 10, 'high (20 µm)': 20}[elvl.lower()] if pe else 0
                    
                    # 2. TVMS (Harmonics)
                    ps = st.checkbox("Enable TVMS (Harmonics)", False)
                    inputs['enable_periodic_stiffness'] = ps
                    inputs['tvms_amp_scale'] = 1.0 if ps else 0.0
                    inputs['tvms_order'] = st.selectbox("TVMS Harmonic Order", [1, 2, 3], index=0, disabled=not ps)

                    # 3. Overall Stiffness Multipliers
                    st.markdown("**Stiffness Multipliers**")
                    mod_stiff = st.checkbox("Modify stiffness scales", False)
                    s_c1, s_c2 = st.columns(2)
                    with s_c1:
                        inputs['mesh_scale_factor'] = st.number_input("Mesh scale", value=1.0, disabled=not mod_stiff)
                    with s_c2:
                        inputs['bearing_scale_factor'] = st.number_input("Bearing scale", value=1.0, disabled=not mod_stiff)
                    
                with ae2:
                    inputs['enable_temperature_effects'] = st.checkbox("Enable thermal effects", False)
                    inputs['temperature_C'] = st.number_input("Operating temp [C]", value=20.0, disabled=not inputs['enable_temperature_effects'])
                    
                    # Combined Tilt & Settlement Support
                    ts = st.checkbox("Enable tilt & settlement support", False)
                    
                    tilt_opts = {'1e3': 1e3, '1e4': 1e4, '1e5': 1e5, '1e6': 1e6}
                    tilt_choice = st.selectbox("Tilt support k_phi [Nm/rad]", list(tilt_opts.keys()), index=1, disabled=not ts)
                    inputs['k_support_phi_NmRad'] = tilt_opts[tilt_choice] if ts else 0.0
                    
                    settle_opts = {'1e4': 1e4, '1e6': 1e6, '1e8': 1e8}
                    settle_choice = st.selectbox("Settlement support k_w [N/m]", list(settle_opts.keys()), index=2, disabled=not ts)
                    inputs['k_support_w_Nm'] = settle_opts[settle_choice] if ts else 0.0
                    
                    inputs['phase_steps'] = int(st.number_input("Phase steps", value=181))
                    inputs['seed'] = int(st.number_input("Seed", value=42))
                    inputs['run_sensitivity'] = st.checkbox("Run sensitivity study", False)
                    inputs['run_monte_carlo'] = st.checkbox("Run Monte Carlo", False)

            else:
                inputs.update(dict(
                    error_preset='medium', zero_error_override=False, 
                    single_planet_error_override=False, single_planet_error_um=0.0, 
                    enable_periodic_ecc=False, ecc_amp_um=0, 
                    enable_periodic_stiffness=False, tvms_amp_scale=0.0, tvms_order=1,
                    mesh_scale_factor=1.0, bearing_scale_factor=1.0, 
                    enable_temperature_effects=False, temperature_C=20.0, 
                    k_support_phi_NmRad=0.0, k_support_w_Nm=0.0, 
                    phase_steps=181, seed=42, run_sensitivity=False, run_monte_carlo=False
                ))
                
        # Hidden hardcoded constants needed for solver
        inputs.update(dict(
            temperature_ref_C=20, 
            enable_thermal_geometry_error=inputs.get('enable_temperature_effects', False), 
            use_auto_thermal_gradient=True, thermal_gradient_ratio=0.1, 
            thermal_gradient_C=0, thermal_hotspot_deg=0, 
            thermal_pin_weight=1.0, thermal_planet_weight=0.7, 
            r_support_mm=inputs.get('R_sun_mm', 50.0), 
            ecc_phase_deg=0, ecc_order=1, 
            enable_periodic_stiffness=inputs.get('enable_periodic_stiffness', False), 
            tvms_amp_scale=inputs.get('tvms_amp_scale', 0.0), 
            tvms_order=inputs.get('tvms_order', 1), 
            tvms_planet_phase_scale=1.0, 
            monte_carlo_base_seed=inputs.get('seed', 42), sensitivity_seed=inputs.get('seed', 42), nMonteCarlo=50, 
            sensitivity_ecc_values_um=[0,5,10,20,35,50,75,100], 
            sensitivity_mesh_scale_values=[0.05,0.1,0.25,1.0,5.0,10.0,20.0], 
            sensitivity_bearing_scale_values=[0.01,0.05,0.1,1.0,10.0,50.0,100.0]
        ))

        if inputs['zero_error_override']: 
            inputs['tol_override'] = dict(pin_rad_um=0, pin_tan_um=0, runout_um=0, profile_um=0, commonProfile_um=0, commonXY_um=0)
        if inputs['single_planet_error_override']: 
            inputs['zero_error_override'] = False
            inputs['tol_override'] = dict(pin_rad_um=0, pin_tan_um=0, runout_um=0, profile_um=0, commonProfile_um=0, commonXY_um=0)
            
        st.markdown("---")
        run_btn = st.button("▶ RUN ANALYSIS", type="primary", use_container_width=True)
    return inputs, run_btn

def main():
    st.markdown("""
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #c0c0c0;
        border-radius: 0;
        margin-bottom: 0.5rem;
    }
    .panel-header {
        background-color: #f2f2f2;
        border-bottom: 1px solid #c0c0c0;
        padding: 4px 8px;
        margin: -1rem -1rem 1rem -1rem;
        font-weight: 500;
        color: #333;
        font-size: 14px;
    }
    .results-text {
        font-size: 14px;
        font-family: monospace;
        line-height: 1.5;
        white-space: pre-wrap;
    }
    </style>
    """, unsafe_allow_html=True)

    # 1. Center the GIF using three columns. 
    # By making the outer columns twice as big [2, 1, 2], the GIF stays small and centered!
    spacer_left, img_col, spacer_right = st.columns([2, 1, 2])
    with img_col:
        st.image("GIF_Epicyclic_Gearing.gif", use_container_width=True)
        
    # 2. Center the Title and Subtitle using standard HTML formatting
    st.markdown("<h1 style='text-align: center;'>Epicyclic Gear System App</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Numerical modelling and simulation tool for load-sharing behaviour of planetary gears.</p>", unsafe_allow_html=True)
    
    # Optional: Adds a tiny bit of space before the user inputs start
    st.markdown("<br>", unsafe_allow_html=True) 

  
    # 1. User Inputs at the top
    inputs, run_btn = collect_inputs()
    
    if run_btn:
        with st.spinner("Running solver, please wait..."):
            sys = EpicyclicGearSystem(inputs)
            sys.run()
            st.session_state['res'] = sys.results
            st.session_state['inputs'] = inputs

    res = st.session_state.get('res')
    if not res:
        st.info("Configure inputs and click **RUN ANALYSIS**.")
        return
        
    st.markdown("---")
    
    # 2. TOP BANNER: NUMERICAL RESULTS
    with st.container(border=True):
        st.markdown('<div class="panel-header">Numerical Results</div>', unsafe_allow_html=True)
        c_text, c_tab = st.columns([1, 1.5])
        with c_text:
            active_count = int(np.sum(res['active_final']))
            text_block = f"""
<div class="results-text">
<b>Nominal transmitted force W [N]</b> : {res['W_nominal_N']:.4f}
<b>Maximum K_gamma [-]</b>             : {res['K_gamma_max']:.6f}
<b>K_gamma span over phase [-]</b>     : {res['K_gamma_span']:.6f}
<b>Worst phase angle [deg]</b>         : {np.rad2deg(res['worst_phase_rad']):.3f}
<b>Sun settlement w [um]</b>           : {res['sun_displacement_final'][0]*1e6:.6f}
<b>Sun tilt phi_x [mrad]</b>           : {res['sun_displacement_final'][1]*1e3:.6f}
<b>Sun tilt phi_y [mrad]</b>           : {res['sun_displacement_final'][2]*1e3:.6f}
<b>Maximum planet force [N]</b>        : {np.max(res['F_final_N']):.6f}
<b>Active planets at worst phase</b>   : {active_count} / {res['N']}
</div>
            """
            st.markdown(text_block, unsafe_allow_html=True)
            
        with c_tab:
            idx = res['worst_phase_index']
            df = pd.DataFrame({
                'Planet': range(1, res['N']+1), 
                'Angle_deg': res['planet_angles_deg'],
                'Force_N': res['F_final_N'], 
                'Share_pct': 100*res['load_share_final'],
                'LSF': res['LSF_final'], 
                'Active': res['active_final'].astype(bool),
                'EqError_um': res['e_total_phase_m'][:, idx]*1e6
            })
            st.dataframe(df, use_container_width=True, hide_index=True)
            
    # 3. ROW 1: LSF AND SCHEMATIC
    r1_c1, r1_c2 = st.columns(2)
    with r1_c1:
        with st.container(border=True):
            st.plotly_chart(plot_lsf_bar(res), use_container_width=True)
    with r1_c2:
        with st.container(border=True):
            st.plotly_chart(plot_schematic(res, st.session_state['inputs'].get('R_sun_mm', 50)), use_container_width=True)
            ecc_xy = res['ecc_xy_phase_m'][:, res['worst_phase_index']] * 1000
            st.info(f"**Displacement:** w = {res['sun_displacement_final'][0]*1e6:.2f} um | phi_x = {res['sun_displacement_final'][1]*1e3:.3f} mrad | phi_y = {res['sun_displacement_final'][2]*1e3:.3f} mrad | ecc = [{ecc_xy[0]:.3f}, {ecc_xy[1]:.3f}] mm")
            st.info("**Color Convention:** Red = Sun gear | Gray circles = Planets | Blue arrow = Tilt direction")

    # 4. ROW 2: ERROR BREAKDOWN AND EQ ERROR
    r2_c1, r2_c2 = st.columns(2)
    with r2_c1:
        with st.container(border=True):
            st.plotly_chart(plot_error_components(res), use_container_width=True)
    with r2_c2:
        with st.container(border=True):
            st.plotly_chart(plot_phase_planet(res, 'e_total_phase_m', "Equivalent Error vs Phase", "Error [um]"), use_container_width=True)
            st.info("Total effective compatibility error acting on each planet as the carrier rotates.")

    # 5. ROW 3: PHASE RESPONSE (K_GAMMA + MODES)
    r3_c1, r3_c2 = st.columns([1.3, 0.7])
    with r3_c1:
        with st.container(border=True):
            phase_mode = st.selectbox("Phase plot mode", ["Raw K_gamma", "Normalized K_gamma", "Delta from mean", "Percent variation"], index=0)
            st.plotly_chart(plot_phase_kgamma(res, mode=phase_mode), use_container_width=True)
            
    with r3_c2:
        with st.container(border=True):
            st.markdown("#### $K_\gamma$ Information")
            st.markdown(f"**Worst phase:** {np.rad2deg(res['worst_phase_rad']):.2f}° ({res['worst_phase_rad']:.3f} rad)")
            st.markdown(f"**$K_\gamma$ max:** {res['K_gamma_max']:.6f}  \n**$K_\gamma$ span:** {res['K_gamma_span']:.6g}")
            
            st.markdown("---")
            st.markdown("**Equation used:**")
            if phase_mode == "Raw K_gamma":
                st.latex(r"y = K_{\gamma}")
            elif phase_mode == "Normalized K_gamma":
                st.latex(r"y = \frac{K_{\gamma}}{\mu(K_{\gamma})}")
            elif phase_mode == "Delta from mean":
                st.latex(r"y = K_{\gamma} - \mu(K_{\gamma})")
            elif phase_mode == "Percent variation":
                st.latex(r"y = \left( \frac{K_{\gamma} - \mu(K_{\gamma})}{\mu(K_{\gamma})} \right) \times 100")

    # 6. ROW 4: PLANET LSF AND FORCE VS PHASE
    r4_c1, r4_c2 = st.columns(2)
    with r4_c1:
        with st.container(border=True):
            st.plotly_chart(plot_phase_planet(res, 'LSF_phase', "Individual Planet LSF vs Phase", "LSF [-]"), use_container_width=True)
    with r4_c2:
        with st.container(border=True):
            st.plotly_chart(plot_phase_planet(res, 'force_phase_N', "Individual Planet Force vs Phase", "Force [N]"), use_container_width=True)


    # 7. MONTE CARLO & TRENDS
    mc = res.get('monte_carlo')
    has_mc = mc and 'nRuns' in mc
    
    st.markdown("### Monte Carlo Analysis")
    
    if has_mc:
        r5_c1, r5_c2 = st.columns(2)
        with r5_c1:
            with st.container(border=True):
                st.plotly_chart(plot_mc_hist(mc), use_container_width=True)
        with r5_c2:
            with st.container(border=True):
                st.plotly_chart(plot_mc_box(mc, res['N']), use_container_width=True)
                
        st.success(f"**Monte Carlo Stats:** Runs: {mc['nRuns']} | Mean max K_gamma: {np.mean(mc['K_gamma_max_all']):.6f} | Std max K_gamma: {np.std(mc['K_gamma_max_all']):.6f}")
    else:
        st.warning("Monte Carlo disabled (Enable in Advanced Effects).")

    # 8. TRENDS
    st.markdown("### Structural Trends")
    r6_c1, r6_c2 = st.columns(2)
    k_w = st.session_state['inputs'].get('k_support_w_Nm', 0)
    with r6_c1:
        with st.container(border=True):
            st.plotly_chart(plot_trend(res, k_w, True), use_container_width=True)
    with r6_c2:
        with st.container(border=True):
            st.plotly_chart(plot_trend(res, k_w, False), use_container_width=True)

    # 9. SENSITIVITY
    st.markdown("### Sensitivity Analysis")
    sens = res.get('sensitivity')
    has_sens = sens is not None
    
    if has_sens:
        r7_c1, r7_c2, r7_c3 = st.columns(3)
        with r7_c1:
            with st.container(border=True):
                # Check if BOTH eccentricity and support stiffness are enabled
                if inputs.get('enable_periodic_ecc', False) and inputs.get('k_support_w_Nm', 0) > 0:
                    st.plotly_chart(plot_sens(sens, 'eccentricity', "Eccentricity amplitude [um]", "K_γ vs Eccentricity", "circle"), use_container_width=True)
                else:
                    st.warning("Eccentricity graph hidden: Eccentricity Effects and Settlement Support must be enabled to view this data.")
        with r7_c2:
            with st.container(border=True):
                st.plotly_chart(plot_sens(sens, 'mesh_scale', "Mesh scale factor [-]", "K_γ vs Mesh Scale", "square"), use_container_width=True)
        with r7_c3:
            with st.container(border=True):
                st.plotly_chart(plot_sens(sens, 'bearing_scale', "Bearing scale factor [-]", "K_γ vs Bearing Scale", "diamond"), use_container_width=True)
    else:
        st.warning("Sensitivity study disabled. Enable in Advanced Effects to view.")

if __name__ == '__main__':
    main()
