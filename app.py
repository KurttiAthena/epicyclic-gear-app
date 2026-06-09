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

st.set_page_config(page_title="Epicyclic Gear System Analysis - V4", layout="wide", initial_sidebar_state="collapsed")

# =========================================================================
# HELPER: Plotly figure defaults
# =========================================================================
def _fig_layout(fig: go.Figure, title: str = "", xaxis_title: str = "",
                yaxis_title: str = "", height: int = 300, **kwargs) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="black"), x=0.5, y=0.95),
        xaxis_title=dict(text=xaxis_title, font=dict(size=11, color="black")),
        yaxis_title=dict(text=yaxis_title, font=dict(size=11, color="black")),
        height=height,
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
        font=dict(size=10, color="black"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h", yanchor="top", y=-0.25, x=0.5, xanchor="center",
            bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1
        ),
        **kwargs,
    )
    fig.update_xaxes(gridcolor="#e0e0e0", zeroline=True, zerolinecolor="#888", showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(gridcolor="#e0e0e0", zeroline=True, zerolinecolor="#888", showline=True, linewidth=1, linecolor='black', mirror=True)
    return fig

def _auto_y_range(y_data: np.ndarray, pad_frac: float = 0.15):
    y_clean = y_data[np.isfinite(y_data)]
    if len(y_clean) == 0: return None
    y_min, y_max = float(np.min(y_clean)), float(np.max(y_clean))
    span = y_max - y_min
    if span < 1e-12: pad = max(1e-6, 0.02 * max(abs(y_min), 1))
    else: pad = pad_frac * span
    return [y_min - pad, y_max + pad]

_PHASE_TICKS = dict(
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
    _fig_layout(fig, "Error Component Breakdown (Worst Phase)", "Planet Index", "Equivalent error [μm]", yaxis_range=[-1.15 * max_abs, 1.15 * max_abs])
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
        # Base size calculation for circles simulating planets
        r_planet = 0.4 * R_sun_mm
        if np.max(lsf) - np.min(lsf) > 1e-4:
            if abs(lsf[i] - np.max(lsf)) < 1e-4:
                color = '#333333' # Black (highest load)
                r_planet = 0.55 * R_sun_mm
            elif abs(lsf[i] - np.min(lsf)) < 1e-4:
                color = '#cccccc' # Light gray (lowest load)
                r_planet = 0.25 * R_sun_mm
            else:
                color = '#888888' # Gray
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
    
    # Title margin increased to avoid overlap
    _fig_layout(fig, "System Schematic (qualitative)", "X [mm]", "Y [mm]", height=300, margin=dict(t=50, b=40, l=40, r=40))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

def plot_phase_kgamma(res: dict, mode: str = 'Raw K_gamma') -> go.Figure:
    phase_deg, worst_deg = np.rad2deg(res['phase_rad']), np.rad2deg(res['worst_phase_rad'])
    K_raw = res['K_gamma_phase'].flatten()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=phase_deg, y=K_raw, mode='lines', line=dict(width=2, color="#2e86c1"), showlegend=False))
    fig.add_vline(x=worst_deg, line_dash="dash", line_color="red", annotation_text="Worst phase", annotation_textangle=-90)
    _fig_layout(fig, "Phase Response: K_γ vs Phase", "Phase angle [deg / rad]", "K_γ [-]", yaxis_range=_auto_y_range(K_raw))
    fig.update_xaxes(**_PHASE_TICKS, range=[0, 360])
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
    fig.update_xaxes(**_PHASE_TICKS, range=[0, 360])
    return fig

def plot_mc_hist(mc: dict) -> go.Figure:
    fig = go.Figure(data=[go.Histogram(x=mc['K_gamma_max_all'], nbinsx=16, marker_color="#4285F4", marker_line=dict(color="white", width=1))])
    return _fig_layout(fig, "Monte Carlo: K_γ,max distribution", "K_γ,max [-]", "Count", height=280)

def plot_mc_box(mc: dict, N: int) -> go.Figure:
    fig = go.Figure()
    for i in range(N):
        fig.add_trace(go.Box(y=mc['LSF_worst_all'][i, :], name=f"{i+1}", marker_color=px.colors.qualitative.D3[i % 10], boxmean=True))
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Ideal=1")
    return _fig_layout(fig, "Monte Carlo: worst-phase LSF by planet", "Planet index", "Worst-phase LSF [-]", height=280)

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
    return _fig_layout(fig, tit, "Number of planets", ylab, height=280)

def plot_sens(sens: dict, key: str, xlab: str, tit: str, sym: str) -> go.Figure:
    if not sens or key not in sens: return go.Figure().add_annotation(text="Disabled", showarrow=False)
    s = sens[key]
    x, y = s.get('values_um', s.get('values', [])), [m['K_gamma_max'] for m in s['metrics']]
    fig = go.Figure(data=[go.Scatter(x=x, y=y, mode='lines+markers', marker=dict(size=8, color="#2e86c1", symbol=sym), line=dict(width=2), showlegend=False)])
    return _fig_layout(fig, tit, xlab, "Max K_γ [-]", height=240)


# =========================================================================
# UI LAYOUT & INPUTS
# =========================================================================

def collect_inputs(c_in):
    inputs = {}
    c_in.markdown("###### User Inputs")
    run_btn = c_in.button("RUN ANALYSIS", use_container_width=True)
    c_in.markdown("---")
    
    c_in.markdown("**Basic Inputs**")
    inputs['N'] = c_in.number_input("Number of planets", 3, 12, 5)
    inputs['T_input_Nm'] = c_in.number_input("Input torque [Nm]", value=1000.0)
    inputs['R_sun_mm'] = c_in.number_input("Sun radius [mm]", value=50.0)
    inputs['alpha_n_deg'] = float(c_in.selectbox("Normal pressure angle", ["14", "20", "25"], index=1))
    inputs['alpha_deg'] = inputs['alpha_n_deg']
    inputs['bearing_type'] = c_in.selectbox("Bearing type", ['ball_clearance', 'ball_noclearance', 'standard', 'roller_clearance', 'roller_noclearance'], index=1)
    
    mod_geom = c_in.checkbox("Modify geometrical constraints?", False)
    if mod_geom:
        with c_in.expander("Geometry & Offsets", expanded=True):
            inputs['z1'] = st.number_input("z1 (sun teeth)", value=73)
            inputs['z2'] = st.number_input("z2 (planet teeth)", value=26)
            inputs['z3'] = st.number_input("z3 (ring teeth)", value=125)
            inputs['m0_mm'] = st.number_input("Module m0 [mm]", value=2.0)
            inputs['beta_deg'] = st.number_input("Helix angle beta [deg]", value=0.0)
            if st.checkbox("Non-standard pressure angle?", False):
                inputs['alpha_deg'] = st.number_input("Custom alpha [deg]", value=23.04)
            inputs['aw_mm'] = st.number_input("Center distance aw [mm]", value=99.0)
            inputs['b_face_mm'] = st.number_input("Face width b [mm]", value=25.0)
            inputs['x1'] = st.number_input("Profile shift x1", value=0.0)
            inputs['x2'] = st.number_input("Profile shift x2", value=0.0)
            inputs['sunOffset_xy_mm'] = [st.number_input("Sun offset X [mm]", value=0.0), st.number_input("Sun offset Y [mm]", value=0.0)]
            inputs['carrierOffset_xy_mm'] = [st.number_input("Carrier offset X [mm]", value=0.0), st.number_input("Carrier offset Y [mm]", value=0.0)]
    else:
        inputs.update(dict(z1=73, z2=26, z3=125, m0_mm=2.0, beta_deg=0.0, aw_mm=99.0, b_face_mm=25.0, x1=0.0, x2=0.0, sunOffset_xy_mm=[0,0], carrierOffset_xy_mm=[0,0]))
        
    show_adv = c_in.checkbox("Show advanced effects?", False)
    if show_adv:
        with c_in.expander("Advanced Effects", expanded=True):
            inputs['error_preset'] = st.selectbox("Error level", ['fine', 'medium', 'coarse'], index=1)
            inputs['zero_error_override'] = st.checkbox("Zero error override", False)
            sp = st.checkbox("Error on planet 1 only", False)
            inputs['single_planet_error_override'] = sp
            inputs['single_planet_error_um'] = st.number_input("Single planet error [um]", value=0.0, disabled=not sp)
            inputs['enable_periodic_ecc'] = st.checkbox("Enable periodic eccentricity", False)
            elvl = st.selectbox("Excitation level", ['Low', 'Medium', 'High'], index=1)
            inputs['ecc_amp_um'] = {'low':5, 'medium':10, 'high':20}[elvl.lower()] if inputs['enable_periodic_ecc'] else 0
            inputs['enable_periodic_stiffness'] = st.checkbox("Enable periodic stiffness", False)
            inputs['tvms_amp_scale'] = st.number_input("TVMS scale factor", value=1.0) if inputs['enable_periodic_stiffness'] else 0
            inputs['enable_temperature_effects'] = st.checkbox("Enable thermal effects", False)
            inputs['temperature_C'] = st.number_input("Operating temp [C]", value=20.0, disabled=not inputs['enable_temperature_effects'])
            inputs['k_support_phi_NmRad'] = st.number_input("Tilt support k_phi [Nm/rad]", value=10000.0)
            inputs['k_support_w_Nm'] = st.number_input("Settlement support k_w [N/m]", value=0.0)
            inputs['phase_steps'] = int(st.number_input("Phase steps", value=181))
            inputs['seed'] = int(st.number_input("Seed", value=42))
            inputs['run_sensitivity'] = st.checkbox("Run sensitivity study", False)
            inputs['run_monte_carlo'] = st.checkbox("Run Monte Carlo", False)
    else:
        inputs.update(dict(error_preset='medium', zero_error_override=False, single_planet_error_override=False, single_planet_error_um=0.0, enable_periodic_ecc=False, ecc_amp_um=0, enable_periodic_stiffness=False, tvms_amp_scale=0, enable_temperature_effects=False, temperature_C=20.0, k_support_phi_NmRad=0.0, k_support_w_Nm=0.0, phase_steps=181, seed=42, run_sensitivity=False, run_monte_carlo=False))
        
    inputs.update(dict(temperature_ref_C=20, enable_thermal_geometry_error=inputs['enable_temperature_effects'], use_auto_thermal_gradient=True, thermal_gradient_ratio=0.1, thermal_gradient_C=0, thermal_hotspot_deg=0, thermal_pin_weight=1.0, thermal_planet_weight=0.7, r_support_mm=inputs['R_sun_mm'], mesh_scale_factor=1.0, bearing_scale_factor=1.0, ecc_phase_deg=0, ecc_order=1, tvms_order=1, tvms_planet_phase_scale=1.0, monte_carlo_base_seed=inputs['seed'], sensitivity_seed=inputs['seed'], nMonteCarlo=50, sensitivity_ecc_values_um=[0,5,10,20,35,50,75,100], sensitivity_mesh_scale_values=[0.05,0.1,0.25,1.0,5.0,10.0,20.0], sensitivity_bearing_scale_values=[0.01,0.05,0.1,1.0,10.0,50.0,100.0]))
    
    if inputs['zero_error_override']: inputs['tol_override'] = dict(pin_rad_um=0, pin_tan_um=0, runout_um=0, profile_um=0, commonProfile_um=0, commonXY_um=0)
    if inputs['single_planet_error_override']: inputs['zero_error_override'], inputs['tol_override'] = False, dict(pin_rad_um=0, pin_tan_um=0, runout_um=0, profile_um=0, commonProfile_um=0, commonXY_um=0)

    return inputs, run_btn

def main():
    # CSS to make the app look like the MATLAB UI with exact panel borders and gray headers
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
    div.stTabs [data-baseweb="tab-list"] {
        background-color: #e5e5e5;
        border-radius: 0;
        gap: 1px;
    }
    div.stTabs [data-baseweb="tab"] {
        background-color: #f2f2f2;
        border: 1px solid #c0c0c0;
        border-bottom: none;
        padding: 4px 12px;
        font-size: 13px;
        color: #333;
    }
    div.stTabs [aria-selected="true"] {
        background-color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col_in, col_out = st.columns([1, 3.5])
    
    with col_in:
        with st.container(border=True):
            inputs, run_btn = collect_inputs(st)
    
    if run_btn:
        with st.spinner("Running..."):
            sys = EpicyclicGearSystem(inputs)
            sys.run()
            st.session_state['res'] = sys.results
            st.session_state['inputs'] = inputs

    res = st.session_state.get('res')
    if not res:
        with col_out: st.info("Configure inputs on the left and click **RUN ANALYSIS**.")
        return
        
    with col_out:
        # TOP PANEL: Numerical Results
        with st.container(border=True):
            st.markdown('<div class="panel-header">Numerical Results</div>', unsafe_allow_html=True)
            c_text, c_tab = st.columns([1.2, 1.8])
            with c_text:
                active_count = int(np.sum(res['active_final']))
                t = f"""
Nominal transmitted force W [N] : {res['W_nominal_N']:.4f}  
Maximum K_gamma [-] : {res['K_gamma_max']:.6f}  
K_gamma span over phase [-] : {res['K_gamma_span']:.15f}  
Worst phase angle [deg] : {np.rad2deg(res['worst_phase_rad']):.3f}  
Sun settlement w [um] : {res['sun_displacement_final'][0]*1e6:.6f}  
Sun tilt phi_x [mrad] : {res['sun_displacement_final'][1]*1e3:.6f}  
Sun tilt phi_y [mrad] : {res['sun_displacement_final'][2]*1e3:.6f}  
Maximum planet force [N] : {np.max(res['F_final_N']):.6f}  
Active planets at worst phase : {active_count} / {res['N']}  
Zero error override : {str(res.get('zero_error_override', False)).lower()}  
Planet-1-only error override : {str(res.get('single_planet_error_override', False)).lower()}  
Periodic eccentricity enabled : {str(res.get('periodic_ecc_enabled', False)).lower()}  
Periodic stiffness enabled : {str(res.get('periodic_stiffness_enabled', False)).lower()}  
                """
                st.text(t)
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
                # Formatting table exactly like MATLAB
                st.dataframe(df, use_container_width=True, hide_index=True)
                
        # MIDDLE PANELS: LSF | Schematic | Phase
        c_lsf, c_sch, c_pha = st.columns([1, 1, 1.3])
        
        with c_lsf:
            with st.container(border=True):
                st.markdown('<div class="panel-header">Final LSF</div>', unsafe_allow_html=True)
                t_lsf, t_err = st.tabs(["LSF", "Errors"])
                with t_lsf: st.plotly_chart(plot_lsf_bar(res), use_container_width=True, config={'displayModeBar': False})
                with t_err: st.plotly_chart(plot_error_components(res), use_container_width=True, config={'displayModeBar': False})
                
        with c_sch:
            with st.container(border=True):
                st.markdown('<div class="panel-header">System Schematic</div>', unsafe_allow_html=True)
                st.plotly_chart(plot_schematic(res, st.session_state['inputs'].get('R_sun_mm', 50)), use_container_width=True, config={'displayModeBar': False})
                
                # Text below schematic
                ecc_xy = res['ecc_xy_phase_m'][:, res['worst_phase_index']] * 1000
                st.text(f"w = {res['sun_displacement_final'][0]*1e6:.2f} um\nphi_x = {res['sun_displacement_final'][1]*1e3:.3f} mrad\nphi_y = {res['sun_displacement_final'][2]*1e3:.3f} mrad\necc = [{ecc_xy[0]:.3f}, {ecc_xy[1]:.3f}] mm")
                
        with c_pha:
            with st.container(border=True):
                st.markdown('<div class="panel-header">Phase Response</div>', unsafe_allow_html=True)
                t_kg, t_plsf, t_pf, t_pe = st.tabs(["K_gamma", "Planet LSF", "Planet Force", "Eq. Error"])
                with t_kg: st.plotly_chart(plot_phase_kgamma(res), use_container_width=True, config={'displayModeBar': False})
                with t_plsf: st.plotly_chart(plot_phase_planet(res, 'LSF_phase', "Individual Planet LSF vs Phase", "LSF [-]"), use_container_width=True, config={'displayModeBar': False})
                with t_pf: st.plotly_chart(plot_phase_planet(res, 'force_phase_N', "Individual Planet Force vs Phase", "Force [N]"), use_container_width=True, config={'displayModeBar': False})
                with t_pe: st.plotly_chart(plot_phase_planet(res, 'e_total_phase_m', "Equivalent Error vs Phase", "Error [um]"), use_container_width=True, config={'displayModeBar': False})
                
                phase_mode = st.session_state['inputs'].get('phase_plot_mode', 'Raw K_gamma')
                st.text(f"Worst phase = {np.rad2deg(res['worst_phase_rad']):.2f} deg / {res['worst_phase_rad']:.3f} rad\nK_gamma max = {res['K_gamma_max']:.6f} | span = {res['K_gamma_span']:.5g}\nPlot mode = {phase_mode}")
                
        # BOTTOM PANEL: Sensitivity / Monte Carlo
        with st.container(border=True):
            st.markdown('<div class="panel-header">Sensitivity / Monte Carlo</div>', unsafe_allow_html=True)
            t_mc, t_sens = st.tabs(["Monte Carlo / Trends", "Sensitivity"])
            
            with t_mc:
                mc1, mc2, mc3 = st.columns(3)
                mc = res.get('monte_carlo')
                has_mc = mc and 'nRuns' in mc
                with mc1: 
                    if has_mc: st.plotly_chart(plot_mc_hist(mc), use_container_width=True, config={'displayModeBar': False})
                    else: st.plotly_chart(plot_mc_hist({'K_gamma_max_all': [res['K_gamma_max']]}), use_container_width=True, config={'displayModeBar': False}) # show single run if MC disabled
                with mc2:
                    if has_mc: st.plotly_chart(plot_mc_box(mc, res['N']), use_container_width=True, config={'displayModeBar': False})
                    else: st.plotly_chart(plot_mc_box({'LSF_worst_all': res['LSF_final'].reshape(-1, 1)}, res['N']), use_container_width=True, config={'displayModeBar': False})
                with mc3:
                    ts, td = st.tabs(["Stiffness", "Deflection"])
                    k_w = st.session_state['inputs'].get('k_support_w_Nm', 0)
                    with ts: st.plotly_chart(plot_trend(res, k_w, True), use_container_width=True, config={'displayModeBar': False})
                    with td: st.plotly_chart(plot_trend(res, k_w, False), use_container_width=True, config={'displayModeBar': False})
                
                # Bottom text bar
                if has_mc:
                    st.text(f"Monte Carlo enabled\nRuns: {mc['nRuns']}\nMean max K_gamma: {np.mean(mc['K_gamma_max_all']):.6f}\nStd max K_gamma: {np.std(mc['K_gamma_max_all']):.6f}")
                else:
                    st.text("Monte Carlo disabled")
                    
            with t_sens:
                sens = res.get('sensitivity')
                has_sens = sens is not None
                if has_sens:
                    s1, s2, s3 = st.columns(3)
                    with s1: 
                        st.plotly_chart(plot_sens(sens, 'eccentricity', "Eccentricity amplitude [um]", "Sensitivity: K_γ vs Eccentricity", "circle"), use_container_width=True, config={'displayModeBar': False})
                        st.text("Shows how worst-case load sharing changes as\nperiodic eccentricity increases.\nA steeper rise means the system is more\nsensitive to rotating compatibility errors.")
                    with s2: 
                        st.plotly_chart(plot_sens(sens, 'mesh_scale', "Mesh scale factor [-]", "Sensitivity: K_γ vs Mesh Scale", "square"), use_container_width=True, config={'displayModeBar': False})
                        st.text("Shows how worst-case load sharing changes when\noverall mesh stiffness is scaled.\nStrong variation means the result depends heavily\non tooth-mesh stiffness assumptions.")
                    with s3: 
                        st.plotly_chart(plot_sens(sens, 'bearing_scale', "Bearing scale factor [-]", "Sensitivity: K_γ vs Bearing Scale", "diamond"), use_container_width=True, config={'displayModeBar': False})
                        st.text("Shows how worst-case load sharing changes with\nbearing/support stiffness.\nStrong sensitivity means support flexibility\nstrongly affects how the load redistributes.")
                else:
                    st.text("Sensitivity disabled. Enable in Advanced Effects to view.")

if __name__ == '__main__':
    main()
