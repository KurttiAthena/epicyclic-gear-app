import streamlit as st
import numpy as np

from epicyclic_gear_system import EpicyclicGearSystem

st.title("Epicyclic Gear Load-Sharing App")

st.write("Simple test version")

N = st.number_input("Number of planets", min_value=3, max_value=20, value=5, step=1)
T_input_Nm = st.number_input("Input torque [Nm]", value=1000.0)
R_sun_mm = st.number_input("Sun pitch radius [mm]", value=50.0)

if st.button("Run solver"):
    inputs = {
        "N": int(N),
        "T_input_Nm": float(T_input_Nm),
        "R_sun_mm": float(R_sun_mm),
    }

    system = EpicyclicGearSystem(inputs)
    system.run()
    results = system.results

    st.subheader("Results")
    st.write("Maximum K_gamma:", results["K_gamma_max"])
    st.write("Final load sharing factors:", results["LSF_final"])

    st.subheader("Planet forces at worst phase")
    st.write(results["F_final_N"])

    st.subheader("Sun displacement at worst phase")
    st.write(results["sun_displacement_final"])
