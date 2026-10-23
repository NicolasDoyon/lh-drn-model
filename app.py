import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="LH -> DRN Model", layout="wide")

st.title("Modèle LH -> DRN (Wilson-Cowan)")
st.write("Ajustez les poids synaptiques et cliquez sur le bouton pour lancer la simulation.")

# Default weight values
default_weights = {
    'ac': 4.0,
    'ad': 1.0,
    'ae': 3.0,
    'bc': 0.8 * 3.63,
    'bd': 3.63,
    'be': 0.8 * 2.49,
    'cc': 0.896,
    'cd': 0.896,
    'ce': 3.0,
    'dc': 1.0,
    'dd': 4.48,
    'de': 2.0,
}

# Initialize session state for weights
if 'weights' not in st.session_state:
    st.session_state.weights = default_weights.copy()

def reset_defaults():
    st.session_state.weights = default_weights.copy()

# Sidebar form so values don't trigger calculations until submitted
with st.sidebar:
    st.header("Synaptic weights (w)")
    
    # Reset button outside or above the form
    st.button("Initial values", on_click=reset_defaults, use_container_width=True)
    
    with st.form("simulation_form"):
        w = {
            'ac': st.number_input('LH_e -> DRN_e (ac)', value=st.session_state.weights['ac']),
            'ad': st.number_input('LH_e -> DRN_i (ad)', value=st.session_state.weights['ad']),
            'ae': st.number_input('LH_e -> 5HT (ae)', value=st.session_state.weights['ae']),
            'bc': st.number_input('LH_i -> DRN_e (bc)', value=st.session_state.weights['bc']),
            'bd': st.number_input('LH_i -> DRN_i (bd)', value=st.session_state.weights['bd']),
            'be': st.number_input('LH_i -> 5HT (be)', value=st.session_state.weights['be']),
            'cc': st.number_input('DRN_e -> DRN_e (cc)', value=st.session_state.weights['cc']),
            'cd': st.number_input('DRN_e -> DRN_i (cd)', value=st.session_state.weights['cd']),
            'ce': st.number_input('DRN_e -> 5HT (ce)', value=st.session_state.weights['ce']),
            'dc': st.number_input('DRN_i -> DRN_e (dc)', value=st.session_state.weights['dc']),
            'dd': st.number_input('DRN_i -> DRN_i (dd)', value=st.session_state.weights['dd']),
            'de': st.number_input('DRN_i -> 5HT (de)', value=st.session_state.weights['de']),
        }
        
        # Submit button matching your MATLAB "Simulate and trace graph"
        submitted = st.form_submit_button("Simulate and trace graph", use_container_width=True)

# Update session state if form was submitted
if submitted:
    st.session_state.weights = w

# Fixed parameters
p_a = 4.0
p_theta = 1.0
p_r = 10.0

def simulate(ww, w_bd_override=None):
    Dur = 100
    dt = 0.05
    nstep = int(Dur / dt)
    
    fe_results = []
    for j_stim in range(101):
        I_val = j_stim / 100.0
        p_Iexc = I_val
        p_Iinh = I_val
        
        fa, fb, fc, fd, fe = 0.0, 0.0, 0.0, 0.0, 0.0
        local_ww = ww.copy()
        if w_bd_override is not None:
            local_ww['bd'] = w_bd_override
            
        for j in range(nstep - 1):
            Inp_a = p_Iexc
            Inp_b = p_Iinh
            Inp_c = fa * local_ww['ac'] - fb * local_ww['bc'] + fc * local_ww['cc'] - fd * local_ww['dc']
            Inp_d = fa * local_ww['ad'] - fb * local_ww['bd'] + fc * local_ww['cd']
            Inp_e = fa * local_ww['ae'] - fb * local_ww['be'] + fc * local_ww['ce'] - fd * local_ww['de']
            
            fa += (dt / p_r) * (-fa + (2 - fa) / (1 + np.exp(-p_a * (Inp_a - p_theta))))
            fb += (dt / p_r) * (-fb + (2 - fb) / (1 + np.exp(-p_a * (Inp_b - p_theta))))
            fc += (dt / p_r) * (-fc + (2 - fc) / (1 + np.exp(-p_a * (Inp_c - p_theta + 1))))
            fd += (dt / p_r) * (-fd + (2 - fd) / (1 + np.exp(-p_a * (Inp_d - p_theta + 1))))
            fe += (dt / p_r) * (-fe + (2 - fe) / (1 + np.exp(-p_a * (Inp_e - p_theta + 1))))
            
        fe_results.append(fe)
    return fe_results

# Run simulations using the active weights in session state
x_vals = np.linspace(0, 1, 101)
ctrl_res = simulate(st.session_state.weights)
nd_res = simulate(st.session_state.weights, w_bd_override=0.0)

# Status info
st.info("CTRL : LH_i->DRN_i ajustable | without disinh : LH_i->DRN_i = 0")

# Plotting
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_vals, ctrl_res, label='CTRL', linewidth=2)
ax.plot(x_vals, nd_res, label='without disinh', linewidth=2, linestyle='--')
ax.set_xlabel('Intensity of stimulation to LH')
ax.set_ylabel('Response of 5HT neurons')
ax.set_title('Response of 5HT neurons')
ax.set_ylim(0, 1.05)
ax.set_xlim(0, 1)
ax.grid(True)
ax.legend(loc='lower right')

st.pyplot(fig)
