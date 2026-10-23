import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="LH -> DRN Model", layout="wide")

st.title("Modele LH -> DRN (Wilson-Cowan)")
st.write("Ajustez les poids synaptiques pour observer la réponse des neurones 5HT.")

# Sidebar for weights
st.sidebar.header("Synaptic weights (w)")
w = {
    'ac': st.sidebar.number_input('LH_e -> DRN_e (ac)', value=4.0),
    'ad': st.sidebar.number_input('LH_e -> DRN_i (ad)', value=1.0),
    'ae': st.sidebar.number_input('LH_e -> 5HT (ae)', value=3.0),
    'bc': st.sidebar.number_input('LH_i -> DRN_e (bc)', value=0.8 * 3.63),
    'bd': st.sidebar.number_input('LH_i -> DRN_i (bd)', value=3.63),
    'be': st.sidebar.number_input('LH_i -> 5HT (be)', value=0.8 * 2.49),
    'cc': st.sidebar.number_input('DRN_e -> DRN_e (cc)', value=0.896),
    'cd': st.sidebar.number_input('DRN_e -> DRN_i (cd)', value=0.896),
    'ce': st.sidebar.number_input('DRN_e -> 5HT (ce)', value=3.0),
    'dc': st.sidebar.number_input('DRN_i -> DRN_e (dc)', value=1.0),
    'dd': st.sidebar.number_input('DRN_i -> DRN_i (dd)', value=4.48),
    'de': st.sidebar.number_input('DRN_i -> 5HT (de)', value=2.0),
}

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

# Run simulations
x_vals = np.linspace(0, 1, 101)
ctrl_res = simulate(w)
nd_res = simulate(w, w_bd_override=0.0)

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
ax.legend(loc='southeast' if 'southeast' in plt.gca().get_legend_handles_labels()[1] else 'lower right')

st.pyplot(fig)
