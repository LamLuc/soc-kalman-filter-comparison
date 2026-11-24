"""Create FAKE files in the same layout as the Kollmeyer .mat files (struct 'meas', discharge current negative),
from a simulated cell, ONLY to test the pipeline end to end. Not data. Writes into ./fake_data/."""
import os, numpy as np
from scipy.io import savemat
from model import truth_params, simulate_truth, pack_power, ocv
from cycles import urban, highway, nedc

os.makedirs('fake_data', exist_ok=True)
TP = dict(cq=0.58, cr0=1.7, crc=1.2, ctau=0.8, hyst=0.012, docv=0.0, shape=0.004, r0_lowsoc=0.5)  # ~2.9 Ah cell
rng = np.random.default_rng(0)

def run_current(I_dem, soc0=1.0):
    """Simulate a current-driven profile by giving a power that reproduces it approximately (P = I*3.7)."""
    tp = truth_params(1, soc0=soc0, **TP)
    I, V, S, valid, first = simulate_truth((I_dem * 3.7)[None, :], tp, soc_stop=-1, soc_park=-1)
    return I[0], V[0], S[0]

def save(name, I, V):
    n = len(I); t = np.arange(n, dtype=float)
    Iq = -(I + 0.002 * rng.standard_normal(n))            # cycler convention: discharge negative
    Vq = V + 0.0005 * rng.standard_normal(n)
    ah = np.cumsum(np.r_[0, Iq[:-1]]) / 3600
    meas = dict(Time=t, Voltage=Vq, Current=Iq, Ah=ah, Battery_Temp_degC=np.full(n, 25.0))
    savemat(f'fake_data/{name}.mat', {'meas': meas}); print(name, n, 's')

# C/20 discharge then charge
c = 2.9 / 20
I = np.r_[np.zeros(60), np.full(int(19.9 * 3600), c), np.zeros(1800), np.full(int(19.9 * 3600), -c), np.zeros(60)]
Itr, V, S = run_current(I); keep = (S > 0.0) & (S < 1.0) | (np.abs(Itr) < 1e-6)
save('C20_25degC', Itr, V)
# HPPC: at each SOC step: 1C discharge 10 s, rest 40 s, 0.75C charge 10 s, rest 40 s, C/2 discharge to next SOC, rest 30 min
prof = [np.zeros(600)]
for step in [0.05] + [0.1] * 8 + [0.05]:
    prof += [np.full(10, 2.9), np.zeros(40), np.full(10, -2.2), np.zeros(40), np.full(int(step * 2.9 / 1.45 * 3600), 1.45), np.zeros(1800)]
I = np.concatenate(prof); Itr, V, S = run_current(I); save('HPPC_25degC', Itr, V)
# drive cycles: power-driven, stop when 95% of capacity is removed
def cycle(name, v, scale=1 / 1200.0):
    P = np.tile(pack_power(v), 30) * scale
    tp = truth_params(1, soc0=1.0, **TP)
    I, V, S, valid, first = simulate_truth(P[None, :], tp, soc_stop=0.05, soc_park=0.049)
    n = int(first[0]); save(name, I[0, :n], V[0, :n])
for k in range(1, 5):
    cycle(f'Mixed{k}_25degC', np.concatenate([urban(100 + k), highway(200 + k)]))
cycle('UDDSlike_25degC', urban(300)); cycle('HWFETlike_25degC', highway(301)); cycle('NEDC_25degC', nedc())
