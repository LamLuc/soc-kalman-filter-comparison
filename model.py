# Author: Luc Lam Thanh
# Affiliation: Industrial University of Ho Chi Minh City (IUH)
#              Bosch Global Software Technologies Vietnam
# ORCID: 0009-0005-2382-3696

"""Vehicle power model, nominal 2RC cell model, and a vectorised 'truth' cell simulator whose
structure/parameters may deviate from the nominal model used inside the estimators."""
import numpy as np
from scipy.interpolate import PchipInterpolator

# Representative NMC/graphite OCV curve (illustrative, not identified from a specific cell)
SOC_PTS = np.array([0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1.0])
OCV_PTS = np.array([3.00, 3.30, 3.45, 3.56, 3.62, 3.66, 3.71, 3.78, 3.87, 3.96, 4.06, 4.12, 4.18])
_ocv = PchipInterpolator(SOC_PTS, OCV_PTS)
_docv = _ocv.derivative()
_inv_s = np.linspace(0, 1, 2001); _inv_v = _ocv(_inv_s)

def ocv(s):   # same clipping domain for value and derivative
    return _ocv(np.clip(s, 0.0, 1.0))

def docv(s):
    s = np.asarray(s)
    return np.where((s > 0) & (s < 1), _docv(np.clip(s, 0, 1)), 0.0)

def ocv_inv(v):
    return np.interp(v, _inv_v, _inv_s)

NOM = dict(Q=5.0, R0=0.015, R1=0.010, tau1=10.0, R2=0.015, tau2=200.0)
VEH = dict(m=1850.0, Cd=0.29, A=2.4, Crr=0.010, rho=1.2, g=9.81, eta_dt=0.90, eta_regen=0.65,
           regen_max_kw=60.0, p_aux=400.0, rot_factor=1.04, Ns=96, Np=14)
NCELL = VEH['Ns'] * VEH['Np']

def pack_power(v_kmh, veh=VEH):
    v = v_kmh / 3.6
    a = np.gradient(v)
    F = (veh['m'] * veh['rot_factor'] * a + 0.5 * veh['rho'] * veh['Cd'] * veh['A'] * v**2
         + veh['Crr'] * veh['m'] * veh['g'] * (v > 0.01))
    Pw = F * v
    P = np.where(Pw >= 0, Pw / veh['eta_dt'], np.maximum(Pw * veh['eta_regen'], -veh['regen_max_kw'] * 1e3))
    return P + veh['p_aux']

def truth_params(N, **kw):
    """Matched-model defaults; override any field with arrays of length N."""
    p = dict(cq=np.ones(N), cr0=np.ones(N), crc=np.ones(N), ctau=np.ones(N), docv=np.zeros(N),
             shape=np.zeros(N), hyst=np.zeros(N), r0_lowsoc=np.zeros(N), soc0=np.full(N, 0.9))
    for k, v in kw.items():
        p[k] = np.broadcast_to(np.asarray(v, float), (N,)).copy()
    return p

def simulate_truth(Pcell, tp, gamma=150.0, soc_stop=0.20, soc_park=0.15):
    """Pcell: (N,n) cell power demand [W]. Returns I, V, SOC (N,n) and valid mask (SOC >= soc_stop, before
    first crossing). Once SOC < soc_park the vehicle is parked (P=0) so the cell never leaves its valid range."""
    N, n = Pcell.shape
    Q = NOM['Q'] * tp['cq']
    R1 = NOM['R1'] * tp['crc']; R2 = NOM['R2'] * tp['crc']
    a1 = np.exp(-1.0 / (NOM['tau1'] * tp['ctau'])); a2 = np.exp(-1.0 / (NOM['tau2'] * tp['ctau']))
    soc = tp['soc0'].copy(); v1 = np.zeros(N); v2 = np.zeros(N); h = np.zeros(N)
    I = np.zeros((N, n)); V = np.zeros((N, n)); S = np.zeros((N, n))
    for k in range(n):
        R0 = NOM['R0'] * tp['cr0'] * (1 + tp['r0_lowsoc'] * np.clip((0.3 - soc) / 0.3, 0, 1))
        E = (ocv(soc) + tp['docv'] + tp['shape'] * np.sin(3 * np.pi * soc) + tp['hyst'] * h - v1 - v2)
        P = np.where(soc < soc_park, 0.0, Pcell[:, k])
        i = (E - np.sqrt(np.maximum(E * E - 4 * R0 * P, 0))) / (2 * R0)
        I[:, k] = i; V[:, k] = E - i * R0; S[:, k] = soc
        soc = soc - i / (3600 * Q)
        v1 = a1 * v1 + R1 * (1 - a1) * i; v2 = a2 * v2 + R2 * (1 - a2) * i
        f = np.exp(-np.abs(i) * gamma / (3600 * Q))
        h = f * h - (1 - f) * np.sign(i)
    below = S < soc_stop
    first = np.where(below.any(axis=1), below.argmax(axis=1), n)
    valid = np.arange(n)[None, :] < first[:, None]
    return I, V, S, valid, first
