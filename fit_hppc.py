# Author: Luc Lam Thanh
# Affiliation: Industrial University of Ho Chi Minh City (IUH)
#              Bosch Global Software Technologies Vietnam
# ORCID: 0009-0005-2382-3696

"""Identify 2RC parameters (R0, R1, tau1, R2, tau2) from an HPPC test.

Usage:
  python3 fit_hppc.py <hppc_file> --ocv ocv.csv --cap 2.9 [--soc0 1.0] [--out params.json]
For every discharge pulse (current > --thr for --min-s..--max-s seconds, preceded by a rest) the 2RC model is
simulated over [pulse start - 5 s, next current change] with the MEASURED current and fitted by least squares.
Writes per-pulse results (params_per_pulse.csv) and the median over pulses with 20% <= SOC <= 90%
(params.json), which is what run_real.py uses.
"""
import argparse, json
import numpy as np
from scipy.optimize import least_squares
from io_cell import load

def sim2rc(i, p, v_ocv):
    R0, R1, t1, R2, t2 = p
    a1, a2 = np.exp(-1 / t1), np.exp(-1 / t2)
    v1 = v2 = 0.0; out = np.empty_like(i)
    for k, ik in enumerate(i):
        out[k] = v_ocv[k] - v1 - v2 - R0 * ik
        v1 = a1 * v1 + R1 * (1 - a1) * ik; v2 = a2 * v2 + R2 * (1 - a2) * ik
    return out

def pulses(i, thr, min_s, max_s, rest_s=10):
    on = i > thr
    out, k, n = [], 0, len(i)
    while k < n:
        if on[k]:
            j = k
            while j < n and on[j]:
                j += 1
            if min_s <= j - k <= max_s and k >= rest_s and np.all(np.abs(i[k - rest_s:k]) < thr / 4):
                e = j
                while e < n and abs(i[e]) < thr / 4 and e - j < 600:
                    e += 1
                if e - j >= 10:
                    out.append((k, j, e))
            k = j
        else:
            k += 1
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('file'); ap.add_argument('--ocv', required=True); ap.add_argument('--cap', type=float, required=True)
    ap.add_argument('--soc0', type=float, default=1.0, help='SOC at the start of the file (1.0 after a full charge)')
    ap.add_argument('--thr', type=float, default=0.5, help='A; minimum discharge current that counts as a pulse')
    ap.add_argument('--min-s', type=int, default=5); ap.add_argument('--max-s', type=int, default=60)
    ap.add_argument('--discharge-positive', action='store_true'); ap.add_argument('--out', default='params.json')
    a = ap.parse_args()
    d = load(a.file, discharge_negative=not a.discharge_positive)
    tab = np.loadtxt(a.ocv, delimiter=',', skiprows=1)
    ocv = lambda s: np.interp(s, tab[:, 0], tab[:, 1])
    soc = a.soc0 + (d['ah'] - d['ah'][0]) / a.cap
    rows = []
    for (s, e_p, e) in pulses(d['i'], a.thr, a.min_s, a.max_s):
        prev = np.flatnonzero(d['logged'][:s])
        if prev.size == 0 or s - prev[-1] > 120:
            continue                                   # no real rest sample shortly before the pulse
        s0 = min(prev[-1], s - 1)                      # last LOGGED rest sample before the pulse
        w = slice(s0, e)
        i, v, lg = d['i'][w], d['v'][w], d['logged'][w]
        if lg.sum() < 8:
            continue                                   # too few real samples around this pulse
        base = v[0] - ocv(soc[s0])                     # local OCV offset (hysteresis, OCV error), from a real rest sample
        vo = ocv(soc[w]) + base
        f = lambda p: (sim2rc(i, p, vo) - v)[lg]      # residual on LOGGED samples only (no interpolated voltage)
        r = least_squares(f, x0=[0.02, 0.01, 5.0, 0.01, 60.0],
                          bounds=([1e-4, 1e-5, 0.3, 1e-5, 10.0], [0.2, 0.2, 10.0, 0.2, 2000.0]))
        rmse = float(np.sqrt(np.mean(r.fun ** 2)) * 1e3)
        at_bound = bool(np.any(r.active_mask != 0))   # a parameter stuck at its bound -> fit not trusted
        rows.append(dict(soc=float(soc[s]), I=float(d['i'][(s + e_p) // 2]), R0=r.x[0], R1=r.x[1], tau1=r.x[2],
                         R2=r.x[3], tau2=r.x[4], rmse_mV=rmse, at_bound=at_bound))
    if not rows:
        raise SystemExit('No pulses found: check --thr/--min-s/--max-s and the current sign (inspect_file.py).')
    keys = ['soc', 'I', 'R0', 'R1', 'tau1', 'R2', 'tau2', 'rmse_mV', 'at_bound']
    with open(a.out.replace('.json', '_per_pulse.csv'), 'w') as fh:
        fh.write(','.join(keys) + '\n')
        for r in rows:
            fh.write(','.join(f'{r[k]:.6g}' if not isinstance(r[k], bool) else str(int(r[k])) for k in keys) + '\n')
    sel = [r for r in rows if 0.2 <= r['soc'] <= 0.9 and not r['at_bound']] or rows
    med = {k: float(np.median([r[k] for r in sel])) for k in ['R0', 'R1', 'tau1', 'R2', 'tau2']}
    med['Q'] = a.cap; med['n_pulses_used'] = len(sel); med['fit_rmse_mV_median'] = float(np.median([r['rmse_mV'] for r in sel]))
    json.dump(med, open(a.out, 'w'), indent=1)
    print(f'{len(rows)} pulses fitted, {len(sel)} used'); print(json.dumps(med, indent=1))

if __name__ == '__main__':
    main()
