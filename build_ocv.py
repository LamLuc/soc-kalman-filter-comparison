# Author: Luc Lam Thanh
# Affiliation: Industrial University of Ho Chi Minh City (IUH)
#              Bosch Global Software Technologies Vietnam
# ORCID: 0009-0005-2382-3696

"""Build OCV(SOC) from a low-current (C/20 or similar) discharge [+ charge] test.

Usage:
  python3 build_ocv.py <c20_file> [--charge <c20_charge_file>] [--r0 0.02] [--out ocv.csv]
If the discharge and charge are in one file, pass only that file. With both directions the OCV is the
average of the two curves (cancels most of the IR drop and hysteresis). With discharge only, the curve
is corrected by + I*R0 (give --r0, e.g. from fit_hppc.py) and the result is biased low by the hysteresis.
"""
import argparse, json
import numpy as np
from io_cell import load

def branch(d, sign, thr):
    """Return (soc, v) for the longest run with current in direction `sign` (+1 discharge, -1 charge)."""
    on = sign * d['i'] > thr
    # longest contiguous run
    best, start = (0, 0), None
    for k, f in enumerate(np.append(on, False)):
        if f and start is None:
            start = k
        if not f and start is not None:
            if k - start > best[1] - best[0]:
                best = (start, k)
            start = None
    a, b = best
    i, v, t = d['i'][a:b], d['v'][a:b], d['t'][a:b]
    q = np.concatenate([[0], np.cumsum(sign * i[:-1] * np.diff(t))]) / 3600   # Ah moved in this run
    Q = q[-1]
    soc = 1 - q / Q if sign > 0 else q / Q
    return soc, v, np.abs(i).mean(), Q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('file'); ap.add_argument('--charge'); ap.add_argument('--r0', type=float, default=None)
    ap.add_argument('--thr', type=float, default=0.02, help='A; current magnitude that counts as flowing')
    ap.add_argument('--discharge-positive', action='store_true', help='file logs discharge current as positive')
    ap.add_argument('--out', default='ocv.csv')
    a = ap.parse_args()
    d = load(a.file, discharge_negative=not a.discharge_positive)
    grid = np.linspace(0, 1, 201)
    sd, vd, Id, Qd = branch(d, +1, a.thr)
    vd_g = np.interp(grid, sd[::-1], vd[::-1])
    info = dict(Q_discharge_Ah=float(Qd), I_discharge_A=float(Id))
    dc = load(a.charge, discharge_negative=not a.discharge_positive) if a.charge else d
    try:
        sc, vc, Ic, Qc = branch(dc, -1, a.thr)
        ok = Qc > 0.5 * Qd
    except Exception:
        ok = False
    if ok:
        vc_g = np.interp(grid, sc, vc)
        ocv = 0.5 * (vd_g + vc_g)
        info.update(Q_charge_Ah=float(Qc), I_charge_A=float(Ic), method='average of charge and discharge',
                    hysteresis_mV_at_50pct=float(1e3 * (vc_g - vd_g)[100] / 2))
    else:
        if a.r0 is None:
            raise SystemExit('No charge branch found: pass --charge <file> or --r0 <ohm> for IR correction.')
        ocv = vd_g + Id * a.r0
        info.update(method=f'discharge + I*R0 (R0={a.r0} ohm); hysteresis not removed')
    ocv = np.maximum.accumulate(ocv)          # enforce monotone OCV (noise at the ends)
    np.savetxt(a.out, np.c_[grid, ocv], delimiter=',', header='soc,ocv_V', comments='', fmt='%.5f')
    json.dump(info, open(a.out.replace('.csv', '_info.json'), 'w'), indent=1)
    print(json.dumps(info, indent=1)); print('wrote', a.out)

if __name__ == '__main__':
    main()
