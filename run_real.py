# Author: Luc Lam Thanh
# Affiliation: Industrial University of Ho Chi Minh City (IUH)
#              Bosch Global Software Technologies Vietnam
# ORCID: 0009-0005-2382-3696

"""Run CC, CC-OCV, EKF, UKF, AEKF, JEKF on measured drive-cycle files with an identified model.

Usage (example for LG HG2 at 25 degC):
  python3 run_real.py --ocv ocv.csv --params params.json \
      --tune 25degC/*Mixed1*.mat 25degC/*Mixed2*.mat ... \
      --test 25degC/*UDDS*.mat 25degC/*LA92*.mat 25degC/*US06*.mat 25degC/*Mixed8*.mat \
      --cap-from-run 0.95 --init-errors -0.2 0 0.2 --biases -0.03 0 0.03 --out results_real.json

Reference SOC = soc0 + (Ah - Ah0)/C_ref from the lab cycler (0.1% FS accuracy). C_ref per file is either --cap
(Ah, e.g. from the 1C capacity test at that temperature) or discharged Ah / FRACTION with --cap-from-run FRACTION
(the HG2 drive cycles stop after 95% of the 1C capacity has been removed).
Scenarios: each file x each initial SOC error x each artificial current offset (added to the measured current,
to reproduce sensor bias; the lab current itself is near-exact).
"""
import argparse, glob, itertools, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model, filters
from io_cell import load

def set_model(ocv_csv, params):
    tab = np.loadtxt(ocv_csv, delimiter=',', skiprows=1)
    s, v = tab[:, 0], tab[:, 1]
    dv = np.gradient(v, s)
    filters.ocv = lambda x: np.interp(np.clip(x, 0, 1), s, v)
    filters.docv = lambda x: np.where((np.asarray(x) > 0) & (np.asarray(x) < 1), np.interp(np.clip(x, 0, 1), s, dv), 0.0)
    filters.ocv_inv = lambda y: np.interp(y, v, s)
    filters.NOM.update({k: params[k] for k in ['Q', 'R0', 'R1', 'tau1', 'R2', 'tau2']})  # in place: defaults see it

def scenarios(files, a):
    """Stack all runs (file x init error x bias) into equal-length arrays (padded, with a validity mask)."""
    runs = []
    for f in files:
        d = load(f, discharge_negative=not a.discharge_positive)
        dis = -(d['ah'][-1] - d['ah'][0])
        cref = a.cap if a.cap else dis / a.cap_from_run
        S = a.soc0 + (d['ah'] - d['ah'][0]) / cref
        for e0, b in itertools.product(a.init_errors, a.biases):
            runs.append(dict(file=os.path.basename(f), err0=e0, bias=b, I=d['i'] + b, Y=d['v'], S=S,
                             s0=float(np.clip(S[0] + e0, 0.02, 1.0)), cref=cref))
    n = max(len(r['S']) for r in runs)
    pad = lambda x: np.pad(x, (0, n - len(x)), mode='edge')
    D = dict(I=np.stack([pad(r['I']) for r in runs]), Y=np.stack([pad(r['Y']) for r in runs]),
             S=np.stack([pad(r['S']) for r in runs]), s0=np.array([r['s0'] for r in runs]))
    D['valid'] = np.stack([np.arange(n) < len(r['S']) for r in runs])
    D['meta'] = [{k: r[k] for k in ('file', 'err0', 'bias', 'cref')} for r in runs]
    return D

GRIDS = {
    'EKF':  (filters.ekf, [dict(q=q, r_mv=r) for q in [1e-10, 1e-9, 1e-8, 1e-7] for r in [5, 10, 20, 40, 80]]),
    'UKF':  (filters.ukf, [dict(q=q, r_mv=r, alpha=al) for q in [1e-10, 1e-9, 1e-8, 1e-7] for r in [5, 10, 20, 40, 80] for al in [0.1, 1.0]]),
    'AEKF': (filters.ekf, [dict(q=1e-9, r_mv=r, adapt=(0.995, qmin, qmax)) for qmin in [1e-11, 1e-10] for qmax in [1e-9, 1e-8, 1e-7] for r in [10, 20, 40]]),
    'JEKF': (filters.jekf, [dict(q=q, r_mv=r, q_r0=a, q_c=c) for q in [1e-9, 1e-8, 1e-7] for a in [1e-11, 1e-9] for c in [1e-11, 1e-9] for r in [10, 20, 40]]),
}

def run(name, cfg, D):
    if name == 'CC':
        return filters.cc(D['I'], D['s0'])
    if name == 'CC-OCV':
        return filters.cc_ocv(D['I'], D['Y'])
    return GRIDS[name][0](D['I'], D['Y'], D['s0'], **cfg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ocv', required=True); ap.add_argument('--params', required=True)
    ap.add_argument('--tune', nargs='+', required=True); ap.add_argument('--test', nargs='+', required=True)
    ap.add_argument('--cap', type=float); ap.add_argument('--cap-from-run', type=float, default=0.95)
    ap.add_argument('--soc0', type=float, default=1.0)
    ap.add_argument('--init-errors', type=float, nargs='+', default=[-0.2, 0.0, 0.2])
    ap.add_argument('--biases', type=float, nargs='+', default=[-0.03, 0.0, 0.03], help='A, added to measured current')
    ap.add_argument('--discharge-positive', action='store_true'); ap.add_argument('--out', default='results_real.json')
    a = ap.parse_args()
    expand = lambda L: sorted({p for x in L for p in (glob.glob(x) or [x])})
    tune, test = expand(a.tune), expand(a.test)
    overlap = set(map(os.path.basename, tune)) & set(map(os.path.basename, test))
    if overlap:
        raise SystemExit(f'files in both tuning and test sets: {overlap}')
    params = json.load(open(a.params)); set_model(a.ocv, params)
    t0 = time.time()
    Dt = scenarios(tune, a); Ds = scenarios(test, a)
    print(f'tuning runs {len(Dt["s0"])}, test runs {len(Ds["s0"])}, loaded in {time.time()-t0:.0f}s', flush=True)
    best, tuning = {}, {}
    for name, (_, grid) in GRIDS.items():
        res = []
        for cfg in grid:
            r, _ = filters.metrics(run(name, cfg, Dt), Dt['S'], Dt['valid'])
            res.append(float(np.percentile(r, 90)))
        k = int(np.argmin(res)); best[name] = grid[k]
        tuning[name] = [dict(cfg={kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in c.items()}, p90=p) for c, p in zip(grid, res)]
        print(f'{name}: best {grid[k]} (P90 RMSE on tuning set {res[k]:.2f})', flush=True)
    jsonable = lambda c: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in c.items()}
    out = dict(tuning=tuning, best={k: jsonable(v) for k, v in best.items()}, params=params,
               tune_files=[os.path.basename(f) for f in tune], test_meta=Ds['meta'], test={})
    # open-loop model check: identified 2RC driven by measured current and REFERENCE SOC -> voltage error
    p = filters.NOM; a1, a2 = np.exp(-1 / p['tau1']), np.exp(-1 / p['tau2'])
    vr = []
    for j, m in enumerate(Ds['meta']):
        if m['err0'] != 0 or m['bias'] != 0:
            continue
        n = Ds['valid'][j].sum(); i = Ds['I'][j, :n]; v1 = v2 = 0.0; vm = np.empty(n)
        for k in range(n):
            vm[k] = filters.ocv(Ds['S'][j, k]) - v1 - v2 - p['R0'] * i[k]
            v1 = a1 * v1 + p['R1'] * (1 - a1) * i[k]; v2 = a2 * v2 + p['R2'] * (1 - a2) * i[k]
        e = (vm - Ds['Y'][j, :n]) * 1e3
        vr.append(dict(file=m['file'], v_rmse_mV=float(np.sqrt(np.mean(e ** 2))), v_max_mV=float(np.abs(e).max())))
        print(f"model voltage error on {m['file']}: RMSE {vr[-1]['v_rmse_mV']:.1f} mV, max {vr[-1]['v_max_mV']:.1f} mV")
    out['model_voltage_error'] = vr
    print(f'\n{"method":8s} {"median":>7s} {"P90":>7s} {"max":>7s}   RMSE in SOC percentage points (test set)')
    for name in ['CC', 'CC-OCV'] + list(GRIDS):
        r, mx = filters.metrics(run(name, best.get(name, {}), Ds), Ds['S'], Ds['valid'])
        out['test'][name] = dict(rmse=r.round(3).tolist(), max=mx.round(3).tolist())
        print(f'{name:8s} {np.median(r):7.2f} {np.percentile(r, 90):7.2f} {r.max():7.2f}')
    json.dump(out, open(a.out, 'w'), indent=1); print('wrote', a.out)

if __name__ == '__main__':
    main()
