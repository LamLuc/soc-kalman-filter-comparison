# Author: Luc Lam Thanh
# Affiliation: Industrial University of Ho Chi Minh City (IUH)
#              Bosch Global Software Technologies Vietnam
# ORCID: 0009-0005-2382-3696

"""Plot reference SOC vs estimates for one drive-cycle file (for the paper).
Usage: python3 plot_run.py <file> --ocv ocv.csv --params params.json --results results_real.json [--cap 2.9] [--err0 0.2] [--bias 0.03]"""
import argparse, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import filters
from io_cell import load
from run_real import set_model

ap = argparse.ArgumentParser()
ap.add_argument('file'); ap.add_argument('--ocv', required=True); ap.add_argument('--params', required=True)
ap.add_argument('--results', required=True); ap.add_argument('--cap', type=float); ap.add_argument('--cap-from-run', type=float, default=0.95)
ap.add_argument('--err0', type=float, default=0.2); ap.add_argument('--bias', type=float, default=0.0)
ap.add_argument('--discharge-positive', action='store_true'); ap.add_argument('--out', default='soc_trace.png')
a = ap.parse_args()
set_model(a.ocv, json.load(open(a.params))); best = json.load(open(a.results))['best']
d = load(a.file, discharge_negative=not a.discharge_positive)
cref = a.cap or -(d['ah'][-1] - d['ah'][0]) / a.cap_from_run
S = 1.0 + (d['ah'] - d['ah'][0]) / cref
I, Y = (d['i'] + a.bias)[None], d['v'][None]; s0 = np.array([np.clip(S[0] + a.err0, 0.02, 1)])
adapt = lambda c: {k: (tuple(v) if k == 'adapt' else v) for k, v in c.items()}
est = {'CC': filters.cc(I, s0), 'EKF': filters.ekf(I, Y, s0, **adapt(best['EKF'])), 'UKF': filters.ukf(I, Y, s0, **best['UKF']),
       'JEKF': filters.jekf(I, Y, s0, **best['JEKF'])}
t = d['t'] / 3600
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax[0].plot(t, S * 100, 'k', lw=2, label='Tham chiếu (Ah của máy thử)')
for k, e in est.items():
    ax[0].plot(t, e[0] * 100, lw=1, label=k); ax[1].plot(t, (e[0] - S) * 100, lw=1, label=k)
ax[0].set_ylabel('SOC (%)'); ax[1].set_ylabel('Sai số (điểm %)'); ax[1].set_xlabel('Thời gian (h)')
ax[0].legend(); ax[1].axhline(0, color='grey', lw=0.5); fig.tight_layout(); fig.savefig(a.out, dpi=150); print('wrote', a.out)
