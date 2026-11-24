"""Usage: python3 inspect_file.py <file.mat|file.csv>
Prints the fields in the file, which ones io_cell.load() picks, and a short summary to check sign and units."""
import sys
import numpy as np
from io_cell import load, _from_mat, _from_csv

p = sys.argv[1]
raw = _from_mat(p) if p.lower().endswith('.mat') else _from_csv(p)
print('Fields in file:')
for k, v in raw.items():
    a = np.asarray(v)
    print(f'  {k:28s} n={a.size:<8d} dtype={a.dtype}  first={a.flat[0] if a.size else None}')
d = load(p)
print('\nPicked:', d['fields_used'])
print(f"duration {d['t'][-1]/3600:.2f} h at 1 Hz, V {d['v'].min():.3f}..{d['v'].max():.3f} V, "
      f"I {d['i'].min():.2f}..{d['i'].max():.2f} A (+ = discharge after sign flip), Ah {d['ah'][0]:.3f} -> {d['ah'][-1]:.3f}")
print('Check: during a discharge the converted current must be POSITIVE and Ah must DECREASE.\n'
      'If not, call load(path, discharge_negative=False).')
