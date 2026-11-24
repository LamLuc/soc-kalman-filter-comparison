"""check_cref.py – Print a table showing, per .mat file, whether capacity
estimation via Ah/0.95 is reliable or if the cycle ended on the voltage
cutoff (which would under-estimate capacity).

Usage:
    python check_cref.py path\to\*.mat
    python check_cref.py data\n20C\*.mat data\n10C\*.mat ...

Output columns:
    file | Ah_dis | V_min | V_final | ended_on_cutoff | C_ref_0.95
    ended_on_cutoff: YES if V_final < 2.9 V  (HG2 lower cutoff)
"""
import sys, glob, json
import numpy as np
from pathlib import Path

# Add parent of io_cell to path so we can import it directly
sys.path.insert(0, str(Path(__file__).parent))
from io_cell import load   # noqa: E402

VCUT = 2.9          # HG2 lower cutoff voltage (V)
CAP_FRAC = 0.95     # the fraction used in cref = Ah / 0.95

def check_file(path):
    d = load(path)
    ah = d['ah']          # cumulative Ah (convention: discharge negative)
    # discharged Ah: total Ah drawn out (positive number)
    dis = float(ah[0] - ah[-1]) if ah[0] > ah[-1] else float(ah[-1] - ah[0])
    v = d['v']
    v_min = float(np.min(v))
    v_final = float(v[-1])
    cutoff = v_final < VCUT
    cref = dis / CAP_FRAC
    return dict(file=Path(path).name, Ah_dis=round(dis, 4),
                V_min=round(v_min, 4), V_final=round(v_final, 4),
                ended_on_cutoff='YES' if cutoff else 'no',
                C_ref_0p95=round(cref, 4))

def main():
    paths = []
    for arg in sys.argv[1:]:
        expanded = glob.glob(arg)
        if expanded:
            paths.extend(expanded)
        else:
            paths.append(arg)

    if not paths:
        print("Usage: python check_cref.py <file_or_glob> ...")
        sys.exit(1)

    rows = []
    errors = []
    for p in sorted(paths):
        try:
            rows.append(check_file(p))
        except Exception as e:
            errors.append((p, str(e)))

    # Header
    hdr = f"{'File':<35} {'Ah_dis':>8} {'V_min':>7} {'V_final':>8} {'Cutoff?':>8} {'C_ref':>7}"
    sep = '-' * len(hdr)
    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        flag = ' <-- CHECK' if r['ended_on_cutoff'] == 'YES' else ''
        print(f"{r['file']:<35} {r['Ah_dis']:>8.4f} {r['V_min']:>7.4f} {r['V_final']:>8.4f} "
              f"{r['ended_on_cutoff']:>8} {r['C_ref_0p95']:>7.4f}{flag}")
    print(sep)

    if errors:
        print("\nFailed to load:")
        for p, e in errors:
            print(f"  {p}: {e}")

    # summary
    cutoff_files = [r['file'] for r in rows if r['ended_on_cutoff'] == 'YES']
    if cutoff_files:
        print(f"\nWARNING: {len(cutoff_files)} file(s) ended on voltage cutoff.")
        print("C_ref from Ah/0.95 may UNDER-estimate capacity for these files.")
        print("Consider using --cap <measured_capacity_Ah> for those temperatures.")
    else:
        print(f"\nAll {len(rows)} files look OK (did not end on voltage cutoff).")

if __name__ == '__main__':
    main()
