# Author: Luc Lam Thanh
# Affiliation: Industrial University of Ho Chi Minh City (IUH)
#              Bosch Global Software Technologies Vietnam
# ORCID: 0009-0005-2382-3696

"""Read cell test files (Kollmeyer LG HG2 / Panasonic / Samsung style .mat or .csv) into a common form.

Returns dict with 1 Hz arrays: t [s], v [V], i [A, POSITIVE = DISCHARGE], ah [Ah, as logged], temp [degC or nan].
Field names differ between releases, so columns are matched by keyword. Run inspect_file.py first and,
if a field is not found, pass it explicitly, e.g. load(path, fields={'i': 'Current_A'}).
"""
import numpy as np

KEYS = {  # first match wins, case-insensitive substring
    't': ['time_s', 'prog time', 'progtime', 'time'],
    'v': ['voltage', 'volt'],
    'i': ['current', 'curr'],
    'ah': ['ah', 'capacity'],
    'temp': ['battery_temp', 'temperature', 'temp'],
}

def _pick(names, key, override):
    if override and key in override:
        return override[key]
    low = {n.lower(): n for n in names}
    for kw in KEYS[key]:
        for ln, n in low.items():
            if kw == ln or (kw in ln and not (key == 'ah' and 'wh' in ln)):
                if key == 't' and ('stamp' in ln or 'step' in ln):
                    continue
                return n
    return None

def _from_mat(path):
    from scipy.io import loadmat
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    s = m.get('meas')
    if s is None:  # take the first struct-like variable
        s = next(v for k, v in m.items() if not k.startswith('__'))
    return {f: np.atleast_1d(np.asarray(getattr(s, f))).astype(float, copy=False) if np.asarray(getattr(s, f)).dtype != object
            else np.asarray(getattr(s, f)) for f in s._fieldnames}

def _from_csv(path):
    import csv
    with open(path, newline='', encoding='utf-8', errors='replace') as fh:
        rows = list(csv.reader(fh))
    h = next(k for k, r in enumerate(rows) if any('volt' in c.lower() for c in r) and any('curr' in c.lower() for c in r))
    names = [c.strip() for c in rows[h]]
    data = {}
    body = rows[h + 1:]
    for j, n in enumerate(names):
        col = []
        for r in body:
            try:
                col.append(float(r[j]))
            except (ValueError, IndexError):
                col.append(np.nan)
        data[n] = np.array(col)
    return data

def _to_seconds(t):
    t = np.asarray(t)
    if t.dtype.kind in 'fiu':
        return t.astype(float)
    # "hh:mm:ss.s" strings
    out = []
    for x in t:
        p = str(x).split(':')
        out.append(sum(float(a) * 60 ** k for k, a in enumerate(reversed(p))))
    return np.array(out)

def load(path, fields=None, discharge_negative=True, dt=1.0):
    raw = _from_mat(path) if str(path).lower().endswith('.mat') else _from_csv(path)
    names = list(raw)
    col = {k: _pick(names, k, fields) for k in KEYS}
    missing = [k for k in ('t', 'v', 'i') if col[k] is None]
    if missing:
        raise KeyError(f'cannot find {missing} among fields {names}; pass fields={{...}}')
    t = _to_seconds(raw[col['t']]); t = t - t[0]
    v = np.asarray(raw[col['v']], float); i = np.asarray(raw[col['i']], float)
    ah = np.asarray(raw[col['ah']], float) if col['ah'] else None
    temp = np.asarray(raw[col['temp']], float) if col['temp'] else np.full_like(v, np.nan)
    ok = np.isfinite(t) & np.isfinite(v) & np.isfinite(i)
    t, v, i, temp = t[ok], v[ok], i[ok], temp[ok]
    if ah is not None:
        ah = ah[ok]
    order = np.argsort(t, kind='stable'); t, v, i, temp = t[order], v[order], i[order], temp[order]
    if ah is not None:
        ah = ah[order]
    # resample to a uniform grid: mean of V/I/T in each bin, last Ah (variable logging rate in source files)
    edges = np.arange(0, t[-1] + dt, dt)
    idx = np.clip(np.searchsorted(edges, t, side='right') - 1, 0, len(edges) - 1)
    cnt = np.bincount(idx, minlength=len(edges)); good = cnt > 0
    mean = lambda x: (np.bincount(idx, x, minlength=len(edges))[good] / cnt[good])
    tg = edges[good]
    # The cyclers log sparsely during rests (e.g. one sample per minute), so many 1 s bins are empty.
    # Fill them by linear interpolation so every output is on a UNIFORM 1 s grid (filters assume dt = 1 s).
    fill = lambda x: np.interp(edges, tg, x)
    # current: hold the previous logged value (cyclers log on change; a ramp would smear pulse edges)
    hold = lambda x: x[np.clip(np.searchsorted(tg, edges, side='right') - 1, 0, len(tg) - 1)]
    out = dict(t=edges.copy(), v=fill(mean(v)), i=hold(mean(i)),
               temp=fill(mean(np.nan_to_num(temp))) if np.isfinite(temp).any() else np.full(len(edges), np.nan))
    out['logged'] = good.copy()                    # True where the 1 s bin holds a real sample
    out['filled_fraction'] = float(1 - good.mean())
    sign = -1.0 if discharge_negative else 1.0
    out['i'] = sign * out['i']                      # internal convention: + = discharge
    if ah is not None:
        last = np.zeros(len(edges)); last[idx] = ah   # later samples overwrite earlier ones
        out['ah'] = fill(last[good])
    else:  # no Ah counter in the file: integrate current; Ah falls while discharging (same as the HG2 logs)
        out['ah'] = -np.concatenate([[0], np.cumsum(out['i'][:-1] * np.diff(out['t']))]) / 3600
    out['fields_used'] = col
    return out
