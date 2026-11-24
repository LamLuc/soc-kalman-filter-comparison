"""Drive cycles, speed [km/h] at 1 Hz.
nedc(): piecewise-linear reconstruction of ECE-15 (x4) + EUDC (UNECE R83 segment table; gear-shift
pauses merged into ramps) -> 1180 s, ~10.8 km.
urban(seed): synthetic stop-and-go. highway(seed): synthetic motorway driving.
Ramps use ceil() so realised accelerations never exceed the sampled target."""
import numpy as np

def _build(points):
    v = [0.0]
    for d, tgt in points:
        d = max(int(d), 1)
        s = v[-1]
        v += [s + (tgt - s) * k / d for k in range(1, d + 1)]
    return np.array(v)

ECE = [(11, 0), (4, 15), (8, 15), (5, 0), (21, 0), (12, 32), (24, 32), (11, 0), (21, 0),
       (26, 50), (12, 50), (8, 35), (13, 35), (12, 0), (7, 0)]
EUDC = [(20, 0), (41, 70), (50, 70), (8, 50), (69, 50), (13, 70), (50, 70), (35, 100),
        (30, 100), (20, 120), (10, 120), (34, 0), (20, 0)]

def nedc():
    return _build(ECE * 4 + EUDC)

def _ramp(dv_kmh, a):
    return int(np.ceil(abs(dv_kmh) / 3.6 / a))

def urban(seed, T=1200):
    """At least T s, always ending at standstill. Stops 5-20 s; accelerate at 1.5-3.0 m/s^2 to 25-75 km/h; cruise 5-40 s; brake at 2.0-3.5 m/s^2."""
    rng = np.random.default_rng(seed)
    pts, t = [], 0
    while t < T:
        idle = int(rng.integers(5, 21)); pts.append((idle, 0))
        vmax = float(rng.uniform(25, 75))
        acc = _ramp(vmax, rng.uniform(1.5, 3.0)); cru = int(rng.integers(5, 41)); dec = _ramp(vmax, rng.uniform(2.0, 3.5))
        pts += [(acc, vmax), (cru, vmax), (dec, 0)]
        t += idle + acc + cru + dec
    pts.append((5, 0))
    return _build(pts)   # ends at standstill (length ~T..T+150 s), so repeated cycles join continuously

def highway(seed, T=1800):
    """Motorway: accelerate to 90-125 km/h, cruise 60-300 s with +-15 km/h speed changes (0.5-1.2 m/s^2),
    occasional slow-down to 50-80 km/h; one full stop at start and end."""
    rng = np.random.default_rng(seed)
    pts = [(10, 0)]
    v = float(rng.uniform(95, 120)); pts.append((_ramp(v, rng.uniform(1.0, 2.0)), v)); t = sum(p[0] for p in pts)
    while t < T - 120:
        if rng.random() < 0.15:
            nv = float(rng.uniform(50, 80))
        else:
            nv = float(np.clip(v + rng.uniform(-15, 15), 90, 125))
        d = _ramp(nv - v, rng.uniform(0.5, 1.2)); c = int(rng.integers(60, 301))
        pts += [(d, nv), (c, nv)]; t += d + c; v = nv
    pts += [(_ramp(v, rng.uniform(1.5, 2.5)), 0), (10, 0)]
    return _build(pts)
