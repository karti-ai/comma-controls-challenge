"""Sweep the future-target averaging window (navg) + forward roll-avg + PID tweaks."""
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, COST_END_IDX
from controllers.karti import Controller

MODEL_PATH = './models/tinyphysics.onnx'
SEGS = sorted(Path('data').glob('*.csv'))[:100]
NOC0 = (0.0, 0.70797, -0.006298, -0.000029)

_model = None
def _init():
    global _model
    _model = TinyPhysicsModel(MODEL_PATH, debug=False)

def _ev(args):
    seg, params = args
    sim = TinyPhysicsSimulator(_model, str(seg), controller=Controller(params), debug=False)
    while sim.step_idx < COST_END_IDX:
        sim.step()
    c = sim.compute_cost()
    return c['total_cost'], c['lataccel_cost'], c['jerk_cost']

def evalp(params, pool):
    r = np.array(pool.map(_ev, [(s, params) for s in SEGS]))
    return r[:, 0].mean(), r[:, 1].mean(), r[:, 2].mean()

def base(**kw):
    d = dict(ff=NOC0)
    d.update(kw)
    return d

TESTS = {
    'navg8':                  base(ff_navg=8),
    'navg10':                 base(ff_navg=10),
    'navg12':                 base(ff_navg=12),
    'navg15':                 base(ff_navg=15),
    'navg20':                 base(ff_navg=20),
    'navg25':                 base(ff_navg=25),
    'navg15 + rollnavg10':    base(ff_navg=15, roll_navg=10),
    'navg15 + rollnavg20':    base(ff_navg=15, roll_navg=20),
    'navg15 + kd=-0.03':      base(ff_navg=15, kd=-0.03),
    'navg15 + ki=0.05':       base(ff_navg=15, ki=0.05),
    'navg15 + ki0.05 kd-0.03':base(ff_navg=15, ki=0.05, kd=-0.03),
    'navg12 keepC0':          dict(ff_navg=12),
}

if __name__ == "__main__":
    with Pool(4, initializer=_init) as pool:
        print(f"{'config':<28} {'total':>8} {'lat':>7} {'jerk':>7}")
        print("-" * 54)
        best = None
        for name, p in TESTS.items():
            t, l, j = evalp(p, pool)
            star = ""
            if best is None or t < best[1]:
                best, star = (name, t), "  <"
            print(f"{name:<28} {t:8.3f} {l:7.3f} {j:7.3f}{star}", flush=True)
        print(f"\nbest: {best[0]} ({best[1]:.3f})  | beat pid=84.85")
