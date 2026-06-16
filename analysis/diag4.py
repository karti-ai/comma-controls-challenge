"""Final fine sweep: pin navg (4-9) + perturb kp/kd/ff_scale around the optimum."""
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

def b(**kw):
    d = dict(ff=NOC0); d.update(kw); return d

TESTS = {
    'navg4':  b(ff_navg=4),
    'navg5':  b(ff_navg=5),
    'navg6':  b(ff_navg=6),
    'navg7':  b(ff_navg=7),
    'navg8':  b(ff_navg=8),
    'navg9':  b(ff_navg=9),
    'navg6 kp0.25':       b(ff_navg=6, kp=0.25),
    'navg6 kp0.30':       b(ff_navg=6, kp=0.30),
    'navg6 ffs1.05':      b(ff_navg=6, ff_scale=1.05),
    'navg6 ffs1.10':      b(ff_navg=6, ff_scale=1.10),
    'navg6 kd-0.03':      b(ff_navg=6, kd=-0.03),
    'navg6 kp0.25 ffs1.05': b(ff_navg=6, kp=0.25, ff_scale=1.05),
    'navg5 kp0.25 ffs1.05': b(ff_navg=5, kp=0.25, ff_scale=1.05),
    'navg6 kp0.25 kd-0.04 ffs1.05': b(ff_navg=6, kp=0.25, kd=-0.04, ff_scale=1.05),
}

if __name__ == "__main__":
    with Pool(4, initializer=_init) as pool:
        print(f"{'config':<32} {'total':>8} {'lat':>7} {'jerk':>7}")
        print("-" * 58)
        best = None
        for name, p in TESTS.items():
            t, l, j = evalp(p, pool)
            star = ""
            if best is None or t < best[2]:
                best, star = (name, p, t), "  <"
            print(f"{name:<32} {t:8.3f} {l:7.3f} {j:7.3f}{star}", flush=True)
        import json
        print(f"\nbest: {best[0]} ({best[2]:.3f})")
        print("PARAMS=" + json.dumps(best[1]))
