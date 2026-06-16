"""Diagnostic matrix: evaluate several controller configs on a fixed 100-seg set."""
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

TESTS = {
    'pid-repro (FF off, baseline gains)': dict(ff=(0,0,0,0), preview=0, kp=0.195, ki=0.1, kd=-0.053, int_clip=1e9),
    'FF + baseline-PID gains':            dict(preview=0, kp=0.195, ki=0.1, kd=-0.053, int_clip=1e9),
    'FF + strong PID':                    dict(preview=0, kp=0.4, ki=0.1, kd=-0.05, int_clip=1.0),
    'FF + very strong PID':               dict(preview=0, kp=0.6, ki=0.15, kd=-0.07, int_clip=1.0),
    'FF noC0 + baseline gains':           dict(ff=NOC0, preview=0, kp=0.195, ki=0.1, kd=-0.053, int_clip=1e9),
    'FF noC0 + strong PID':               dict(ff=NOC0, preview=0, kp=0.4, ki=0.1, kd=-0.05, int_clip=1.0),
    'FF noC0 + strong PID + preview1':    dict(ff=NOC0, preview=1, kp=0.4, ki=0.1, kd=-0.05, int_clip=1.0),
}

if __name__ == "__main__":
    with Pool(4, initializer=_init) as pool:
        print(f"{'config':<38} {'total':>8} {'lat':>7} {'jerk':>7}")
        print("-" * 64)
        for name, p in TESTS.items():
            t, l, j = evalp(p, pool)
            print(f"{name:<38} {t:8.3f} {l:7.3f} {j:7.3f}", flush=True)
        print("\n(target to beat: pid total=84.85 lat=1.27 jerk=21.3)")
