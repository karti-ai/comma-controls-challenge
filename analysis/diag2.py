"""Smoothing sweep: keep FF's tracking gain, kill the jerk it injects."""
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, COST_END_IDX
from controllers.karti import Controller

MODEL_PATH = './models/tinyphysics.onnx'
SEGS = sorted(Path('data').glob('*.csv'))[:100]

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

# inherent jerk if you tracked the target perfectly (floor reference)
jf = []
for seg in SEGS:
    t = pd.read_csv(seg)['targetLateralAcceleration'].values[CONTROL_START_IDX:COST_END_IDX]
    jf.append(np.mean((np.diff(t) / 0.1) ** 2) * 100)
print(f"target inherent jerk_cost (perfect-tracking floor): {np.mean(jf):.2f}\n")

TESTS = {
    'FF base (no smoothing)':        dict(),
    'roll_a=0.3':                    dict(roll_alpha=0.3),
    'roll_a=0.2':                    dict(roll_alpha=0.2),
    'roll_a=0.1':                    dict(roll_alpha=0.1),
    'navg=5':                        dict(ff_navg=5),
    'navg=10':                       dict(ff_navg=10),
    'ff_a=0.4':                      dict(ff_alpha=0.4),
    'roll0.2 + navg5':               dict(roll_alpha=0.2, ff_navg=5),
    'roll0.2 + navg5 + ff_a0.5':     dict(roll_alpha=0.2, ff_navg=5, ff_alpha=0.5),
    'roll0.15+navg8+ff_a0.4':        dict(roll_alpha=0.15, ff_navg=8, ff_alpha=0.4),
    'roll0.2+navg5 + gentle pid':    dict(roll_alpha=0.2, ff_navg=5, kp=0.12, ki=0.06, kd=-0.03),
    'noC0 roll0.2 navg5':            dict(ff=(0.0,0.70797,-0.006298,-0.000029), roll_alpha=0.2, ff_navg=5),
}

if __name__ == "__main__":
    with Pool(4, initializer=_init) as pool:
        print(f"{'config':<32} {'total':>8} {'lat':>7} {'jerk':>7}")
        print("-" * 58)
        for name, p in TESTS.items():
            t, l, j = evalp(p, pool)
            print(f"{name:<32} {t:8.3f} {l:7.3f} {j:7.3f}", flush=True)
        print("\n(beat: pid total=84.85 lat=1.27 jerk=21.3)")
