"""Validate the controller's baked-in defaults on a held-out segment range.
Run: PYTHONPATH=. .venv/bin/python scratch/validate.py LO HI [controller]
"""
import sys
import importlib
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, COST_END_IDX

MODEL_PATH = './models/tinyphysics.onnx'
LO = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
HI = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
CTRL = sys.argv[3] if len(sys.argv) > 3 else 'karti'
SEGS = sorted(Path('data').glob('*.csv'))[LO:HI]
Controller = importlib.import_module(f'controllers.{CTRL}').Controller

_model = None
def _init():
    global _model
    _model = TinyPhysicsModel(MODEL_PATH, debug=False)

def _ev(seg):
    sim = TinyPhysicsSimulator(_model, str(seg), controller=Controller(), debug=False)
    limit = min(COST_END_IDX, len(sim.data))   # some segments are shorter than 500 rows
    while sim.step_idx < limit:
        sim.step()
    c = sim.compute_cost()
    return c['total_cost'], c['lataccel_cost'], c['jerk_cost']

if __name__ == "__main__":
    with Pool(4, initializer=_init) as pool:
        r = np.array(pool.map(_ev, SEGS))
    tot = r[:, 0]
    print(f"controller={CTRL}  held-out segs [{LO}:{HI}] (n={len(SEGS)})")
    print(f"  total={tot.mean():.3f}  lat={r[:,1].mean():.3f}  jerk={r[:,2].mean():.3f}")
    print(f"  median total={np.median(tot):.3f}  %under100={100*(tot<100).mean():.1f}%")
