"""Coordinate-descent tuner for controllers/karti.py against the TinyPhysics sim.
Deterministic (sim is seeded per-segment), so the search is noise-free.
Runs to COST_END_IDX only (identical cost to a full rollout) to save compute.
"""
import json
import sys
import numpy as np
from pathlib import Path
from multiprocessing import Pool

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, COST_END_IDX
from controllers.karti import Controller

MODEL_PATH = './models/tinyphysics.onnx'
ALL = sorted(Path('data').glob('*.csv'))
TUNE = ALL[:80]            # fast search set
VAL = ALL[80:480]          # held-out validation

_model = None
def _init():
    global _model
    _model = TinyPhysicsModel(MODEL_PATH, debug=False)

def _eval_one(args):
    seg, params = args
    sim = TinyPhysicsSimulator(_model, str(seg), controller=Controller(params), debug=False)
    while sim.step_idx < COST_END_IDX:
        sim.step()
    c = sim.compute_cost()
    return (c['total_cost'], c['lataccel_cost'], c['jerk_cost'])

def evaluate(params, segs, pool):
    res = np.array(pool.map(_eval_one, [(s, params) for s in segs]))
    return res[:, 0].mean(), res[:, 1].mean(), res[:, 2].mean()

GRIDS = {
    'ff_scale': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
    'kp':       [0.0, 0.05, 0.1, 0.2, 0.3],
    'ki':       [0.0, 0.02, 0.05, 0.1],
    'kd':       [-0.08, -0.05, -0.02, 0.0],
    'preview':  [0, 1, 2, 3, 4],
    'int_clip': [0.5, 1.0, 2.0],
}
ORDER = ['ff_scale', 'kp', 'ki', 'kd', 'preview', 'int_clip']

def main():
    cur = dict(ff_scale=1.0, kp=0.10, ki=0.05, kd=-0.02, preview=2, int_clip=1.0)
    cache = {}
    def ev(params):
        key = tuple(sorted(params.items()))
        if key not in cache:
            cache[key] = evaluate(params, TUNE, pool)
        return cache[key]

    with Pool(4, initializer=_init) as pool:
        base = ev(cur)
        print(f"start {cur} -> total={base[0]:.3f} (lat={base[1]:.3f} jerk={base[2]:.3f})", flush=True)
        for it in range(2):
            print(f"\n===== pass {it+1} =====", flush=True)
            for name in ORDER:
                best_v, best_cost = cur[name], ev(cur)[0]
                for v in GRIDS[name]:
                    if v == cur[name]:
                        continue
                    trial = dict(cur); trial[name] = v
                    cost = ev(trial)[0]
                    tag = ""
                    if cost < best_cost:
                        best_cost, best_v, tag = cost, v, "  <-- best"
                    print(f"  {name}={v!s:<7} total={cost:.3f}{tag}", flush=True)
                cur[name] = best_v
                print(f"  => {name} := {best_v}  (total={best_cost:.3f})", flush=True)

        tcost = ev(cur)
        print(f"\nFINAL params: {cur}", flush=True)
        print(f"TUNE(80):  total={tcost[0]:.3f} lat={tcost[1]:.3f} jerk={tcost[2]:.3f}", flush=True)
        vcost = evaluate(cur, VAL, pool)
        print(f"VAL(400):  total={vcost[0]:.3f} lat={vcost[1]:.3f} jerk={vcost[2]:.3f}", flush=True)
        Path('scratch/best_params.json').write_text(json.dumps(cur, indent=2))
        print("saved scratch/best_params.json", flush=True)

if __name__ == "__main__":
    main()
