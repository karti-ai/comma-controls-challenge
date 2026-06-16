"""Re-fit the feedforward against the SIMULATOR (not the real-car logs).
Run the pid controller (tracks well) over many segs, log the (steer -> resulting
lataccel) pairs TinyPhysics actually produces, and fit steer = f(lat, roll, v).
This closes the sim/real gain gap directly.
Run: PYTHONPATH=. .venv/bin/python scratch/fit_ff_sim.py [num_segs]
"""
import sys
import numpy as np
from pathlib import Path
from multiprocessing import Pool

from tinyphysics import (TinyPhysicsModel, TinyPhysicsSimulator,
                         CONTROL_START_IDX, COST_END_IDX)
from controllers.pid import Controller as PID

MODEL_PATH = './models/tinyphysics.onnx'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
ALL = sorted(Path('data').glob('*.csv'))[:N]

_model = None
def _init():
    global _model
    _model = TinyPhysicsModel(MODEL_PATH, debug=False)

def _collect(seg):
    sim = TinyPhysicsSimulator(_model, str(seg), controller=PID(), debug=False)
    while sim.step_idx < COST_END_IDX:
        sim.step()
    lo, hi = CONTROL_START_IDX, COST_END_IDX
    steer = np.array(sim.action_history[lo:hi])
    lat = np.array(sim.current_lataccel_history[lo:hi])
    st = sim.state_history[lo:hi]
    roll = np.array([s.roll_lataccel for s in st])
    v = np.array([s.v_ego for s in st])
    a = np.array([s.a_ego for s in st])
    return np.column_stack([steer, lat, roll, v, a])

def fit(X, y, name):
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[m], y[m]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2)
    print(f"  {name:<34} R2={r2:.4f} rmse={np.sqrt(np.mean((y-pred)**2)):.4f}")
    print(f"     coef={np.round(coef,6).tolist()}")
    return coef

if __name__ == "__main__":
    print(f"collecting sim (steer->lataccel) from pid over {len(ALL)} segs...")
    with Pool(4, initializer=_init) as pool:
        data = np.vstack(pool.map(_collect, ALL))
    steer, lat, roll, v, a = data.T
    net = lat - roll
    ones = np.ones_like(net)
    print(f"samples: {len(steer)}")
    fit(np.column_stack([ones, lat]), steer, "1,lat")
    fit(np.column_stack([ones, lat, roll]), steer, "1,lat,roll")
    fit(np.column_stack([ones, net, net*v, net*v*v]), steer, "1,net,net*v,net*v2")
    fit(np.column_stack([ones, net, net*v, net*v*v, a, roll]), steer, "+a,roll")
    print("\ndata ranges: lat[%.2f,%.2f] roll[%.2f,%.2f] v[%.2f,%.2f]" % (
        lat.min(), lat.max(), roll.min(), roll.max(), v.min(), v.max()))
