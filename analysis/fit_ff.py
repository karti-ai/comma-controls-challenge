"""Fit a feedforward inverse-model: given desired lataccel + state, what steer?
Also sweep a lead delay d (steer_t vs target_{t+d}) to find the natural preview.
Run: .venv/bin/python scratch/fit_ff.py [num_files]
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ACC_G = 9.81
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
files = sorted(Path('data').glob('*.csv'))[:N]
print(f"loading {len(files)} segments...")

cols = []
for f in files:
    df = pd.read_csv(f)
    roll_la = np.sin(df['roll'].values) * ACC_G
    v = df['vEgo'].values
    a = df['aEgo'].values
    tgt = df['targetLateralAcceleration'].values
    steer = -df['steerCommand'].values  # right-positive, as sim uses
    cols.append((steer, tgt, roll_la, v, a))

def stack(delay=0):
    S, T, R, V, A = [], [], [], [], []
    for steer, tgt, roll, v, a in cols:
        n = len(steer)
        if delay >= 0:
            s = steer[:n-delay] if delay else steer
            t = tgt[delay:]
            r = roll[:n-delay] if delay else roll
            vv = v[:n-delay] if delay else v
            aa = a[:n-delay] if delay else a
        S.append(s); T.append(t); R.append(r); V.append(vv); A.append(aa)
    return (np.concatenate(S), np.concatenate(T), np.concatenate(R),
            np.concatenate(V), np.concatenate(A))

def fit(X, y, name):
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = np.sum((y - pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y - pred)**2))
    print(f"  {name:<42} R2={r2:.4f} rmse={rmse:.4f}")
    print(f"       coef={np.round(coef, 6).tolist()}")
    return coef, r2

print("\n=== lead-delay sweep (model: 1, net, net*v, net*v^2  where net=tgt-roll) ===")
best = None
for d in range(0, 7):
    steer, tgt, roll, v, a = stack(d)
    net = tgt - roll
    X = np.column_stack([np.ones_like(net), net, net*v, net*v*v])
    coef, r2 = fit(X, steer, f"delay={d}")
    if best is None or r2 > best[1]:
        best = (d, r2, coef)
print(f"\nBEST lead delay d={best[0]} (R2={best[1]:.4f})")

print("\n=== richer models at best delay ===")
d = best[0]
steer, tgt, roll, v, a = stack(d)
net = tgt - roll
ones = np.ones_like(net)
fit(np.column_stack([ones, tgt]), steer, "M0: 1,tgt")
fit(np.column_stack([ones, tgt, roll]), steer, "M1: 1,tgt,roll")
fit(np.column_stack([ones, net, net*v, net*v*v]), steer, "M2: 1,net,net*v,net*v2")
c, _ = fit(np.column_stack([ones, net, net*v, net*v*v, a, roll]), steer,
           "M3: +a,roll")
print("\nv_ego range:", round(float(steer.min()), 3))
print("data ranges: tgt[%.2f,%.2f] roll[%.2f,%.2f] v[%.2f,%.2f]" % (
    tgt.min(), tgt.max(), roll.min(), roll.max(), v.min(), v.max()))
