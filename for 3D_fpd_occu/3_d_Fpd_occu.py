#!/usr/bin/env python
"""
3-D lazy random walk (parallel + optional numba JIT, 32-bit seed safe)
- Reflecting boundary (Neumann): P(x, y=z=n0, t*)
- Absorbing boundary (Dirichlet): first-passage F(t)

Notes:
- ProcessPoolExecutor multiprocessing.
- Bounce-back reflecting boundary; non-exclusive symmetry pooling.
- Exports JSON before plotting.
- High-quality PDF export; no titles/captions; no uniform dashed line.
- If numba is installed (and not disabled), inner MC loops are JIT-compiled.
- Seeds are sanitized to 32-bit to avoid Numba OverflowError.
"""
import argparse, json, itertools
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ---------------- optional numba ----------------
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# ---------------- constants ----------------
DIRS = np.array([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)], dtype=np.int8)

# ---------------- analytics ----------------
def analytic_absorbing_F(q, L, n0, T_max_F):
    ka  = np.arange(1, L-1)
    psi = lambda k,i: np.sqrt(2/(L-1))*np.sin(np.pi*k*i/(L-1))
    A   = np.array([sum(psi(k,i) for i in range(1,L-1)) for k in ka])
    B   = np.array([psi(k,n0) for k in ka])
    AB  = A * B
    lam_abs = (1-q) + (q/3)*( np.cos(np.pi*ka[:,None,None]/(L-1))
                             + np.cos(np.pi*ka[None,:,None]/(L-1))
                             + np.cos(np.pi*ka[None,None,:]/(L-1)) )
    W = (AB[:,None,None] * AB[None,:,None] * AB[None,None,:])
    S = np.array([np.sum(W * (lam_abs**t)) for t in range(T_max_F+1)])
    F_ana = np.zeros_like(S);  F_ana[1:] = S[:-1] - S[1:]
    return F_ana

def analytic_reflecting_P(q, L, n0, t_star):
    k = np.arange(0, L)
    def phi(kk,i):
        return (np.sqrt(1/L) if kk==0 else np.sqrt(2/L)*np.cos((i+0.5)*np.pi*kk/L))
    phi_tab = np.array([[phi(kk,i) for i in range(L)] for kk in k])
    lam_ref = (1-q) + (q/3)*( np.cos(np.pi*k[:,None,None]/L)
                             + np.cos(np.pi*k[None,:,None]/L)
                             + np.cos(np.pi*k[None,None,:]/L) )
    P_ref = np.zeros(L)
    for kx,ky,kz in itertools.product(k,k,k):
        P_ref += (lam_ref[kx,ky,kz]**t_star) * \
                 phi_tab[kx] * phi_tab[kx,n0] * \
                 phi_tab[ky,n0]**2 * phi_tab[kz,n0]**2
    return P_ref

# ---------------- Python workers (fallback) ----------------
def _reflect_step_py(x,y,z, dx,dy,dz, L):
    nx, ny, nz = x+dx, y+dy, z+dz
    if nx < 0:      nx = 1
    elif nx >= L:   nx = L-2
    if ny < 0:      ny = 1
    elif ny >= L:   ny = L-2
    if nz < 0:      nz = 1
    elif nz >= L:   nz = L-2
    return nx, ny, nz

def mc_reflecting_worker_py(n_paths, L, n0, t_star, q, seed32):
    rng  = np.random.default_rng(np.uint32(seed32))
    cnt  = np.zeros(L, dtype=np.int64)
    for _ in range(n_paths):
        x = y = z = n0
        for _ in range(t_star):
            if rng.random() < q:
                dx,dy,dz = DIRS[rng.integers(6)]
                x,y,z = _reflect_step_py(x,y,z, dx,dy,dz, L)
        if (y==n0 and z==n0): cnt[x] += 1
        if (x==n0 and z==n0): cnt[y] += 1
        if (x==n0 and y==n0): cnt[z] += 1
    return cnt

def mc_absorbing_worker_py(n_paths, L, n0, T_max_F, q, seed32):
    rng   = np.random.default_rng(np.uint32(seed32))
    F_cnt = np.zeros(T_max_F+1, dtype=np.int64)
    for _ in range(n_paths):
        x = y = z = n0
        for t in range(1, T_max_F+1):
            if rng.random() < q:
                dx,dy,dz = DIRS[rng.integers(6)]
                x += dx; y += dy; z += dz
            if (x <= 0) or (x >= L-1) or (y <= 0) or (y >= L-1) or (z <= 0) or (z >= L-1):
                F_cnt[t] += 1
                break
    return F_cnt

# ---------------- Numba workers (auto if available) ----------------
if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _simulate_reflecting_numba(n_paths, L, n0, t_star, q, dirs, seed32):
        np.random.seed(np.uint32(seed32))
        cnt = np.zeros(L, np.int64)
        for _ in range(n_paths):
            x = n0; y = n0; z = n0
            for _ in range(t_star):
                if np.random.random() < q:
                    idx = np.random.randint(0, 6)
                    dx = int(dirs[idx,0]); dy = int(dirs[idx,1]); dz = int(dirs[idx,2])
                    nx = x + dx; ny = y + dy; nz = z + dz
                    if nx < 0:      nx = 1
                    elif nx >= L:   nx = L-2
                    if ny < 0:      ny = 1
                    elif ny >= L:   ny = L-2
                    if nz < 0:      nz = 1
                    elif nz >= L:   nz = L-2
                    x = nx; y = ny; z = nz
            if (y==n0 and z==n0): cnt[x] += 1
            if (x==n0 and z==n0): cnt[y] += 1
            if (x==n0 and y==n0): cnt[z] += 1
        return cnt

    @njit(cache=True, fastmath=True)
    def _simulate_absorbing_numba(n_paths, L, n0, T_max_F, q, dirs, seed32):
        np.random.seed(np.uint32(seed32))
        F_cnt = np.zeros(T_max_F+1, np.int64)
        for _ in range(n_paths):
            x = n0; y = n0; z = n0
            for t in range(1, T_max_F+1):
                if np.random.random() < q:
                    idx = np.random.randint(0, 6)
                    dx = int(dirs[idx,0]); dy = int(dirs[idx,1]); dz = int(dirs[idx,2])
                    x += dx; y += dy; z += dz
                if (x <= 0) or (x >= L-1) or (y <= 0) or (y >= L-1) or (z <= 0) or (z >= L-1):
                    F_cnt[t] += 1
                    break
        return F_cnt

def mc_reflecting_worker(n_paths, L, n0, t_star, q, seed32, use_numba):
    if NUMBA_AVAILABLE and use_numba:
        return _simulate_reflecting_numba(n_paths, L, n0, t_star, q, DIRS, np.uint32(seed32))
    else:
        return mc_reflecting_worker_py(n_paths, L, n0, t_star, q, seed32)

def mc_absorbing_worker(n_paths, L, n0, T_max_F, q, seed32, use_numba):
    if NUMBA_AVAILABLE and use_numba:
        return _simulate_absorbing_numba(n_paths, L, n0, T_max_F, q, DIRS, np.uint32(seed32))
    else:
        return mc_absorbing_worker_py(n_paths, L, n0, T_max_F, q, seed32)

# ---------------- parallel drivers ----------------
def split_load(total, n_tasks):
    base = total // n_tasks
    rem  = total %  n_tasks
    return [base + (1 if i < rem else 0) for i in range(n_tasks)]

def _make_task_seeds(total_tasks, seed_master):
    ss   = np.random.SeedSequence(int(seed_master))
    kids = ss.spawn(total_tasks)
    raw  = np.array([int(k.generate_state(1, dtype=np.uint64)[0]) for k in kids], dtype=np.uint64)
    seeds32 = (raw & np.uint64(0xFFFF_FFFF)).astype(np.uint32)  # clamp to 32-bit
    return seeds32.tolist()

def run_parallel_reflecting(Nw_P, L, n0, t_star, q, seed, workers, use_numba):
    n_tasks = max(workers * 4, 1)
    paths_per_task = [n for n in split_load(Nw_P, n_tasks) if n > 0]
    task_seeds = _make_task_seeds(len(paths_per_task), seed)
    agg = np.zeros(L, dtype=np.int64)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(mc_reflecting_worker, n, L, n0, t_star, q, int(s), use_numba)
                for n,s in zip(paths_per_task, task_seeds)]
        for fu in as_completed(futs):
            agg += fu.result()
    return agg / (3.0 * Nw_P)

def run_parallel_absorbing(Nw_F, L, n0, T_max_F, q, seed, workers, use_numba):
    n_tasks = max(workers * 4, 1)
    paths_per_task = [n for n in split_load(Nw_F, n_tasks) if n > 0]
    task_seeds = _make_task_seeds(len(paths_per_task), seed + 1)
    agg = np.zeros(T_max_F+1, dtype=np.int64)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(mc_absorbing_worker, n, L, n0, T_max_F, q, int(s), use_numba)
                for n,s in zip(paths_per_task, task_seeds)]
        for fu in as_completed(futs):
            agg += fu.result()
    return agg / Nw_F

# ---------------- plotting/export ----------------
def setup_matplotlib():
    plt.rcParams["savefig.bbox"]       = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.02
    plt.rcParams["pdf.fonttype"]       = 42
    plt.rcParams["ps.fonttype"]        = 42
    plt.rcParams["figure.dpi"]         = 150

def plot_and_save(x_axis, P_ref, P_ref_mc, t_axis, F_ana, F_mc, out_dir, t_star):
    fig1 = plt.figure(figsize=(6.0, 4.0), constrained_layout=True)
    ax1 = fig1.gca()
    ax1.plot(x_axis, P_ref, label=f"Analytic  t={t_star}")
    ax1.scatter(x_axis, P_ref_mc, s=10, label="MC")
    ax1.set_xlabel("x   (y = z = n0)")
    ax1.set_ylabel("P")
    ax1.legend(frameon=False, loc="best")
    fig1.savefig(out_dir/"P_ref.pdf")
    fig1.savefig(out_dir/"P_ref.png", dpi=300)

    fig2 = plt.figure(figsize=(6.0, 4.0), constrained_layout=True)
    ax2 = fig2.gca()
    ax2.plot(t_axis, F_ana[1:], label="Analytic")
    ax2.scatter(t_axis, F_mc[1:], s=6, label="MC")
    ax2.set_xlabel("time step  t")
    ax2.set_ylabel("F(t)")
    ax2.legend(frameon=False, loc="best")
    fig2.savefig(out_dir/"F_abs.pdf")
    fig2.savefig(out_dir/"F_abs.png", dpi=300)
    plt.show()

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(description="3D lazy random walk: parallel MC + analytics (+ optional numba, seed-safe)")
    parser.add_argument("--q", type=float, default=0.9)
    parser.add_argument("--L", type=int, default=31)
    parser.add_argument("--n0", type=int, default=15)
    parser.add_argument("--t_star", type=int, default=300)
    parser.add_argument("--T_max_F", type=int, default=800)
    parser.add_argument("--Nw_P", type=int, default=4_000_000)
    parser.add_argument("--Nw_F", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--workers", type=int, default=None, help="defaults to min(8, cpu_count())")
    parser.add_argument("--disable-numba", action="store_true")
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    workers = args.workers if args.workers is not None else min(8, mp.cpu_count())
    use_numba = (NUMBA_AVAILABLE and not args.disable_numba)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()

    F_ana  = analytic_absorbing_F(args.q, args.L, args.n0, args.T_max_F)
    P_ref  = analytic_reflecting_P(args.q, args.L, args.n0, args.t_star)

    P_ref_mc = run_parallel_reflecting(args.Nw_P, args.L, args.n0, args.t_star,
                                       args.q, args.seed, workers, use_numba)
    F_mc     = run_parallel_absorbing(args.Nw_F, args.L, args.n0, args.T_max_F,
                                      args.q, args.seed, workers, use_numba)

    x_axis = np.arange(args.L)
    t_axis = np.arange(1, args.T_max_F+1)
    payload = {
        "params": {
            "q": args.q, "L": args.L, "n0": args.n0,
            "t_star": args.t_star, "T_max_F": args.T_max_F,
            "Nw_P": int(args.Nw_P), "Nw_F": int(args.Nw_F),
            "seed": int(args.seed), "workers": int(workers),
            "numba_enabled": bool(use_numba)
        },
        "axes": {"x": x_axis.tolist(), "t": t_axis.tolist()},
        "reflecting": {"P_analytic": P_ref.tolist(), "P_mc": P_ref_mc.tolist()},
        "absorbing":  {"F_analytic": F_ana[1:].tolist(), "F_mc": F_mc[1:].tolist()}
    }
    with open(out_dir/"results.json", "w") as f:
        json.dump(payload, f)

    plot_and_save(x_axis, P_ref, P_ref_mc, t_axis, F_ana, F_mc, out_dir, args.t_star)

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
