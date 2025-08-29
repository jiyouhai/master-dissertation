#!/usr/bin/env python
# 2-D Lazy Random Walk (Reflecting P & Absorbing F) — parallel MC, PDF output

import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # comment out for interactive use
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"]  = 42
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------- default parameters ----------
q       = 0.9
L       = 51
n0      = 25
t_star  = 200
T_max_F = 600
Nw_P    = 1_000_000
Nw_F    = 1_000_000
seed    = 2025
# ----------------------------------------

# ---------- analytics ----------
def _phi(k_, i, L_):
    if k_ == 0:
        return np.sqrt(1.0 / L_)
    return np.sqrt(2.0 / L_) * np.cos((i + 0.5) * np.pi * k_ / L_)

def analytic_P_reflecting(L_, n0_, q_, t_star_):
    k = np.arange(L_)
    phi_tab = np.array([[ _phi(kk, i, L_) for i in range(L_) ] for kk in k])  # (L,L)
    lam_ref = (1 - q_) + (q_ / 2) * (
        np.cos(np.pi * k[:, None] / L_) + np.cos(np.pi * k[None, :] / L_)
    )  # (L,L)
    P_ref = np.zeros(L_)
    for kx in k:
        for ky in k:
            P_ref += (lam_ref[kx, ky] ** t_star_) * phi_tab[kx] * phi_tab[kx, n0_] * (phi_tab[ky, n0_] ** 2)
    return P_ref

def analytic_F_absorbing(L_, n0_, q_, T_max_):
    ka = np.arange(1, L_ - 1)
    def psi(k_, i):
        return np.sqrt(2.0 / (L_ - 1)) * np.sin(np.pi * k_ * i / (L_ - 1))
    A = np.array([sum(psi(k_, i) for i in range(1, L_ - 1)) for k_ in ka])
    B = np.array([psi(k_, n0_) for k_ in ka])
    lam = (1 - q_) + (q_ / 2) * (
        np.cos(np.pi * ka[:, None] / (L_ - 1)) + np.cos(np.pi * ka[None, :] / (L_ - 1))
    )
    weight = (A * B)[:, None] * (A * B)[None, :]
    S = np.array([np.sum(weight * (lam ** t)) for t in range(T_max_ + 1)])
    F_ana = np.zeros_like(S)
    F_ana[1:] = S[:-1] - S[1:]
    return F_ana

# ---------- vectorized MC workers ----------
def _mc_reflecting_P_counts(batch_size, q_, L_, n0_, t_star_, seed_):
    rng = np.random.default_rng(int(seed_))
    x = np.full(batch_size, n0_, dtype=np.int32)
    y = np.full(batch_size, n0_, dtype=np.int32)

    for _ in range(t_star_):
        move = rng.random(batch_size) < q_
        dirs = rng.integers(0, 4, size=batch_size)
        dx = (dirs == 0).astype(np.int8) - (dirs == 1).astype(np.int8)
        dy = (dirs == 2).astype(np.int8) - (dirs == 3).astype(np.int8)
        x += dx * move
        y += dy * move
        np.clip(x, 0, L_ - 1, out=x)
        np.clip(y, 0, L_ - 1, out=y)

    mask = (y == n0_)
    if not np.any(mask):
        return np.zeros(L_, dtype=np.int64)
    return np.bincount(x[mask], minlength=L_).astype(np.int64)

def _mc_absorbing_F_counts(batch_size, q_, L_, n0_, T_max_, seed_):
    rng = np.random.default_rng(int(seed_))
    x = np.full(batch_size, n0_, dtype=np.int32)
    y = np.full(batch_size, n0_, dtype=np.int32)
    active = np.ones(batch_size, dtype=bool)
    F_cnt = np.zeros(T_max_ + 1, dtype=np.int64)

    for t in range(1, T_max_ + 1):
        if not np.any(active):
            break
        move = rng.random(batch_size) < q_
        dirs = rng.integers(0, 4, size=batch_size)
        dx = (dirs == 0).astype(np.int8) - (dirs == 1).astype(np.int8)
        dy = (dirs == 2).astype(np.int8) - (dirs == 3).astype(np.int8)
        m = active & move
        x[m] += dx[m]
        y[m] += dy[m]
        newly = active & ((x <= 0) | (x >= L_ - 1) | (y <= 0) | (y >= L_ - 1))
        if newly.any():
            F_cnt[t] += int(newly.sum())
            active[newly] = False
    return F_cnt

# ---------- parallel helpers ----------
def _split_batches(total, chunk):
    n_full, rem = divmod(total, chunk)
    return [chunk] * n_full + ([rem] if rem else [])

def _seed_for(base, idx):
    return int(base + (idx + 1) * 1_000_003)

def run_mc_reflecting_P(total_walkers, chunk, jobs, q_, L_, n0_, t_star_, seed_):
    sizes = _split_batches(total_walkers, chunk)
    counts = np.zeros(L_, dtype=np.int64)
    if not sizes:
        return counts
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(_mc_reflecting_P_counts, bs, q_, L_, n0_, t_star_, _seed_for(seed_, i))
                for i, bs in enumerate(sizes)]
        for f in as_completed(futs):
            counts += f.result()
    return counts

def run_mc_absorbing_F(total_walkers, chunk, jobs, q_, L_, n0_, T_max_, seed_):
    sizes = _split_batches(total_walkers, chunk)
    F_cnt = np.zeros(T_max_ + 1, dtype=np.int64)
    if not sizes:
        return F_cnt
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(_mc_absorbing_F_counts, bs, q_, L_, n0_, T_max_, _seed_for(seed_ + 1234567, i))
                for i, bs in enumerate(sizes)]
        for f in as_completed(futs):
            F_cnt += f.result()
    return F_cnt

# ---------- plotting (PDF only, no titles, no dashed guides) ----------
def plot_results_pdf(x_axis, P_ref, P_ref_mc, t_axis, F_ana, F_mc):
    plt.figure(figsize=(5.5, 3.4))
    plt.plot(x_axis, P_ref, lw=1.5, label=f"Analytic  t={t_star}")
    plt.scatter(x_axis, P_ref_mc, s=10, label="MC")
    plt.xlabel("x  (y = n0)")
    plt.ylabel("P")
    plt.legend()
    plt.tight_layout()
    plt.savefig("P_reflecting.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5.5, 3.4))
    plt.plot(t_axis, F_ana[1:], lw=1.2, label="Analytic")
    plt.scatter(t_axis, F_mc[1:], s=6, label="MC")
    plt.xlabel("time step  t")
    plt.ylabel("F(t)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("F_absorbing.pdf", bbox_inches="tight")
    plt.close()

# ---------- JSON export ----------
def export_json(path, params, x_axis, t_axis, P_ref, F_ana, P_ref_mc, F_mc):
    data = {
        "params": params,
        "axes": {"x_axis": x_axis.tolist(), "t_axis": t_axis.tolist()},
        "analytic": {"P_ref": P_ref.tolist(), "F_ana": F_ana[1:].tolist()},
        "mc": {"P_ref_mc": P_ref_mc.tolist(), "F_mc": F_mc[1:].tolist()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---------- CLI / main ----------
def main():
    parser = argparse.ArgumentParser(description="Parallel MC for 2D lazy random walk; outputs PDF figures")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(), help="parallel processes")
    parser.add_argument("--chunk", type=int, default=100_000, help="walkers per task")
    parser.add_argument("--NwP", type=int, default=Nw_P, help="MC walkers for P")
    parser.add_argument("--NwF", type=int, default=Nw_F, help="MC walkers for F")
    parser.add_argument("--seed", type=int, default=seed, help="base RNG seed")
    args = parser.parse_args()

    P_ref = analytic_P_reflecting(L, n0, q, t_star)
    F_ana = analytic_F_absorbing(L, n0, q, T_max_F)

    P_counts = run_mc_reflecting_P(args.NwP, args.chunk, args.jobs, q, L, n0, t_star, args.seed)
    P_ref_mc = P_counts / args.NwP

    F_cnt = run_mc_absorbing_F(args.NwF, args.chunk, args.jobs, q, L, n0, T_max_F, args.seed)
    F_mc = F_cnt / args.NwF

    x_axis = np.arange(L)
    t_axis = np.arange(1, T_max_F + 1)
    params = {
        "q": q, "L": L, "n0": n0, "t_star": t_star, "T_max_F": T_max_F,
        "Nw_P": int(args.NwP), "Nw_F": int(args.NwF), "seed": int(args.seed),
        "jobs": int(args.jobs), "chunk": int(args.chunk)
    }

    export_json("rw2d_results.json", params, x_axis, t_axis, P_ref, F_ana, P_ref_mc, F_mc)
    plot_results_pdf(x_axis, P_ref, P_ref_mc, t_axis, F_ana, F_mc)

if __name__ == "__main__":
    main()
