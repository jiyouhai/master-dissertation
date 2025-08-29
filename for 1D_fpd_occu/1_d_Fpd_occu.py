#!/usr/bin/env python3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
from pathlib import Path

# High-quality PDF settings (no color styling specified)
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
})

# ---- Parameters ----
q, t_star, T_max_F, seed = 0.9, 200, 600, 2025
L, n0 = 51, 25
Nw_P, Nw_F = 4_0000, 2_00000
FIGSIZE = (3.6, 2.7)
OUTDIR = Path("figs"); OUTDIR.mkdir(exist_ok=True)
rng = np.random.default_rng(seed)

# ---- Analytic (reflecting) ----
k = np.arange(L)
lam = (1 - q) + q * np.cos(np.pi * k / L)

def phi(kk, n):
    return (np.sqrt(1 / L) if kk == 0
            else np.sqrt(2 / L) * np.cos((n + 0.5) * np.pi * kk / L))

phi_tab = np.array([[phi(kk, n) for n in range(L)] for kk in k])
P_ref_vec = ((lam**t_star)[:, None] * phi_tab * phi_tab[:, n0:n0+1]).sum(axis=0)

# ---- Analytic (absorbing) ----
ka = np.arange(1, L - 1)
lam_a = (1 - q) + q * np.cos(np.pi * ka / (L - 1))

def psi(kk, n):
    return np.sqrt(2 / (L - 1)) * np.sin(np.pi * kk * (n + 1) / (L - 1))

psi_tab = np.array([[psi(kk, n) for n in range(1, L - 1)] for kk in ka])

P_abs = np.zeros((L - 2, T_max_F + 1))
for t in range(T_max_F + 1):
    P_abs[:, t] = ((lam_a**t)[:, None] * psi_tab * psi_tab[:, n0 - 1:n0]).sum(axis=0)
S = P_abs.sum(axis=0)
F_ana = np.zeros_like(S)
F_ana[1:] = S[:-1] - S[1:]

# ---- Monte Carlo (reflecting) ----
def step_ref(pos):
    u = rng.random()
    if u < q / 2:
        pos = max(0, pos - 1)
    elif u < q:
        pos = min(L - 1, pos + 1)
    return pos

cnt = np.zeros(L, int)
for _ in range(Nw_P):
    p = n0
    for _ in range(t_star):
        p = step_ref(p)
    cnt[p] += 1
P_ref_mc = cnt / Nw_P

# ---- Monte Carlo (absorbing) ----
def step_abs(pos):
    u = rng.random()
    if u < q / 2:
        pos -= 1
    elif u < q:
        pos += 1
    return pos

F_cnt = np.zeros(T_max_F + 1, int)
for _ in range(Nw_F):
    p = n0
    for t in range(1, T_max_F + 1):
        p = step_abs(p)
        if p == 0 or p == L - 1:
            F_cnt[t] += 1
            break
F_mc = F_cnt / Nw_F

# ---- JSON (before plotting) ----
out = {
    "params": {
        "q": q, "t_star": t_star, "T_max_F": T_max_F, "seed": int(seed),
        "L": L, "n0": n0, "Nw_P": int(Nw_P), "Nw_F": int(Nw_F)
    },
    "grid": {"n": list(range(L)), "t": list(range(1, T_max_F + 1))},
    "P_ref_analytic": P_ref_vec.tolist(),
    "P_ref_mc": P_ref_mc.tolist(),
    "F_analytic": F_ana[1:].tolist(),
    "F_mc": F_mc[1:].tolist()
}
Path("one_d_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

# ---- Figure 1: P_ref(n, t*) ----
fig1 = plt.figure(figsize=FIGSIZE)
ax1 = fig1.add_subplot(111)
n_axis = np.arange(L)
ax1.plot(n_axis, P_ref_vec, label="Analytic")
ax1.scatter(n_axis, P_ref_mc, s=12, label="Sim")
ax1.set_xlabel("n")
ax1.set_ylabel("P")
ax1.legend()
fig1.savefig(OUTDIR / "P_ref.pdf", bbox_inches="tight", pad_inches=0.01, transparent=True)

# ---- Figure 2: F_abs(t) ----
fig2 = plt.figure(figsize=FIGSIZE)
ax2 = fig2.add_subplot(111)
t = np.arange(1, T_max_F + 1)
ax2.plot(t, F_ana[1:], label="Analytic")
ax2.scatter(t, F_mc[1:], s=8, label="Sim")
ax2.set_xlabel("t")
ax2.set_ylabel("F")
ax2.legend()
fig2.savefig(OUTDIR / "F_abs.pdf", bbox_inches="tight", pad_inches=0.01, transparent=True)

# Close figures (optional in scripts; helps in notebooks)
plt.close(fig1); plt.close(fig2)

# ---- Inline PDF display in notebooks (optional) ----
try:
    from IPython.display import display, IFrame
    display(IFrame(src=str(OUTDIR / "P_ref.pdf"), width=640, height=480))
    display(IFrame(src=str(OUTDIR / "F_abs.pdf"), width=640, height=480))
except Exception:
    # If not in a notebook, just skip inline display
    pass
