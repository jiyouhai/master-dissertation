#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import mpmath as mp

# --------- precision ---------
mp.dps = 50

# --------- parameters ---------
q        = 1.0
N_min    = 20
N_max    = 1000
step_N   = 20

# --------- analytic constants ---------
C0 = mp.mpf('-1.771368057')   # full constant
C1 = -mp.pi / 4
C2 = mp.pi**2 / 48

# --------- exact summand ---------
def f_term_mp(k: int, N: int) -> mp.mpf:
    theta  = mp.pi * k / N
    sigma  = 2 - mp.cos(theta)
    varphi = mp.acosh(sigma)
    Ak     = mp.cos(theta/2) ** 2
    num = (mp.cosh(N*varphi)
           + mp.cosh((N-1)*varphi)
           - (-1)**k * (1 + mp.cosh(varphi)))
    den = mp.sinh(varphi) * mp.sinh(N*varphi)
    return Ak * num / den

def mfpt_exact(N: int, q: float = 1.0) -> float:
    s = mp.mpf('0')
    for k in range(1, N):
        s += f_term_mp(k, N)
    return float(2*N*(N-1)/q + (4*N/q)*s)

# --------- asymptotic (with C0) ---------
def mfpt_approx(N: int, q: float = 1.0) -> float:
    N_mp = mp.mpf(N)
    return float((8/(mp.pi*q))*N_mp**2*mp.log(N_mp-1)
                 + ((2 + C0)/q)*N_mp**2
                 + (C1/q)*N_mp*mp.log(N_mp)
                 + (C2/q)*N_mp)

# --------- batch compute ---------
Ns = np.arange(N_min, N_max + 1, step_N, dtype=int)
exact_arr  = np.array([mfpt_exact(int(N), q) for N in Ns], dtype=float)
approx_arr = np.array([mfpt_approx(int(N), q) for N in Ns], dtype=float)
abs_err = exact_arr - approx_arr
pct_err = abs_err / exact_arr * 100.0

# --------- export JSON ---------
out_dir = Path.cwd()
json_path = out_dir / "mfpt_data.json"
payload = {
    "params": {
        "q": q,
        "N_min": int(N_min),
        "N_max": int(N_max),
        "step_N": int(step_N),
        "mp_dps": int(mp.dps),
    },
    "constants": {
        "C0": float(C0),
        "C1": float(C1),
        "C2": float(C2),
    },
    "data": {
        "N": Ns.tolist(),
        "exact": exact_arr.tolist(),
        "approx": approx_arr.tolist(),
        "abs_err": abs_err.tolist(),
        "pct_err": pct_err.tolist(),
    },
    "summary": {
        "max_abs_err": float(np.max(np.abs(abs_err))),
        "max_pct_err_percent": float(np.max(np.abs(pct_err))),
    },
}
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print(f"[OK] JSON saved -> {json_path}")

# --------- plotting style (bigger, print-ready) ---------
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(
    1, 2, figsize=(14, 5),
    gridspec_kw={"width_ratios": [1.25, 1.0]},
    constrained_layout=True
)

# --------- left: exact vs asymptotic (high contrast) ---------
ax = axes[0]

# High-contrast styling:
# - Exact: thick black line with white outline + hollow circle markers
# - Asymptotic: thick dashed colored line + filled square markers
# - Semi-transparent band highlighting the gap between them
exact_line, = ax.plot(
    Ns, exact_arr, linestyle='-', linewidth=2.6, marker='o', markersize=5,
    markerfacecolor='none', markeredgewidth=1.6, color='black', zorder=3,
)
exact_line.set_path_effects([pe.Stroke(linewidth=4.0, foreground='white'), pe.Normal()])

approx_line, = ax.plot(
    Ns, approx_arr, linestyle='--', linewidth=2.6, marker='s', markersize=5,
    color='#d62728',  # distinct, colorblind-friendly red
    zorder=2,
)

# Band to emphasize difference (does not imply uncertainty)
ax.fill_between(Ns, exact_arr, approx_arr, alpha=0.18, color='#ff9896', zorder=1)

ax.set_xlabel('N')
ax.set_ylabel('MFPT')
ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
ax.legend(
    [exact_line, approx_line],
    ['Exact (sum)', 'Asymptotic (with C0)'],
    frameon=False, loc='best'
)

# --------- zoom inset around the largest absolute error ---------
i_star = int(np.argmax(np.abs(abs_err)))
i_left  = max(0, i_star - 3)
i_right = min(len(Ns) - 1, i_star + 3)
x1, x2 = Ns[i_left], Ns[i_right]
y_min = min(np.min(exact_arr[i_left:i_right+1]), np.min(approx_arr[i_left:i_right+1]))
y_max = max(np.max(exact_arr[i_left:i_right+1]), np.max(approx_arr[i_left:i_right+1]))
y_pad = 0.06 * (y_max - y_min + 1e-9)

axins = zoomed_inset_axes(ax, zoom=2.2, loc='lower right', borderpad=1.0)
axins.plot(Ns, exact_arr, linestyle='-', linewidth=2.2, marker='o', markersize=4,
           markerfacecolor='none', markeredgewidth=1.2, color='black', zorder=3)
axins.plot(Ns, approx_arr, linestyle='--', linewidth=2.2, marker='s', markersize=4,
           color='#d62728', zorder=2)
axins.fill_between(Ns, exact_arr, approx_arr, alpha=0.18, color='#ff9896', zorder=1)
axins.set_xlim(x1, x2)
axins.set_ylim(y_min - y_pad, y_max + y_pad)
axins.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
# Neat connectors (no extra caption)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.3", lw=0.8)

# --------- right: errors (abs + percent on twin y-axis) ---------
ax2 = axes[1]
l_abs, = ax2.plot(Ns, abs_err, marker='o', linestyle='-', linewidth=2.0, markersize=4,
                  color='tab:blue', label='Absolute error', zorder=2)
ax2.set_xlabel('N')
ax2.set_ylabel('Error')
ax2.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)

ax2b = ax2.twinx()
l_pct, = ax2b.plot(Ns, pct_err, marker='s', linestyle='--', linewidth=2.0, markersize=4,
                   color='tab:green', label='Percentage error (%)', zorder=2)
ax2b.set_ylabel('Percentage error (%)')

# Unified legend on the right subplot
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc='best')

# --------- save PDF ---------
pdf_path = out_dir / "mfpt_exact_vs_approx.pdf"
fig.savefig(pdf_path, format='pdf')
plt.close(fig)
print(f"[OK] PDF saved  -> {pdf_path}")

# --------- console summary ---------
print(f"{'N':>6}  {'Exact':>14} {'Approx':>14} {'AbsErr':>14} {'PctErr(%)':>12}")
print("-"*68)
for N, ex, ap, ae, pe in zip(Ns, exact_arr, approx_arr, abs_err, pct_err):
    print(f"{N:6d} {ex:14.6f} {ap:14.6f} {ae:14.6f} {pe:12.6f}")
print("-"*68)
print(f"Max |AbsErr|   : {np.max(np.abs(abs_err)):.6e}")
print(f"Max |PctErr| % : {np.max(np.abs(pct_err)):.6f}")
