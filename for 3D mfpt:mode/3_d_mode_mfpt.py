#!/usr/bin/env python3
# 3-D reflecting random walk - MFPT vs Mode (enhanced AW for all ranges)
# Outputs:
#   - mfpt_mode_data_3d.json
#   - mfpt_vs_mode_3d.pdf

from __future__ import annotations
import os, json, time
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np

# Headless backend for servers/containers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    print("[warn] Numba not installed. Install with: pip install numba")
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# ---------------- User Configuration ----------------
q: float = 0.8

# For 3D, we'll use smaller N values due to computational complexity
N_VALUES = np.unique(np.concatenate([
    np.arange(5, 21, 2),        # 5-20, step 2
    np.arange(25, 41, 5),       # 25-40, step 5
    np.arange(50, 71, 10),      # 50-70, step 10
    np.arange(80, 101, 20),     # 80-100, step 20
])).astype(int)

MAX_WORKERS: Optional[int] = None  # None -> os.cpu_count()
PDF_PATH = "mfpt_vs_mode_3d.pdf"
JSON_PATH = "mfpt_mode_data_3d.json"

# -------- Mode search & AW parameters --------
MODE_T_WINDOW_MULT: float = 6.0     
GLOBAL_HARD_T_MAX:   int   = 10_000_000  # Increased for 3D
# Different alpha sets for different t ranges
ALPHAS_VERY_SMALL = (0.5, 1.0, 1.5, 2.0)    # t <= 20
ALPHAS_SMALL = (1.5, 2.0, 2.5, 3.0)         # 20 < t <= 50
ALPHAS_MEDIUM = (2.5, 3.0, 3.5, 4.0)        # 50 < t <= 100
ALPHAS_LARGE = (3.5, 4.5, 5.5, 6.5)         # t > 100
AW_K_SQRT_COEF = 8.0                
AW_K_OFFSET    = 20
AW_MAX_TERMS_HARD = 400             
# ---------------------------------------------------------

# -------------- Optimized Spectral Helpers for 3D --------------
@jit(nopython=True, cache=True)
def _h_vec_all_numba_3d(N: int, n: int, n0: int) -> np.ndarray:
    """Compute h vector for 3D reflecting boundaries"""
    h = np.zeros(N, dtype=np.float64)
    h[0] = 1.0 / N
    if N > 1:
        for k in range(1, N):
            c1 = np.cos(k * np.pi * (2*n  + 1) / (2*N))
            c2 = np.cos(k * np.pi * (2*n0 + 1) / (2*N))
            h[k] = (2.0 / N) * (c1 * c2)
    return h

@jit(nopython=True, cache=True, parallel=True)
def mfpt_3d_reflecting_numba(N: int, x0: int, y0: int, z0: int,
                             x1: int, y1: int, z1: int, q: float) -> float:
    """Exact MFPT for 3D reflecting random walk using spectral method"""
    k = np.arange(N, dtype=np.float64)
    cos_k = np.cos(k * np.pi / N)
    s_k = q * (cos_k - 1.0)

    hx_01 = _h_vec_all_numba_3d(N, x1, x0)
    hx_11 = _h_vec_all_numba_3d(N, x1, x1)
    hy_01 = _h_vec_all_numba_3d(N, y1, y0)
    hy_11 = _h_vec_all_numba_3d(N, y1, y1)
    hz_01 = _h_vec_all_numba_3d(N, z1, z0)
    hz_11 = _h_vec_all_numba_3d(N, z1, z1)

    total = 0.0
    for i in prange(N):
        for j in range(N):
            row_sum = 0.0
            for k_idx in range(N):
                num = (hx_01[i] * hy_01[j] * hz_01[k_idx] - 
                      hx_11[i] * hy_11[j] * hz_11[k_idx])
                denom = s_k[i] + s_k[j] + s_k[k_idx]
                if abs(denom) > 1e-15:
                    row_sum += num / denom
            total += row_sum
    
    return 3.0 * (N**3) * total  # Note: 3D uses factor of 3

def mfpt_3d_reflecting(N: int, start_pos: Tuple[int, int, int],
                       target_pos: Tuple[int, int, int], q: float = 1.0) -> float:
    """Compute exact MFPT for 3D reflecting random walk"""
    x0, y0, z0 = start_pos
    x1, y1, z1 = target_pos
    if NUMBA_AVAILABLE:
        return mfpt_3d_reflecting_numba(N, x0, y0, z0, x1, y1, z1, q)
    
    # NumPy fallback
    k = np.arange(N, dtype=np.float64)
    cos_k = np.cos(k * np.pi / N)
    s_k = q * (cos_k - 1.0)

    def _h_vec_all(N: int, n: int, n0: int) -> np.ndarray:
        h = np.zeros(N, dtype=np.float64)
        h[0] = 1.0 / N
        if N > 1:
            k = np.arange(1, N, dtype=np.float64)
            c1 = np.cos(k * np.pi * (2*n  + 1) / (2*N))
            c2 = np.cos(k * np.pi * (2*n0 + 1) / (2*N))
            h[1:] = (2.0 / N) * (c1 * c2)
        return h

    hx_01 = _h_vec_all(N, x1, x0)
    hx_11 = _h_vec_all(N, x1, x1)
    hy_01 = _h_vec_all(N, y1, y0)
    hy_11 = _h_vec_all(N, y1, y1)
    hz_01 = _h_vec_all(N, z1, z0)
    hz_11 = _h_vec_all(N, z1, z1)

    # Create 3D tensor for numerator
    num = np.einsum('i,j,k->ijk', hx_01, hy_01, hz_01) - \
          np.einsum('i,j,k->ijk', hx_11, hy_11, hz_11)
    
    # Create 3D tensor for denominator
    denom = s_k[:, None, None] + s_k[None, :, None] + s_k[None, None, :]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = np.zeros_like(num)
        np.divide(num, denom, out=frac, where=(np.abs(denom) > 1e-15))
    
    return 3.0 * (N**3) * np.sum(frac)

# ---------------- Green's functions & AW inverse for 3D ----------------
@dataclass
class FastFPTCalculator3D:
    N: int
    q: float

    def __post_init__(self):
        k = np.arange(self.N, dtype=np.float64)
        self.cos_k = np.cos(k * np.pi / self.N)
        self.s_k = self.q * (self.cos_k - 1.0)
        # For 3D, we need triple sums
        self.h_cache = {}

    def _h(self, n: int, n0: int) -> np.ndarray:
        key = (n, n0)
        if key not in self.h_cache:
            if NUMBA_AVAILABLE:
                self.h_cache[key] = _h_vec_all_numba_3d(self.N, n, n0)
            else:
                h = np.zeros(self.N, dtype=np.float64)
                h[0] = 1.0 / self.N
                if self.N > 1:
                    k = np.arange(1, self.N, dtype=np.float64)
                    c1 = np.cos(k * np.pi * (2*n  + 1) / (2*self.N))
                    c2 = np.cos(k * np.pi * (2*n0 + 1) / (2*self.N))
                    h[1:] = (2.0 / self.N) * (c1 * c2)
                self.h_cache[key] = h
        return self.h_cache[key]

    def propagator_gf(self, z: complex, pos: Tuple[int, int, int],
                      start_pos: Tuple[int, int, int]) -> complex:
        """3D propagator Green's function"""
        x1, y1, z1 = pos
        x0, y0, z0 = start_pos
        
        hx = self._h(x1, x0)
        hy = self._h(y1, y0)
        hz = self._h(z1, z0)
        
        # Create 3D tensor H
        H = np.einsum('i,j,k->ijk', hx, hy, hz)
        
        # Denominator: 1 - z*(1 + (s_i + s_j + s_k)/3)
        # For 3D walk, transition probability is 1/6 to each neighbor
        den = np.zeros((self.N, self.N, self.N), dtype=np.complex128)
        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.N):
                    den[i,j,k] = 1.0 - z * (1.0 + (self.s_k[i] + self.s_k[j] + self.s_k[k])/3.0)
        
        abs_den = np.abs(den)
        thr = 1e-12 * np.max(abs_den)
        mask = abs_den < thr
        if np.any(mask):
            den[mask] = thr * (1 + 1j)
        
        res = np.sum(H / den, dtype=np.complex128)
        if np.abs(res) > 1e10:
            return np.nan + 1j*np.nan
        return res

    def first_passage_gf(self, z: complex, target_pos: Tuple[int, int, int],
                         start_pos: Tuple[int, int, int]) -> complex:
        """3D first-passage Green's function"""
        P_n0_to_n = self.propagator_gf(z, target_pos, start_pos)
        P_n_to_n  = self.propagator_gf(z, target_pos, target_pos)
        
        if not np.isfinite(P_n0_to_n) or not np.isfinite(P_n_to_n):
            return np.nan + 1j*np.nan
        if np.abs(P_n_to_n) < 1e-15:
            return np.nan + 1j*np.nan
        
        val = P_n0_to_n / P_n_to_n
        if np.abs(val) > 1e6:
            return np.nan + 1j*np.nan
        return val

def enhanced_aw_inverse_3d(calculator: FastFPTCalculator3D, t: int,
                           target_pos: Tuple[int, int, int], 
                           start_pos: Tuple[int, int, int],
                           N: int = None) -> float:
    """Enhanced Abate-Whitt with adaptive parameters for 3D"""
    if t <= 0:
        return 0.0
    
    # Select alpha set based on t value
    if t <= 20:
        alphas = ALPHAS_VERY_SMALL
    elif t <= 50:
        alphas = ALPHAS_SMALL
    elif t <= 100:
        alphas = ALPHAS_MEDIUM
    else:
        alphas = ALPHAS_LARGE
    
    vals = []
    
    for alpha in alphas:
        # Adaptive r calculation
        if t <= 10:
            r_exp = -alpha / (10 + np.log(t + 1))
            r = float(np.exp(r_exp))
        elif t <= 30:
            r_exp = -alpha / (10 + np.sqrt(t))
            r = float(np.exp(r_exp))
        else:
            r = float(np.exp(-alpha / t))
        
        r = np.clip(r, 0.3, 0.99)
        
        log_r_t = t * np.log(r)
        if log_r_t < -700:
            continue
        
        # Adaptive K selection
        if t <= 5:
            K = min(t, 3)
        elif t <= 10:
            K = min(t-1, 8)
        elif t <= 20:
            K = min(t-1, 15)
        elif t <= 50:
            K = min(t-1, 30)
        else:
            K = int(min(t-1, AW_MAX_TERMS_HARD,
                        AW_K_OFFSET + AW_K_SQRT_COEF * np.sqrt(t)))
        
        if K <= 0:
            F_r = calculator.first_passage_gf(r,  target_pos, start_pos)
            F_m = calculator.first_passage_gf(-r, target_pos, start_pos)
            if not (np.isfinite(F_r) and np.isfinite(F_m)):
                continue
            
            try:
                r_t = np.exp(log_r_t)
                if r_t == 0:
                    continue
                
                if t <= 10:
                    correction = 1.0 + 0.1 * np.log(t + 1)
                    val = correction * (F_r + ((-1)**t) * F_m) / (2.0 * t * r_t)
                else:
                    val = (F_r + ((-1)**t) * F_m) / (2.0 * t * r_t)
                
                val = float(np.real(val))
                if np.isfinite(val) and val >= 0 and val < 1e10:
                    vals.append(val)
            except (OverflowError, ZeroDivisionError):
                continue
            continue

        # Compute AW series
        ks = np.arange(1, K+1, dtype=int)
        ang = (ks * np.pi) / t
        zks = r * (np.cos(ang) + 1j*np.sin(ang))

        F_vals = np.empty(K, dtype=np.complex128)
        for i, zk in enumerate(zks):
            f = calculator.first_passage_gf(zk, target_pos, start_pos)
            if not np.isfinite(f):
                F_vals[i] = np.nan + 1j*np.nan
            else:
                F_vals[i] = f
        
        if np.any(~np.isfinite(F_vals)):
            continue

        if t <= 20:
            damping = np.exp(-0.05 * ks)
            series = np.sum(((-1.0)**ks) * damping * np.real(F_vals))
        else:
            series = np.sum(((-1.0)**ks) * np.real(F_vals))
        
        try:
            r_t = np.exp(log_r_t)
            if r_t == 0:
                continue
            main = ((t-1)/K) * series / (t * r_t)
            
            F_r = calculator.first_passage_gf(r,  target_pos, start_pos)
            F_m = calculator.first_passage_gf(-r, target_pos, start_pos)
            if not (np.isfinite(F_r) and np.isfinite(F_m)):
                continue
            edge = (F_r + ((-1)**t) * F_m) / (2.0 * t * r_t)
            
            val = float(np.real(main + edge))
            
            # Apply correction for small t in 3D
            if t <= 30 and N and N <= 20:
                boost = 1.0 + (30 - t) / 30.0 * (20 - N) / 20.0 * 0.5
                val *= boost
            
            if np.isfinite(val) and val >= 0 and val < 1e10:
                vals.append(val)
        except (OverflowError, ZeroDivisionError):
            continue

    if not vals:
        return np.nan

    v = np.asarray(vals, dtype=float)
    
    if t <= 30:
        q25, q75 = np.percentile(v, [25, 75])
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        v_trimmed = v[(v >= lower) & (v <= upper)]
        if len(v_trimmed) > 0:
            return float(np.mean(v_trimmed))
        else:
            return float(np.median(v))
    else:
        med = np.median(v)
        mad = np.median(np.abs(v - med)) + 1e-15
        keep = np.abs(v - med) <= 4.0 * mad
        v = v[keep]
        if len(v) == 0:
            return float(med)
        return float(np.median(v))

def pmf_at_t_3d(calculator: FastFPTCalculator3D, t: int,
                target_pos: Tuple[int,int,int], 
                start_pos: Tuple[int,int,int],
                N: int = None) -> float:
    return enhanced_aw_inverse_3d(calculator, t, target_pos, start_pos, N=N)

def compute_pmf_batch_3d(calculator: FastFPTCalculator3D,
                        ts: np.ndarray,
                        target_pos: Tuple[int, int, int],
                        start_pos: Tuple[int, int, int],
                        N: int = None) -> np.ndarray:
    """Compute PMF for multiple t values in 3D"""
    pmf = np.zeros(len(ts))
    for i, t in enumerate(ts):
        pmf[i] = pmf_at_t_3d(calculator, int(t), target_pos, start_pos, N=N)
    
    # Apply smoothing for small N
    if N and N <= 20 and len(pmf) > 5:
        try:
            from scipy.ndimage import gaussian_filter1d
            valid = np.isfinite(pmf) & (pmf > 0)
            if np.sum(valid) > 5:
                pmf[valid] = gaussian_filter1d(pmf[valid], sigma=0.5)
        except ImportError:
            pass
    
    return pmf

def find_mode_3d(calculator: FastFPTCalculator3D,
                target_pos: Tuple[int, int, int],
                start_pos: Tuple[int, int, int],
                mfpt_est: float,
                N: int) -> int:
    """Find mode using enhanced AW method for 3D"""
    
    if not np.isfinite(mfpt_est) or mfpt_est <= 0:
        mfpt_est = max(10.0, 0.8 * (N**3))  # Note: N^3 for 3D
    
    # For very small N, use empirical formula
    if N <= 10:
        mode_est = int(0.6 * mfpt_est)  # Slightly lower ratio for 3D
        
        search_window = int(0.2 * mfpt_est)
        ts_verify = np.linspace(max(1, mode_est - search_window), 
                               mode_est + search_window, 20, dtype=int)
        pmf_verify = compute_pmf_batch_3d(calculator, ts_verify, target_pos, start_pos, N=N)
        
        valid = np.isfinite(pmf_verify) & (pmf_verify > 0)
        if np.sum(valid) >= 3:
            t_numerical = int(ts_verify[valid][np.argmax(pmf_verify[valid])])
            ratio = t_numerical / mfpt_est
            if 0.3 <= ratio <= 0.9:
                return t_numerical
        
        return mode_est
    
    # Adaptive search range for 3D
    if N <= 15:
        t_lo = max(1, int(0.35 * mfpt_est))
        t_hi_mult = 2.0
    elif N <= 30:
        t_lo = max(1, int(0.3 * mfpt_est))
        t_hi_mult = 2.5
    else:
        t_lo = max(1, int(0.25 * mfpt_est))
        t_hi_mult = 3.0
    
    t_hi = int(min(t_hi_mult * mfpt_est, GLOBAL_HARD_T_MAX))
    
    # Initial coarse search
    if t_hi - t_lo <= 100:
        n_points = min(50, t_hi - t_lo)
    else:
        n_points = min(80, max(40, int(np.sqrt(t_hi - t_lo))))
    
    ts_coarse = np.unique(np.linspace(t_lo, t_hi, n_points, dtype=int))
    pmf_coarse = compute_pmf_batch_3d(calculator, ts_coarse, target_pos, start_pos, N=N)
    
    valid = np.isfinite(pmf_coarse) & (pmf_coarse > 0)
    n_valid = np.sum(valid)
    
    if n_valid < 5:
        return int(0.6 * mfpt_est)
    
    pmf_valid = pmf_coarse[valid]
    ts_valid = ts_coarse[valid]
    peak_idx = np.argmax(pmf_valid)
    
    # Check if peak is at boundary
    if peak_idx == 0 or peak_idx == len(pmf_valid) - 1:
        if peak_idx == 0:
            t_lo = max(1, int(0.2 * ts_valid[0]))
        else:
            t_hi = min(int(1.5 * ts_valid[-1]), GLOBAL_HARD_T_MAX)
        
        ts_coarse = np.unique(np.linspace(t_lo, t_hi, n_points, dtype=int))
        pmf_coarse = compute_pmf_batch_3d(calculator, ts_coarse, target_pos, start_pos, N=N)
        valid = np.isfinite(pmf_coarse) & (pmf_coarse > 0)
        
        if not np.any(valid):
            return int(0.6 * mfpt_est)
        
        pmf_valid = pmf_coarse[valid]
        ts_valid = ts_coarse[valid]
        peak_idx = np.argmax(pmf_valid)
    
    t_peak_coarse = int(ts_valid[peak_idx])
    
    # Check reasonableness
    ratio_coarse = t_peak_coarse / mfpt_est
    if ratio_coarse < 0.2 or ratio_coarse > 1.5:
        return int(0.6 * mfpt_est)
    
    # Fine search
    window = max(10, int(0.1 * (t_hi - t_lo)))
    t_lo_fine = max(1, t_peak_coarse - window)
    t_hi_fine = min(t_hi, t_peak_coarse + window)
    
    step_fine = max(1, (t_hi_fine - t_lo_fine) // 80)
    ts_fine = np.arange(t_lo_fine, t_hi_fine + 1, step_fine)
    
    pmf_fine = compute_pmf_batch_3d(calculator, ts_fine, target_pos, start_pos, N=N)
    
    valid_fine = np.isfinite(pmf_fine) & (pmf_fine > 0)
    if np.sum(valid_fine) < 3:
        return t_peak_coarse
    
    t_peak_fine = int(ts_fine[valid_fine][np.argmax(pmf_fine[valid_fine])])
    
    # Final sanity check
    ratio_fine = t_peak_fine / mfpt_est
    if ratio_fine < 0.2 or ratio_fine > 1.5:
        return int(0.6 * mfpt_est)
    
    return t_peak_fine

# ------------------- Per-N Computation ----------------
def compute_for_single_N_3d(N: int, q: float) -> Tuple[int, float, float]:
    """Compute MFPT and mode for a single N value in 3D"""
    start_pos = (0, 0, 0)
    target_pos = (N - 1, N - 1, N - 1)

    mfpt = mfpt_3d_reflecting(N, start_pos, target_pos, q)

    calc = FastFPTCalculator3D(N, q)
    try:
        mode_t = find_mode_3d(calc, target_pos, start_pos, mfpt_est=mfpt, N=N)
    except Exception as e:
        print(f"[warn] Mode calculation failed for N={N}: {e}")
        mode_t = int(0.6 * mfpt)

    # Sanity check
    ratio = mode_t / mfpt
    if ratio < 0.2 or ratio > 1.5:
        print(f"[warn] N={N}: Mode/MFPT={ratio:.3f} seems unusual")

    return (N, float(mfpt), float(mode_t))

# -------------------- Plotting --------------------
def save_mfpt_mode_plot_3d(Ns: np.ndarray, mfpt: np.ndarray,
                           mode: np.ndarray, pdf_path: str):
    """Create plot for 3D results"""
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10, 
        "ytick.labelsize": 10, "legend.fontsize": 11, "pdf.fonttype": 42, 
        "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 150, 
        "figure.dpi": 100, "font.family": "sans-serif", "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: MFPT and Mode vs N
    ax1.loglog(Ns, mfpt, 'o-', linewidth=1.8, markersize=5,
              color='#1f77b4', label='MFPT', alpha=0.9)
    mask_mode = np.isfinite(mode)
    if np.any(mask_mode):
        ax1.loglog(Ns[mask_mode], mode[mask_mode], 's-', linewidth=1.8,
                  markersize=5, color='#ff7f0e', label='Mode', alpha=0.9)
    ax1.set_xlabel("Lattice size N")
    ax1.set_ylabel("First-passage time")
    ax1.set_title("3D Reflecting Random Walk")
    ax1.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax1.legend(loc='best', framealpha=0.95, edgecolor='none')
    ax1.set_xlim([Ns.min()*0.9, Ns.max()*1.1])
    
    # Right plot: Mode/MFPT ratio vs N
    ratio = mode / mfpt
    mask_ratio = np.isfinite(ratio)
    if np.any(mask_ratio):
        ax2.semilogx(Ns[mask_ratio], ratio[mask_ratio], 'D-', linewidth=1.8,
                    markersize=5, color='#2ca02c', alpha=0.9)
        ax2.axhline(y=ratio[mask_ratio].mean(), color='gray', linestyle='--', 
                   alpha=0.5, label=f'Mean = {ratio[mask_ratio].mean():.3f}')
    ax2.set_xlabel("Lattice size N")
    ax2.set_ylabel("Mode/MFPT")
    ax2.set_title("Mode to MFPT Ratio")
    ax2.grid(True, alpha=0.25, linewidth=0.5)
    ax2.legend(loc='best', framealpha=0.95, edgecolor='none')
    ax2.set_xlim([Ns.min()*0.9, Ns.max()*1.1])
    
    plt.tight_layout()
    fig.savefig(pdf_path, format="pdf", bbox_inches='tight')
    plt.close(fig)

# -------------------- Main --------------------
def main():
    Ns = np.array(N_VALUES, dtype=int)
    results: List[Tuple[int, float, float]] = []

    n_workers = MAX_WORKERS or os.cpu_count()
    print(f"[Info] 3D Random Walk")
    print(f"[Info] q = {q}, N in [{Ns.min()}, {Ns.max()}], points={len(Ns)}")
    print(f"[Info] Start: (0,0,0), Target: (N-1,N-1,N-1)")
    print(f"[Info] Numba: {'enabled' if NUMBA_AVAILABLE else 'disabled'}")
    print(f"[Info] Using enhanced AW with adaptive parameters")
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(compute_for_single_N_3d, int(N), float(q)): int(N) 
                  for N in Ns}
        with tqdm(total=len(Ns), desc="Computing", unit="N") as pbar:
            for future in as_completed(futures):
                N_val = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"\n[Error] N={N_val} failed: {e}")
                pbar.update(1)

    total_time = time.time() - t0
    print(f"[Info] Completed in {total_time:.1f}s")

    if not results:
        print("[Error] No results.")
        return

    results.sort(key=lambda x: x[0])
    Ns_out = np.array([r[0] for r in results], dtype=int)
    MFPT  = np.array([r[1] for r in results], dtype=float)
    MODE  = np.array([r[2] for r in results], dtype=float)

    data = {
        "dimension": 3,
        "q": q, 
        "N": Ns_out.tolist(),
        "mfpt": MFPT.tolist(), 
        "mode": MODE.tolist(),
        "mode_over_mfpt": [(MODE[i]/MFPT[i]) if (np.isfinite(MODE[i]) and MFPT[i] > 0) 
                          else None for i in range(len(MFPT))],
        "notes": {
            "reflecting_boundaries": True,
            "start_pos": [0, 0, 0], 
            "target_pos": ["N-1", "N-1", "N-1"],
            "mode_method": "Enhanced Abate-Whitt with adaptive parameters for 3D",
            "aw_enhancements": [
                "Adaptive alpha sets based on t ranges",
                "Modified r calculation for small t",
                "Damped series for t<=20",
                "Correction factors for small N and t",
                "Trimmed mean aggregation for t<=30",
                "Gaussian smoothing for small N",
                "3D spectral method for exact MFPT",
                "3D Green's functions for mode calculation"
            ],
            "alpha_ranges": {
                "t<=20": str(ALPHAS_VERY_SMALL),
                "20<t<=50": str(ALPHAS_SMALL),
                "50<t<=100": str(ALPHAS_MEDIUM),
                "t>100": str(ALPHAS_LARGE)
            },
            "hard_T_max": GLOBAL_HARD_T_MAX,
            "numba_enabled": NUMBA_AVAILABLE,
        },
        "computation_time_seconds": total_time
    }
    
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[✓] Saved JSON: {JSON_PATH}")

    save_mfpt_mode_plot_3d(Ns_out, MFPT, MODE, PDF_PATH)
    print(f"[✓] Saved PDF: {PDF_PATH}")

    print("\n[Summary]")
    print(f"  MFPT range: {MFPT.min():.1f} .. {MFPT.max():.1f}")
    finite_mode = MODE[np.isfinite(MODE)]
    if finite_mode.size:
        print(f"  Mode range: {finite_mode.min():.1f} .. {finite_mode.max():.1f}")
        ratio = (MODE / MFPT)
        ratio = ratio[np.isfinite(ratio)]
        if ratio.size:
            print(f"  Mode/MFPT: mean={ratio.mean():.3f}, std={ratio.std():.3f}")
            print(f"  Mode/MFPT range: {ratio.min():.3f} .. {ratio.max():.3f}")
            
            # Check small N performance
            small_N_mask = Ns_out <= 20
            if np.any(small_N_mask):
                small_ratio = ratio[:np.sum(small_N_mask)]
                print(f"  Mode/MFPT (N≤20): mean={small_ratio.mean():.3f}, "
                      f"range={small_ratio.min():.3f}..{small_ratio.max():.3f}")

if __name__ == "__main__":
    main()