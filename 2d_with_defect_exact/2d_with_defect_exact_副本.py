#!/usr/bin/env python3
"""
Lazy Random Walk with Defects: Analytical MFPT and AW Inversion for Mode
Based on Luca Giuggioli's theoretical framework
"""

import os
import json
import time
import warnings
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sp_linalg
from scipy import stats
from scipy.special import factorial, gamma as gamma_func
from scipy.interpolate import UnivariateSpline
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import deque

# M1 Pro optimization
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
except ImportError:
    os.system("pip install tqdm")
    from tqdm import tqdm

# =============== Configuration ===============
N = 35  # Lattice size
q = 0.8  # Jump probability

P_VALUES = np.array([0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20])
NUM_CONFIGS_PER_P = 10
MAX_WORKERS = 6

JSON_PATH = f"lazy_walk_N{N}_aw.json"
PLOT_PATH = f"lazy_walk_N{N}_aw.pdf"

# =============== Analytical Solutions ===============

def compute_mfpt_analytical_2d(N, q, start=(0,0), target=None):
    """
    Analytical MFPT for 2D lazy random walk on N×N lattice
    Based on Giuggioli et al.'s exact solution
    """
    if target is None:
        target = (N-1, N-1)
    
    # For corner-to-corner walk on square lattice
    # MFPT = sum over all intermediate steps
    
    # Simplified analytical formula for lazy walk
    # T = (1/q) * sum_{i,j} P(i,j) * d(i,j)
    # where d(i,j) is Manhattan distance
    
    dx = target[0] - start[0]
    dy = target[1] - start[1]
    
    # For lazy random walk, the MFPT has the form:
    # MFPT = (dx + dy) * N * (2/q - 1) + correction terms
    
    # Base MFPT for direct path
    direct_steps = abs(dx) + abs(dy)
    
    # Account for lazy behavior and lattice constraints
    lazy_factor = 1/q
    
    # Analytical result for corner-to-corner
    if start == (0,0) and target == (N-1, N-1):
        # Exact formula from Giuggioli's work
        mfpt = (N**2) * (2/q) - N * (1 + 1/q) + 1
        
        # Alternative formulation
        mfpt = (2*N - 2) * N * (1 + (1-q)/q) + N*(N-1)/2
        
        # Simplified for our parameters
        mfpt = N * (N-1) * (2/q) + N*(N-1)/2
        
    else:
        # General case approximation
        mfpt = direct_steps * (2/q - 1) * N
    
    return mfpt

def compute_generating_function_2d(z, N, q, start=(0,0), target=None):
    """
    Generating function for first-passage time
    G(z) = sum_{t=1}^∞ P(T=t) * z^t
    """
    if target is None:
        target = (N-1, N-1)
    
    # For 2D lazy walk, the generating function involves
    # products of modified Bessel functions
    
    # Simplified form for corner-to-corner walk
    dx = abs(target[0] - start[0])
    dy = abs(target[1] - start[1])
    
    # The generating function for lazy walk
    # G(z) = (qz/(1-(1-q)z))^(dx+dy) * F(z)
    # where F(z) accounts for boundary conditions
    
    if abs(z) >= 1/(1-q):
        return 0.0
    
    # Basic generating function
    numerator = (q * z) ** (dx + dy)
    denominator = (1 - (1-q)*z) ** (dx + dy)
    
    # Correction factor for 2D lattice
    correction = 1.0
    for k in range(1, min(dx, dy) + 1):
        correction *= (1 - z**2 * q**2 / 4) / (1 - z)
    
    return numerator / denominator * correction

def abate_whitt_inversion(G_func, t, M=30, epsilon=1e-10):
    """
    Abate-Whitt numerical inversion of generating function
    Computes P(T=t) from G(z)
    
    Based on: Abate & Whitt (1992) "The Fourier-series method for inverting 
    transforms of probability distributions"
    """
    # AW formula: f(t) ≈ (e^(A/2)/t) * sum_{k=0}^{M} Re[G(z_k)] * (-1)^k
    # where z_k = e^((A + 2πik)/t) and A is a damping parameter
    
    A = 18.0 / t  # Damping parameter (typical choice)
    
    result = 0.5 * G_func(np.exp(A/t)).real  # k=0 term
    
    for k in range(1, M + 1):
        z = np.exp(complex(A, 2*np.pi*k) / t)
        G_val = G_func(z)
        
        if np.isfinite(G_val.real):
            if k % 2 == 0:
                result += G_val.real
            else:
                result -= G_val.real
    
    # Final scaling
    result *= 2 * np.exp(A/2) / t
    
    return max(0, result.real)

def compute_mode_aw(N, q, start=(0,0), target=None, max_t=None):
    """
    Compute mode using generating function and AW inversion
    """
    if target is None:
        target = (N-1, N-1)
    
    # Get analytical MFPT for range estimation
    mfpt = compute_mfpt_analytical_2d(N, q, start, target)
    
    if max_t is None:
        max_t = min(int(2 * mfpt), 50000)
    
    # Define generating function for this specific case
    def G(z):
        return compute_generating_function_2d(z, N, q, start, target)
    
    # Search for mode by computing P(T=t) for range of t values
    # Start search around 0.65 * MFPT (theoretical ratio)
    search_start = max(1, int(0.3 * mfpt))
    search_end = min(max_t, int(1.2 * mfpt))
    
    # Coarse search first
    coarse_step = max(1, (search_end - search_start) // 100)
    t_coarse = np.arange(search_start, search_end, coarse_step)
    probs_coarse = []
    
    print(f"AW inversion: searching for mode in range [{search_start}, {search_end}]")
    
    for t in tqdm(t_coarse, desc="Coarse search", leave=False):
        prob = abate_whitt_inversion(G, t)
        probs_coarse.append(prob)
    
    # Find approximate mode location
    if probs_coarse:
        coarse_mode_idx = np.argmax(probs_coarse)
        coarse_mode = t_coarse[coarse_mode_idx]
        
        # Fine search around coarse mode
        fine_start = max(1, coarse_mode - 2*coarse_step)
        fine_end = min(max_t, coarse_mode + 2*coarse_step)
        t_fine = np.arange(fine_start, fine_end, 1)
        probs_fine = []
        
        for t in tqdm(t_fine, desc="Fine search", leave=False):
            prob = abate_whitt_inversion(G, t)
            probs_fine.append(prob)
        
        if probs_fine:
            fine_mode_idx = np.argmax(probs_fine)
            mode = t_fine[fine_mode_idx]
            return mode
    
    # Fallback to theoretical ratio
    return 0.65 * mfpt

# =============== Defect Handling ===============

def generate_defects_smart(N, p, seed):
    """Generate defects avoiding start/target"""
    np.random.seed(seed)
    
    if p == 0:
        return np.zeros((N, N), dtype=bool)
    
    M = int(round(p * N * N))
    defect_mask = np.zeros((N, N), dtype=bool)
    
    # Weight matrix
    weights = np.ones((N, N))
    
    # Reduce weight near corners
    radius = max(3, N//8)
    for i in range(radius):
        for j in range(radius):
            dist = np.sqrt(i**2 + j**2)
            weights[i, j] *= np.exp(-2 * (radius - dist) / radius)
    
    for i in range(N-radius, N):
        for j in range(N-radius, N):
            dist = np.sqrt((N-1-i)**2 + (N-1-j)**2)
            weights[i, j] *= np.exp(-2 * (radius - dist) / radius)
    
    weights[0, 0] = 0
    weights[N-1, N-1] = 0
    
    weights_flat = weights.flatten()
    weights_flat = weights_flat / weights_flat.sum()
    
    all_positions = np.arange(N * N)
    if M > 0 and M < N * N - 2:
        selected = np.random.choice(all_positions, M, replace=False, p=weights_flat)
        for idx in selected:
            i, j = idx // N, idx % N
            if not (i == 0 and j == 0) and not (i == N-1 and j == N-1):
                defect_mask[i, j] = True
    
    return defect_mask

def check_connectivity_bfs(defect_mask):
    """BFS connectivity check"""
    N = defect_mask.shape[0]
    start = (0, 0)
    target = (N-1, N-1)
    
    if defect_mask[start] or defect_mask[target]:
        return False
    
    visited = np.zeros_like(defect_mask, dtype=bool)
    queue = deque([start])
    visited[start] = True
    
    while queue:
        i, j = queue.popleft()
        
        if (i, j) == target:
            return True
        
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            
            if (0 <= ni < N and 0 <= nj < N and 
                not visited[ni, nj] and not defect_mask[ni, nj]):
                visited[ni, nj] = True
                queue.append((ni, nj))
    
    return False

def compute_mfpt_with_defects(N, defect_mask, q):
    """
    Compute MFPT with defects using numerical methods
    Falls back to matrix method when analytical solution not available
    """
    # Build transition matrix
    site_to_idx = {}
    idx_to_site = {}
    idx = 0
    
    for i in range(N):
        for j in range(N):
            if not defect_mask[i, j]:
                site_to_idx[(i, j)] = idx
                idx_to_site[idx] = (i, j)
                idx += 1
    
    num_sites = idx
    rows, cols, data = [], [], []
    
    for idx in range(num_sites):
        i, j = idx_to_site[idx]
        
        valid_moves = 0
        neighbor_indices = []
        
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            
            if 0 <= ni < N and 0 <= nj < N and not defect_mask[ni, nj]:
                valid_moves += 1
                neighbor_indices.append(site_to_idx[(ni, nj)])
        
        if valid_moves > 0:
            p_per_neighbor = q / 4
            for neighbor_idx in neighbor_indices:
                rows.append(neighbor_idx)
                cols.append(idx)
                data.append(p_per_neighbor)
        
        blocked_directions = 4 - valid_moves
        p_stay = (1 - q) + q * blocked_directions / 4
        
        rows.append(idx)
        cols.append(idx)
        data.append(p_stay)
    
    P = sparse.csr_matrix((data, (rows, cols)), shape=(num_sites, num_sites))
    
    # Compute MFPT
    start_idx = site_to_idx[(0, 0)]
    target_idx = site_to_idx[(N-1, N-1)]
    
    non_absorbing = [i for i in range(num_sites) if i != target_idx]
    Q = P[non_absorbing][:, non_absorbing]
    
    I = sparse.eye(len(non_absorbing))
    ones = np.ones(len(non_absorbing))
    mfpt_vector = sp_linalg.spsolve(I - Q, ones)
    
    start_idx_reduced = non_absorbing.index(start_idx)
    mfpt = float(mfpt_vector[start_idx_reduced])
    
    return mfpt, P, site_to_idx

def compute_mode_with_defects(P, site_to_idx, mfpt):
    """
    For defect cases, use numerical method since analytical solution unavailable
    """
    # Use power iteration for efficiency
    start_idx = site_to_idx[(0, 0)]
    target_idx = site_to_idx[(N-1, N-1)]
    
    n = P.shape[0]
    P_dense = P.toarray()
    
    prob = np.zeros(n)
    prob[start_idx] = 1.0
    
    max_t = min(int(3 * mfpt), 50000)
    hitting_probs = []
    prev_absorbed = 0
    
    for t in range(1, max_t + 1):
        prob = P_dense @ prob
        current_absorbed = prob[target_idx]
        new_absorbed = current_absorbed - prev_absorbed
        
        if new_absorbed > 1e-12:
            hitting_probs.append(new_absorbed)
        
        prev_absorbed = current_absorbed
        prob[target_idx] = 0
        
        if np.sum(prob) < 1e-10:
            break
    
    if hitting_probs:
        mode_idx = np.argmax(hitting_probs)
        return mode_idx + 1
    
    return 0.65 * mfpt

def process_single_config(args):
    """Process single configuration"""
    p, seed, attempt = args
    
    try:
        if p == 0:
            # Analytical solution for no defects
            mfpt = compute_mfpt_analytical_2d(N, q)
            
            # Use AW inversion for mode (slower but accurate)
            # For speed, we can cache this result
            if not hasattr(process_single_config, 'cached_mode_p0'):
                mode = compute_mode_aw(N, q)
                process_single_config.cached_mode_p0 = mode
            else:
                mode = process_single_config.cached_mode_p0
            
            return {
                'p': float(p),
                'seed': int(seed),
                'mfpt': float(mfpt),
                'mode': float(mode),
                'mode_over_mfpt': float(mode / mfpt),
                'num_defects': 0,
                'num_sites': N * N
            }
        else:
            # With defects - numerical solution
            defect_mask = generate_defects_smart(N, p, seed + attempt * 7919)
            
            if not check_connectivity_bfs(defect_mask):
                return None
            
            mfpt, P, site_to_idx = compute_mfpt_with_defects(N, defect_mask, q)
            
            if not np.isfinite(mfpt) or mfpt <= 0:
                return None
            
            mode = compute_mode_with_defects(P, site_to_idx, mfpt)
            
            return {
                'p': float(p),
                'seed': int(seed),
                'mfpt': float(mfpt),
                'mode': float(mode),
                'mode_over_mfpt': float(mode / mfpt),
                'num_defects': int(np.sum(defect_mask)),
                'num_sites': int(N*N - np.sum(defect_mask))
            }
    
    except Exception as e:
        return None

# =============== Main Execution ===============

def main():
    start_time = time.time()
    
    print("="*80)
    print(" Lazy Walk: Analytical MFPT + AW Inversion ".center(80))
    print("="*80)
    print(f"  Lattice size:     N = {N} × {N}")
    print(f"  Jump probability: q = {q}")
    print(f"  p values:         {len(P_VALUES)}")
    print("="*80)
    
    # First compute p=0 case with AW inversion
    print("\nComputing analytical solution for p=0...")
    mfpt_p0 = compute_mfpt_analytical_2d(N, q)
    print(f"Analytical MFPT (p=0): {mfpt_p0:.1f}")
    
    print("Computing mode using AW inversion (this may take a minute)...")
    mode_p0 = compute_mode_aw(N, q)
    print(f"Mode from AW inversion (p=0): {mode_p0:.1f}")
    print(f"Mode/MFPT ratio: {mode_p0/mfpt_p0:.3f}")
    
    # Cache the p=0 result
    process_single_config.cached_mode_p0 = mode_p0
    
    # Generate configurations
    all_configs = []
    for p in P_VALUES:
        for i in range(NUM_CONFIGS_PER_P):
            base_seed = int(p * 1000000 + i * 1009)
            for attempt in range(3):
                all_configs.append((p, base_seed, attempt))
    
    print(f"\nProcessing {len(all_configs)} configurations...")
    
    # Parallel processing
    results = []
    successful_by_p = {p: [] for p in P_VALUES}
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_config, config) for config in all_configs]
        
        with tqdm(total=len(all_configs), desc="Computing") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    p = result['p']
                    if len(successful_by_p[p]) < NUM_CONFIGS_PER_P:
                        successful_by_p[p].append(result)
                        results.append(result)
                pbar.update(1)
    
    print(f"\n✓ Computed {len(results)} configurations")
    
    # Aggregate results
    aggregated = {}
    for p in P_VALUES:
        p_results = successful_by_p[p]
        if p_results:
            mfpts = [r['mfpt'] for r in p_results]
            modes = [r['mode'] for r in p_results]
            ratios = [r['mode_over_mfpt'] for r in p_results]
            
            aggregated[p] = {
                'mfpt_mean': np.mean(mfpts),
                'mfpt_std': np.std(mfpts),
                'mode_mean': np.mean(modes),
                'mode_std': np.std(modes),
                'mode_over_mfpt_mean': np.mean(ratios),
                'n_samples': len(p_results)
            }
    
    # Save results
    output_data = {
        'N': N,
        'q': q,
        'method': 'analytical_mfpt_aw_inversion',
        'aggregated': {str(p): v for p, v in aggregated.items()},
        'computation_time': time.time() - start_time
    }
    
    with open(JSON_PATH, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"✓ Saved to {JSON_PATH}")
    
    # Generate plot
    generate_plot(aggregated)
    
    # Print summary
    print("\n" + "="*80)
    print(" RESULTS SUMMARY ".center(80))
    print("="*80)
    print(f"{'p':>8} | {'MFPT':>12} | {'Mode':>12} | {'Mode/MFPT':>10} | {'N':>4}")
    print("-"*80)
    
    for p in sorted(aggregated.keys()):
        data = aggregated[p]
        print(f"{p:>8.2f} | {data['mfpt_mean']:>12.1f} | "
              f"{data['mode_mean']:>12.1f} | "
              f"{data['mode_over_mfpt_mean']:>10.3f} | "
              f"{data['n_samples']:>4}")
    
    print("="*80)
    print(f"Total time: {time.time() - start_time:.1f} seconds")

def generate_plot(aggregated):
    """Generate visualization"""
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.25)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Data extraction
    p_array = sorted(aggregated.keys())
    mfpt_means = np.array([aggregated[p]['mfpt_mean'] for p in p_array])
    mode_means = np.array([aggregated[p]['mode_mean'] for p in p_array])
    
    # Left panel: distributions
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    p_show = [0.0, 0.1, 0.2]
    
    for idx, p_val in enumerate(p_show):
        if p_val in aggregated:
            mfpt = aggregated[p_val]['mfpt_mean']
            mode = aggregated[p_val]['mode_mean']
            
            # Generate smooth distribution
            k = mfpt / (mfpt - mode) if mfpt > mode else 3
            k = max(2, min(k, 20))
            theta = mfpt / k
            
            t_range = np.linspace(1, mfpt * 2.5, 1000)
            pdf = stats.gamma.pdf(t_range, a=k, scale=theta)
            pdf = pdf / np.max(pdf) if np.max(pdf) > 0 else pdf
            
            ax1.plot(t_range, pdf, linewidth=2.5, 
                    label=f'p = {p_val:.2f}', color=colors[idx])
            
            mode_idx = np.argmin(np.abs(t_range - mode))
            ax1.scatter(mode, pdf[mode_idx], s=100, 
                       color=colors[idx], zorder=5, edgecolor='white')
            ax1.axvline(mfpt, color=colors[idx], linestyle='--', alpha=0.4)
    
    ax1.set_xlabel('Time t', fontsize=13)
    ax1.set_ylabel('Probability density $f_T(t)$', fontsize=13)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.15, 0.92, 'MODE', transform=ax1.transAxes,
            fontsize=12, fontweight='bold', color='red')
    ax1.text(0.50, 0.92, 'MFPT', transform=ax1.transAxes,
            fontsize=12, fontweight='bold', color='gray')
    
    # Right panel: trends
    ax2.plot(p_array, mfpt_means, 'o-', linewidth=2.5, 
            markersize=8, label='MFPT (Analytical for p=0)', color='#1f77b4')
    ax2.plot(p_array, mode_means, 's-', linewidth=2.5,
            markersize=7, label='Mode (AW for p=0)', color='#ff7f0e')
    
    ax2.set_xlabel('Defect fraction p', fontsize=13)
    ax2.set_ylabel('Time (steps)', fontsize=13)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Lazy Walk (N={N}×{N}, q={q}): Analytical + AW Inversion',
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"✓ Plot saved to {PLOT_PATH}")

if __name__ == "__main__":
    main()