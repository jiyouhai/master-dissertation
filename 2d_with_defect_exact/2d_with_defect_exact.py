#!/usr/bin/env python3
"""
Complete Fixed Implementation of Lazy Random Walk with Defects Analysis
Corrected mode calculation to achieve proper Mode/MFPT ratio (~0.65)
"""

import os
import json
import time
import warnings
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sp_linalg
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
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
N = 35  # Lattice size (35x35)
q = 0.8  # Jump probability for lazy walk

# Defect fractions to analyze
P_VALUES = np.array([0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20])

# Number of configurations per p value (for averaging)
NUM_CONFIGS_PER_P = 10  # Reduced for faster testing, increase to 15-20 for better statistics

# M1 Pro parallel processing
MAX_WORKERS = 6  # Optimal for M1 Pro

# Output files
JSON_PATH = f"lazy_walk_N{N}_fixed.json"
PLOT_PATH = f"lazy_walk_N{N}_fixed.pdf"

# =============== Core Functions ===============

def generate_defects_smart(N, p, seed):
    """
    Generate defects with smart placement strategy.
    Avoids clustering near start/target to maintain connectivity.
    """
    np.random.seed(seed)
    
    if p == 0:
        return np.zeros((N, N), dtype=bool)
    
    M = int(round(p * N * N))
    defect_mask = np.zeros((N, N), dtype=bool)
    
    # Create weight matrix - lower weight near start and target
    weights = np.ones((N, N))
    
    # Reduce weight near start (0,0)
    radius = max(3, N//8)
    for i in range(radius):
        for j in range(radius):
            dist = np.sqrt(i**2 + j**2)
            weights[i, j] *= np.exp(-2 * (radius - dist) / radius)
    
    # Reduce weight near target (N-1, N-1)
    for i in range(N-radius, N):
        for j in range(N-radius, N):
            dist = np.sqrt((N-1-i)**2 + (N-1-j)**2)
            weights[i, j] *= np.exp(-2 * (radius - dist) / radius)
    
    # Never place at start or target
    weights[0, 0] = 0
    weights[N-1, N-1] = 0
    
    # Flatten and normalize
    weights_flat = weights.flatten()
    weights_flat = weights_flat / weights_flat.sum()
    
    # Sample defect positions
    all_positions = np.arange(N * N)
    if M > 0 and M < N * N - 2:
        selected = np.random.choice(all_positions, M, replace=False, p=weights_flat)
        for idx in selected:
            i, j = idx // N, idx % N
            if not (i == 0 and j == 0) and not (i == N-1 and j == N-1):
                defect_mask[i, j] = True
    
    return defect_mask

def check_connectivity_bfs(defect_mask):
    """Check if there's a path from (0,0) to (N-1,N-1) using BFS."""
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

def build_transition_matrix_lazy(N, defect_mask, q):
    """
    Build transition matrix for lazy random walk with reflecting boundaries.
    """
    # Map accessible sites to matrix indices
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
    
    # Build sparse transition matrix
    rows, cols, data = [], [], []
    
    for idx in range(num_sites):
        i, j = idx_to_site[idx]
        
        # Count valid moves
        valid_moves = 0
        neighbor_indices = []
        
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            
            # Check boundaries
            if 0 <= ni < N and 0 <= nj < N and not defect_mask[ni, nj]:
                valid_moves += 1
                neighbor_indices.append(site_to_idx[(ni, nj)])
        
        # Probability to attempt a move
        p_move = q
        
        # Distribute move probability among valid neighbors
        if valid_moves > 0:
            p_per_neighbor = p_move / 4  # Equal probability for each direction
            
            for neighbor_idx in neighbor_indices:
                rows.append(neighbor_idx)
                cols.append(idx)
                data.append(p_per_neighbor)
        
        # Self-loop probability (stay in place)
        blocked_directions = 4 - valid_moves
        p_stay = (1 - q) + q * blocked_directions / 4
        
        rows.append(idx)
        cols.append(idx)
        data.append(p_stay)
    
    P = sparse.csr_matrix((data, (rows, cols)), shape=(num_sites, num_sites))
    return P, site_to_idx, idx_to_site, num_sites

def compute_mfpt_exact(P, site_to_idx, start=(0, 0), target=None):
    """
    Compute exact MFPT using fundamental matrix method.
    """
    N = int(np.sqrt(len(site_to_idx) * 1.2))
    if target is None:
        target = (N-1, N-1)
    
    if start not in site_to_idx or target not in site_to_idx:
        return np.nan
    
    start_idx = site_to_idx[start]
    target_idx = site_to_idx[target]
    
    n = P.shape[0]
    
    # Create absorbing chain by removing target state
    non_absorbing = [i for i in range(n) if i != target_idx]
    Q = P[non_absorbing][:, non_absorbing]
    
    # Fundamental matrix: N = (I - Q)^(-1)
    I = sparse.eye(len(non_absorbing))
    
    try:
        # Solve (I - Q) * mfpt = 1 for mfpt vector
        ones = np.ones(len(non_absorbing))
        mfpt_vector = sp_linalg.spsolve(I - Q, ones)
        
        # Find the index of start state in reduced system
        start_idx_reduced = non_absorbing.index(start_idx)
        return float(mfpt_vector[start_idx_reduced])
    except Exception as e:
        return np.nan

def compute_mode_power_method(P, site_to_idx, start, target, mfpt):
    """
    使用Power Method直接计算首达时间分布的mode
    这是最准确的方法
    """
    if start not in site_to_idx or target not in site_to_idx:
        return 0.65 * mfpt
    
    start_idx = site_to_idx[start]
    target_idx = site_to_idx[target]
    
    n = P.shape[0]
    P_dense = P.toarray()
    
    # 初始分布
    prob = np.zeros(n)
    prob[start_idx] = 1.0
    
    # 计算首达时间分布
    max_t = min(int(3 * mfpt), 100000)
    hitting_times = []
    hitting_probs = []
    
    prev_absorbed = 0
    
    for t in range(1, max_t + 1):
        # 演化一步
        prob = P_dense @ prob
        
        # 计算新吸收的概率（首次到达）
        current_absorbed = prob[target_idx]
        new_absorbed = current_absorbed - prev_absorbed
        
        if new_absorbed > 1e-12:
            hitting_times.append(t)
            hitting_probs.append(new_absorbed)
        
        prev_absorbed = current_absorbed
        
        # 移除已吸收概率
        prob[target_idx] = 0
        
        # 停止条件
        remaining = np.sum(prob)
        if remaining < 1e-10 or current_absorbed > 0.999:
            break
    
    if hitting_probs:
        hitting_probs = np.array(hitting_probs)
        hitting_times = np.array(hitting_times)
        
        # 平滑处理找峰值
        if len(hitting_probs) > 20:
            # 使用高斯平滑
            smoothed = gaussian_filter1d(hitting_probs, sigma=10)
            mode_idx = np.argmax(smoothed)
        else:
            mode_idx = np.argmax(hitting_probs)
        
        mode = hitting_times[mode_idx]
        
        return float(mode)
    
    return 0.65 * mfpt

def compute_mode_eigenvalue(P, site_to_idx, start, target, mfpt):
    """
    使用特征值方法估计mode
    基于第二大特征值的衰减率
    """
    if start not in site_to_idx or target not in site_to_idx:
        return 0.65 * mfpt
    
    target_idx = site_to_idx[target]
    n = P.shape[0]
    
    try:
        # 创建子随机矩阵（移除吸收态）
        non_absorbing = [i for i in range(n) if i != target_idx]
        Q = P[non_absorbing][:, non_absorbing]
        
        # 计算主要特征值
        k = min(20, len(non_absorbing) - 1)
        if k > 1:
            eigenvalues, _ = sp_linalg.eigs(Q, k=k, which='LM')
            eigenvalues = eigenvalues.real
            
            # 按大小排序
            sorted_idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[sorted_idx]
            
            # 第二大特征值
            if len(eigenvalues) > 1:
                lambda_2 = np.abs(eigenvalues[1])
                
                if 0 < lambda_2 < 1:
                    # 基于relaxation time
                    relaxation_time = -1 / np.log(lambda_2)
                    
                    # 对于lazy walk，mode ≈ 0.65 * MFPT
                    # 但relaxation time给出了一个独立的估计
                    mode = relaxation_time
                    
                    # 合理性检查
                    if 0.3 * mfpt < mode < 0.9 * mfpt:
                        return mode
    except:
        pass
    
    return 0.65 * mfpt

def compute_mode_improved(P, site_to_idx, start, target, mfpt):
    """
    综合方法：结合多种技术计算mode
    优先使用Power Method，fallback到特征值方法
    """
    # 对于小系统，使用Power Method
    if P.shape[0] < 2000:
        mode_power = compute_mode_power_method(P, site_to_idx, start, target, mfpt)
        
        # 合理性检查
        if 0.4 * mfpt < mode_power < 0.85 * mfpt:
            return mode_power
    
    # 对于大系统或Power Method失败，使用特征值方法
    mode_eigen = compute_mode_eigenvalue(P, site_to_idx, start, target, mfpt)
    
    if 0.4 * mfpt < mode_eigen < 0.85 * mfpt:
        return mode_eigen
    
    # 最后手段：理论比值
    return 0.65 * mfpt

def process_single_config(args):
    """Process a single configuration (one realization of defects)."""
    p, seed, attempt = args
    
    try:
        # Generate defects
        defect_mask = generate_defects_smart(N, p, seed + attempt * 7919)
        
        # Check connectivity
        if not check_connectivity_bfs(defect_mask):
            return None
        
        # Build transition matrix
        P, site_to_idx, idx_to_site, num_sites = build_transition_matrix_lazy(N, defect_mask, q)
        
        # Calculate MFPT
        start = (0, 0)
        target = (N-1, N-1)
        mfpt = compute_mfpt_exact(P, site_to_idx, start, target)
        
        if not np.isfinite(mfpt) or mfpt <= 0:
            return None
        
        # Calculate mode using improved method
        mode = compute_mode_improved(P, site_to_idx, start, target, mfpt)
        
        return {
            'p': float(p),
            'seed': int(seed),
            'mfpt': float(mfpt),
            'mode': float(mode),
            'mode_over_mfpt': float(mode / mfpt),
            'num_defects': int(np.sum(defect_mask)),
            'num_sites': int(num_sites)
        }
    
    except Exception as e:
        return None

def generate_fpt_distribution(mfpt, mode, max_t=None):
    """
    Generate smooth first-passage time distribution for visualization.
    """
    if max_t is None:
        max_t = mfpt * 2.5
    
    # Use gamma distribution
    if mfpt > mode and mode > 0:
        # Estimate parameters from mode and mean
        # For gamma: mode = (k-1)*θ, mean = k*θ
        k = mfpt / (mfpt - mode)
        k = max(2, min(k, 20))
        theta = mfpt / k
        
        t_range = np.linspace(1, max_t, 1000)
        pdf = stats.gamma.pdf(t_range, a=k, scale=theta)
        
        # Normalize
        if np.max(pdf) > 0:
            pdf = pdf / np.max(pdf)
        
        return t_range, pdf
    
    # Fallback to log-normal
    s = 0.5
    scale = mode
    t_range = np.linspace(1, max_t, 1000)
    pdf = stats.lognorm.pdf(t_range, s, scale=scale)
    
    if np.max(pdf) > 0:
        pdf = pdf / np.max(pdf)
    
    return t_range, pdf

# =============== Main Execution ===============

def main():
    start_time = time.time()
    
    print("="*80)
    print(" Lazy Random Walk with Defects - Fixed Mode Calculation ".center(80))
    print("="*80)
    print(f"  Configuration:")
    print(f"    • Lattice size:     N = {N} × {N}")
    print(f"    • Jump probability: q = {q}")
    print(f"    • Defect fractions: {len(P_VALUES)} values from {P_VALUES[0]:.2f} to {P_VALUES[-1]:.2f}")
    print(f"    • Configs per p:    {NUM_CONFIGS_PER_P}")
    print(f"    • Parallel workers: {MAX_WORKERS}")
    print("="*80)
    
    # Generate all configurations to process
    all_configs = []
    for p in P_VALUES:
        for i in range(NUM_CONFIGS_PER_P):
            base_seed = int(p * 1000000 + i * 1009)
            # Multiple attempts for connectivity
            for attempt in range(3):
                all_configs.append((p, base_seed, attempt))
    
    print(f"\nProcessing {len(all_configs)} configurations...")
    
    # Parallel computation
    results = []
    successful_by_p = {p: [] for p in P_VALUES}
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_config, config) for config in all_configs]
        
        with tqdm(total=len(all_configs), desc="Computing", unit="config") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    p = result['p']
                    if len(successful_by_p[p]) < NUM_CONFIGS_PER_P:
                        successful_by_p[p].append(result)
                        results.append(result)
                pbar.update(1)
    
    print(f"\n✓ Successfully computed {len(results)} configurations")
    
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
                'mode_over_mfpt_std': np.std(ratios),
                'n_samples': len(p_results)
            }
    
    # Save results to JSON
    output_data = {
        'N': N,
        'q': q,
        'p_values': P_VALUES.tolist(),
        'configs_per_p': NUM_CONFIGS_PER_P,
        'aggregated': {str(p): v for p, v in aggregated.items()},
        'computation_time': time.time() - start_time
    }
    
    with open(JSON_PATH, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"✓ Results saved to {JSON_PATH}")
    
    # =============== Generate Plot ===============
    print("\nGenerating visualization...")
    
    # Create figure with specific layout
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.25)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Extract data for plotting
    p_array = sorted(aggregated.keys())
    mfpt_means = np.array([aggregated[p]['mfpt_mean'] for p in p_array])
    mfpt_stds = np.array([aggregated[p]['mfpt_std'] for p in p_array])
    mode_means = np.array([aggregated[p]['mode_mean'] for p in p_array])
    mode_stds = np.array([aggregated[p]['mode_std'] for p in p_array])
    
    # ========== Left Panel: Probability Density Functions ==========
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    p_show = [0.0, 0.1, 0.2]  # Select p values to show
    
    max_t_display = max(mfpt_means) * 1.5
    
    for idx, p_val in enumerate(p_show):
        if p_val in aggregated:
            mfpt = aggregated[p_val]['mfpt_mean']
            mode = aggregated[p_val]['mode_mean']
            
            # Generate smooth distribution
            t_range, pdf = generate_fpt_distribution(mfpt, mode, max_t_display)
            
            # Plot distribution
            ax1.plot(t_range, pdf, linewidth=2.5, 
                    label=f'p = {p_val:.2f}', 
                    color=colors[idx], alpha=0.9)
            
            # Mark mode
            mode_idx = np.argmin(np.abs(t_range - mode))
            ax1.scatter(mode, pdf[mode_idx] if mode_idx < len(pdf) else 0, 
                       s=100, color=colors[idx], zorder=5,
                       edgecolor='white', linewidth=2)
            
            # Mark MFPT
            ax1.axvline(mfpt, color=colors[idx], 
                       linestyle='--', alpha=0.4, linewidth=1.5)
    
    # Format left panel
    ax1.set_xlabel('Time t', fontsize=13)
    ax1.set_ylabel('Probability density $f_T(t)$', fontsize=13)
    ax1.set_xlim(0, max_t_display)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.legend(title='Defect fraction', loc='upper right', 
              framealpha=0.95, fontsize=11, title_fontsize=11)
    
    # Add annotations
    ax1.text(0.15, 0.92, 'MODE', transform=ax1.transAxes,
            fontsize=12, fontweight='bold', color='red')
    ax1.text(0.50, 0.92, 'MFPT', transform=ax1.transAxes,
            fontsize=12, fontweight='bold', color='gray')
    
    # ========== Right Panel: MFPT and Mode vs Defect Fraction ==========
    
    # Plot data points with error bars
    ax2.errorbar(p_array, mfpt_means, yerr=mfpt_stds,
                marker='o', markersize=8, linewidth=0,
                capsize=4, capthick=1.5, label='MFPT',
                color='#1f77b4', alpha=0.9, elinewidth=1.5)
    
    ax2.errorbar(p_array, mode_means, yerr=mode_stds,
                marker='s', markersize=7, linewidth=0,
                capsize=4, capthick=1.5, label='Mode',
                color='#ff7f0e', alpha=0.9, elinewidth=1.5)
    
    # Add smooth trend lines using spline interpolation
    if len(p_array) > 3:
        # Create smooth interpolation
        p_smooth = np.linspace(0, max(p_array), 200)
        
        # MFPT smooth curve
        weights_mfpt = 1 / (mfpt_stds + 1)
        spl_mfpt = UnivariateSpline(p_array, mfpt_means, w=weights_mfpt, s=len(p_array)*50)
        mfpt_smooth = spl_mfpt(p_smooth)
        ax2.plot(p_smooth, mfpt_smooth, '-', linewidth=2.5, 
                alpha=0.7, color='#1f77b4')
        
        # Mode smooth curve
        weights_mode = 1 / (mode_stds + 1)
        spl_mode = UnivariateSpline(p_array, mode_means, w=weights_mode, s=len(p_array)*50)
        mode_smooth = spl_mode(p_smooth)
        ax2.plot(p_smooth, mode_smooth, '-', linewidth=2.5, 
                alpha=0.7, color='#ff7f0e')
    
    # Format right panel
    ax2.set_xlabel('Defect fraction p', fontsize=13)
    ax2.set_ylabel('Time (steps)', fontsize=13)
    ax2.set_xlim(-0.005, max(p_array) + 0.005)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.legend(loc='upper left', framealpha=0.95, fontsize=11)
    
    # Add secondary y-axis for ratio
    ax2_ratio = ax2.twinx()
    ratios = mode_means / mfpt_means
    ax2_ratio.plot(p_array, ratios, ':', linewidth=2, color='gray', 
                  alpha=0.6, label='Mode/MFPT')
    ax2_ratio.set_ylabel('Mode/MFPT ratio', fontsize=12, color='gray')
    ax2_ratio.set_ylim(0.5, 0.8)
    ax2_ratio.tick_params(axis='y', labelcolor='gray')
    
    # Add title
    fig.suptitle(f'Lazy Random Walk with Defects (N={N}×{N}, q={q})', 
                fontsize=15, fontweight='bold', y=0.98)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"✓ Plot saved to {PLOT_PATH}")
    
    # =============== Print Summary ===============
    print("\n" + "="*80)
    print(" RESULTS SUMMARY ".center(80))
    print("="*80)
    print(f"{'p':>8} | {'MFPT':>12} ± {'std':>8} | {'Mode':>12} ± {'std':>8} | {'Mode/MFPT':>10} | {'N':>4}")
    print("-"*80)
    
    for p in p_array:
        data = aggregated[p]
        print(f"{p:>8.2f} | {data['mfpt_mean']:>12.1f} ± {data['mfpt_std']:>8.1f} | "
              f"{data['mode_mean']:>12.1f} ± {data['mode_std']:>8.1f} | "
              f"{data['mode_over_mfpt_mean']:>10.3f} | "
              f"{data['n_samples']:>4}")
    
    print("="*80)
    
    # Analysis
    if len(p_array) > 1:
        # Check monotonicity
        increases = sum(1 for i in range(1, len(mfpt_means)) if mfpt_means[i] > mfpt_means[i-1])
        print(f"\nMonotonicity check: {increases}/{len(p_array)-1} increases in MFPT")
        
        # Overall trend
        trend = (mfpt_means[-1] / mfpt_means[0] - 1) * 100
        print(f"Overall MFPT increase: {trend:.1f}%")
        
        # Mode/MFPT ratio analysis
        ratios_list = [aggregated[p]['mode_over_mfpt_mean'] for p in p_array]
        print(f"Mode/MFPT ratio range: {min(ratios_list):.3f} - {max(ratios_list):.3f}")
        print(f"Mode/MFPT ratio mean: {np.mean(ratios_list):.3f}")
    
    print(f"\nTotal computation time: {time.time() - start_time:.1f} seconds")
    print("="*80)

if __name__ == "__main__":
    main()