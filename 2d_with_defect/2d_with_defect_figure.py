#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a single PDF with two plots side by side:
1. Left: MFPT and mode vs defect fraction
2. Right: FPT distributions for different defect fractions
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import ScalarFormatter

# Set readable font sizes
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.linewidth": 1.2,
})

def fit_gamma_from_mean_mode(mean: float, mode: float):
    """Derive Gamma(k, θ) parameters from (mean, mode)."""
    mean = float(mean)
    mode = float(mode)
    if not (mean > 0 and 0 < mode < mean):
        mode = max(1e-12, min(mean * 0.95, mode))
    k = mean / (mean - mode)
    theta = mean - mode
    if k <= 1.0:
        k = 1.0000001
        theta = mean / k
    return k, theta

def gamma_pdf(t, k, theta):
    """PDF of Gamma(k,θ) distribution."""
    from math import gamma
    t = np.asarray(t, dtype=float)
    return np.where(
        t > 0,
        np.power(t, k - 1) * np.exp(-t / theta) / (gamma(k) * (theta ** k)),
        0.0,
    )

def generate_combined_plot():
    """Generate both plots side by side in a single figure."""
    # Read data
    df = pd.read_csv("exact_summary.csv")
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5))
    
    # ========== Left plot: Trend plot (原来的右图) ==========
    # Sort by fraction
    df_sorted = df.sort_values('fraction')
    
    # Plot MFPT and Mode
    ax1.plot(df_sorted['fraction'], df_sorted['MFPT_time'], 
            marker='o', markersize=7, linewidth=2.5, 
            color='#2E7BC9', label='Analytical MFPT')
    
    ax1.plot(df_sorted['fraction'], df_sorted['Mode_time'], 
            marker='s', markersize=7, linewidth=2.5, 
            color='#FF8C00', label='Analytical mode')
    
    # Labels
    ax1.set_xlabel('Defect fraction p', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Time', fontsize=13, fontweight='bold')
    
    # Grid
    ax1.grid(True, which='major', linestyle='--', alpha=0.35, linewidth=0.8)
    ax1.grid(True, which='minor', linestyle=':', alpha=0.18, linewidth=0.6)
    
    # Format y-axis for scientific notation
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax1.yaxis.get_offset_text().set_fontsize(11)
    
    # Legend
    ax1.legend(loc='upper left', frameon=True, 
              fancybox=True, shadow=True, ncol=2)
    
    # Remove top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Set axis below grid
    ax1.set_axisbelow(True)
    
    # ========== Right plot: Overlay plot (原来的左图) ==========
    # Select 4 evenly spaced p values
    target_ps = [0.0, 0.1, 0.2, 0.3]
    
    # Find closest available p values
    selected_data = []
    for target_p in target_ps:
        idx = (df['fraction'] - target_p).abs().idxmin()
        selected_data.append(df.iloc[idx])
    
    # Fit Gamma distributions
    params = []
    for row in selected_data:
        mean = float(row['MFPT_time'])
        mode = float(row['Mode_time'])
        k, theta = fit_gamma_from_mean_mode(mean, mode)
        params.append({
            'p': float(row['fraction']),
            'mean': mean,
            'mode': mode,
            'k': k,
            'theta': theta
        })
    
    # Create time grid
    t_max = max([p['mean'] + 3 * np.sqrt(p['k']) * p['theta'] for p in params])
    t = np.linspace(0, t_max * 0.7, 2000)
    
    # Color scheme
    colors = ['#2E7BC9', '#7B4397', '#C04B5E', '#E74C3C']
    
    # Store positions for annotations
    mode_positions = []
    mean_positions = []
    max_y = 0
    
    # Plot PDFs
    for i, pr in enumerate(params):
        y = gamma_pdf(t, pr['k'], pr['theta'])
        
        # Plot curve
        ax2.plot(t, y, label=f"p = {pr['p']:.2f}", 
                color=colors[i], linewidth=2.8, alpha=0.9)
        
        # MFPT vertical line
        ax2.axvline(pr['mean'], linestyle='--', linewidth=1.6, 
                   color=colors[i], alpha=0.4)
        mean_positions.append(pr['mean'])
        
        # Mode peak point
        y_mode = gamma_pdf(pr['mode'], pr['k'], pr['theta'])
        ax2.scatter([pr['mode']], [y_mode], marker='o', s=120, 
                   color=colors[i], edgecolor='white', linewidth=2.2, 
                   zorder=5, alpha=1.0)
        mode_positions.append(pr['mode'])
        max_y = max(max_y, y_mode)
    
    # Add arrow for MODE
    ax2.annotate('', 
                xy=(mode_positions[0], max_y * 1.12),
                xytext=(mode_positions[-1], max_y * 1.12),
                arrowprops=dict(arrowstyle='<->', lw=2.8, color='#E74C3C', alpha=0.8))
    
    ax2.text((mode_positions[0] + mode_positions[-1]) / 2, max_y * 1.18, 
            'MODE', 
            fontsize=13, fontweight='bold', color='#E74C3C',
            ha='center', va='bottom')
    
    # Add arrow for MFPT
    y_pos_mfpt = max_y * 0.6
    ax2.annotate('', 
                xy=(mean_positions[0], y_pos_mfpt),
                xytext=(mean_positions[-1], y_pos_mfpt),
                arrowprops=dict(arrowstyle='<->', lw=2.2, color='#666', alpha=0.6))
    
    ax2.text((mean_positions[0] + mean_positions[-1]) / 2, y_pos_mfpt * 1.12, 
            'MFPT', 
            fontsize=12, color='#666',
            ha='center', va='bottom')
    
    # Styling for right plot
    ax2.set_xlabel("Time t", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Probability density $f_T(t)$", fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.25, linewidth=0.8)
    ax2.set_xlim(0, t_max * 0.6)
    ax2.set_ylim(0, max_y * 1.28)
    
    # Format y-axis for scientific notation
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax2.yaxis.get_offset_text().set_fontsize(11)
    
    # Legend
    legend = ax2.legend(loc='upper right', frameon=True, 
                      title="Defect fraction", title_fontsize=11,
                      fancybox=True, shadow=True)
    legend.get_frame().set_alpha(0.95)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("combined_plots.pdf", bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Generated combined_plots.pdf")

def main():
    """Main function to generate combined plot."""
    # Check if CSV file exists
    csv_path = Path("exact_summary.csv")
    if not csv_path.exists():
        print("Error: exact_summary.csv not found in current directory!")
        print("Please ensure the CSV file is in the same folder as this script.")
        return
    
    # Generate combined plot
    try:
        print("Starting plot generation...")
        generate_combined_plot()
        print("\nSuccess! Combined PDF file has been generated:")
        print("  - combined_plots.pdf")
    except Exception as e:
        print(f"Error generating plots: {e}")
        print("Please check that exact_summary.csv has the required columns:")
        print("  - fraction")
        print("  - MFPT_time")
        print("  - Mode_time")

if __name__ == "__main__":
    main()