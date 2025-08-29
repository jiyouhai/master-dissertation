#!/usr/bin/env python3
"""
Plot MFPT and Mode vs N from JSON data for 3D reflecting random walk
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt

# File paths
JSON_PATH = "mfpt_mode_data_3d.json"
PDF_PATH = "mfpt_vs_mode_3d_simple.pdf"

def load_data(json_path):
    """Load data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    N = np.array(data['N'])
    mfpt = np.array(data['mfpt'])
    mode = np.array(data['mode'])
    
    return N, mfpt, mode

def create_plot(N, mfpt, mode, pdf_path):
    """Create log-log plot of MFPT and Mode vs N"""
    
    # Set plot style
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
        "figure.dpi": 100,
        "font.family": "sans-serif",
        "axes.linewidth": 1.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot MFPT
    ax.loglog(N, mfpt, 'o-', 
             linewidth=2.0, 
             markersize=7,
             color='#1f77b4', 
             label='MFPT', 
             alpha=0.9)
    
    # Plot Mode
    ax.loglog(N, mode, 's-', 
             linewidth=2.0,
             markersize=7, 
             color='#ff7f0e', 
             label='Mode', 
             alpha=0.9)
    
    # Labels and title
    ax.set_xlabel("Lattice size N", fontsize=14)
    ax.set_ylabel("First-passage time", fontsize=14)
    ax.set_title("3D Reflecting Random Walk (q=0.8)", fontsize=16, pad=15)
    
    # Grid
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
    ax.grid(True, which="major", alpha=0.4, linewidth=0.7)
    
    # Legend
    ax.legend(loc='upper left', 
             framealpha=0.95, 
             edgecolor='gray',
             fancybox=True,
             shadow=True)
    
    # Set axis limits with some padding
    ax.set_xlim([N.min() * 0.9, N.max() * 1.1])
    ax.set_ylim([min(mfpt.min(), mode.min()) * 0.8, 
                max(mfpt.max(), mode.max()) * 1.2])
    
    # Add minor ticks
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    fig.savefig(pdf_path, format="pdf", bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Plot saved to: {pdf_path}")

def main():
    """Main function"""
    print("Loading data from JSON...")
    
    try:
        # Load data
        N, mfpt, mode = load_data(JSON_PATH)
        
        # Print summary statistics
        print(f"\nData Summary:")
        print(f"  N range: {N.min()} to {N.max()} ({len(N)} points)")
        print(f"  MFPT range: {mfpt.min():.1f} to {mfpt.max():.1f}")
        print(f"  Mode range: {mode.min():.1f} to {mode.max():.1f}")
        
        # Calculate and print Mode/MFPT ratio
        ratio = mode / mfpt
        print(f"  Mode/MFPT ratio: mean = {ratio.mean():.4f}, std = {ratio.std():.6f}")
        print(f"  Mode/MFPT range: {ratio.min():.4f} to {ratio.max():.4f}")
        
        # Create plot
        print(f"\nCreating plot...")
        create_plot(N, mfpt, mode, PDF_PATH)
        
        print("\nDone!")
        
    except FileNotFoundError:
        print(f"Error: Could not find {JSON_PATH}")
        print("Make sure the JSON file is in the same directory as this script.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()