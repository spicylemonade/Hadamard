#!/usr/bin/env python3
"""
Baseline Legendre-symbol-based sequences for H(668).
Demonstrates the fundamental PSD gap: 4 Legendre copies give PSD=672, need 668.

Item 007 of research rubric.
Deterministic seed: 42
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.fft import fft
import os
import sys

# Add parent path
sys.path.insert(0, os.path.dirname(__file__))
from hadamard_core import legendre_symbol, psd_check, P, N

def compute_legendre_sequence(p=P):
    """Compute Legendre symbol sequence chi(i) for i = 0, ..., p-1."""
    seq = np.array([legendre_symbol(i, p) for i in range(p)], dtype=np.int8)
    # chi(0) = 0, set to +1 (standard convention for Hadamard construction)
    seq[0] = 1
    return seq

def analyze_legendre_baseline():
    """Full analysis of Legendre-based sequences for GS construction."""
    chi = compute_legendre_sequence()
    
    print(f"Legendre sequence for p={P}:")
    print(f"  Length: {len(chi)}")
    print(f"  Sum: {np.sum(chi)} (expected: 1, since chi(0)=1 and |QR|=|QNR|=(p-1)/2)")
    print(f"  # of +1: {np.sum(chi == 1)}, # of -1: {np.sum(chi == -1)}")
    
    # Strategy 1: Four identical Legendre copies
    a = b = c = d = chi.copy()
    psd_vals, deviation = psd_check(a, b, c, d)
    
    print(f"\n--- Strategy 1: Four identical Legendre copies ---")
    print(f"  PSD at k=0: {psd_vals[0]:.4f}")
    print(f"  PSD at k=1..{P-1} (non-zero freq):")
    print(f"    min: {np.min(psd_vals[1:]):.4f}")
    print(f"    max: {np.max(psd_vals[1:]):.4f}")
    print(f"    mean: {np.mean(psd_vals[1:]):.4f}")
    print(f"    std: {np.std(psd_vals[1:]):.6f}")
    print(f"  Target: {N}")
    print(f"  Excess at non-zero freq: {np.mean(psd_vals[1:]) - N:.4f}")
    print(f"  Max |deviation|: {np.max(np.abs(deviation[1:])):.4f}")
    
    # Strategy 2: Three Legendre + one all-ones
    e = np.ones(P, dtype=np.int8)
    psd2, dev2 = psd_check(chi, chi, chi, e)
    
    print(f"\n--- Strategy 2: Three Legendre + one all-ones ---")
    print(f"  PSD at k=0: {psd2[0]:.4f} (target: {N})")
    print(f"  PSD at k=1..{P-1}:")
    print(f"    min: {np.min(psd2[1:]):.4f}")
    print(f"    max: {np.max(psd2[1:]):.4f}")
    print(f"  Max |deviation|: {np.max(np.abs(dev2)):.4f}")
    
    # Strategy 3: Two Legendre + two negated Legendre
    neg_chi = -chi.copy()
    psd3, dev3 = psd_check(chi, chi, neg_chi, neg_chi)
    
    print(f"\n--- Strategy 3: Two Legendre + two negated ---")
    print(f"  PSD at k=0: {psd3[0]:.4f}")
    print(f"  PSD at k=1..{P-1}:")
    print(f"    min: {np.min(psd3[1:]):.4f}")
    print(f"    max: {np.max(psd3[1:]):.4f}")
    print(f"  Note: Negation doesn't change |DFT|^2, so PSD is identical to Strategy 1")
    
    # Verify the theoretical PSD value
    print(f"\n--- Theoretical Analysis ---")
    print(f"  For p={P} (p ≡ 3 mod 4):")
    print(f"  Gauss sum G = i*sqrt({P})")
    print(f"  |chi_hat(k)|^2 = 1 + p = {1+P} for k ≠ 0")
    print(f"  4 * |chi_hat(k)|^2 = 4 * {1+P} = {4*(1+P)}")
    print(f"  Target PSD = 4p = {4*P}")
    print(f"  Gap = 4(p+1) - 4p = 4")
    
    return chi, psd_vals, deviation

def generate_psd_figure(chi, psd_vals, deviation):
    """Generate diagnostic PSD plot."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    freqs = np.arange(P)
    
    # Top: PSD values
    ax1 = axes[0]
    ax1.plot(freqs[1:], psd_vals[1:], 'b-', linewidth=0.5, alpha=0.7, label='PSD (4 Legendre)')
    ax1.axhline(y=N, color='r', linestyle='--', linewidth=1.5, label=f'Target = {N}')
    ax1.axhline(y=4*(P+1), color='g', linestyle=':', linewidth=1.5, label=f'4(p+1) = {4*(P+1)}')
    ax1.set_xlabel('Frequency index k')
    ax1.set_ylabel('PSD(k)')
    ax1.set_title(f'Power Spectral Density: 4 Legendre Sequences (p={P})')
    ax1.legend()
    ax1.set_xlim(1, P-1)
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Deviation from target
    ax2 = axes[1]
    ax2.bar(freqs[1:], deviation[1:], color='red', alpha=0.6, width=1.0)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axhline(y=4, color='g', linestyle=':', linewidth=1.5, label='Constant gap = +4')
    ax2.set_xlabel('Frequency index k')
    ax2.set_ylabel(f'PSD(k) - {N}')
    ax2.set_title(f'PSD Deviation from Target ({N}): Constant Excess of 4')
    ax2.legend()
    ax2.set_xlim(1, P-1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    outpath = os.path.join(os.path.dirname(__file__), '..', 'figures', 'legendre_psd_gap.png')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to {outpath}")
    return outpath

def main():
    np.random.seed(42)
    
    print("=" * 60)
    print("BASELINE LEGENDRE ANALYSIS FOR H(668)")
    print("=" * 60)
    
    chi, psd_vals, deviation = analyze_legendre_baseline()
    figpath = generate_psd_figure(chi, psd_vals, deviation)
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print(f"The Legendre baseline gives a FLAT PSD of {4*(P+1)} at all {P-1} non-zero frequencies.")
    print(f"The target is {N}, so the gap is exactly 4 at every frequency.")
    print(f"This gap of 4 = 4(p+1) - 4p is the fundamental obstruction for H({N}).")
    print(f"Any valid construction must find four sequences whose combined PSD")
    print(f"differs from the Legendre baseline at enough frequencies to close this gap.")

if __name__ == "__main__":
    main()
