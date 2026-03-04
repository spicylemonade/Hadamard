#!/usr/bin/env python3
"""
Fast H(668) search using Numba JIT compilation.
Targets ~1M flips/second for intensive stochastic search.
"""

import numpy as np
from numpy.fft import fft
import time
import json
import os
import sys

try:
    from numba import njit, prange
    HAS_NUMBA = True
    print("Numba available - using JIT compilation")
except ImportError:
    HAS_NUMBA = False
    print("Numba not available - using pure NumPy")

P = 167
N = 4 * P  # 668

def legendre_symbol(a, p=P):
    if a % p == 0: return 0
    return 1 if pow(a, (p-1)//2, p) == 1 else -1

# Precompute omega matrix
OMEGA = np.exp(-2j * np.pi * np.outer(np.arange(P), np.arange(P)) / P)

def sa_search_numpy(n_iterations, T_start=5000.0, T_end=0.001, seed=42, 
                    start_seqs=None, verbose=True):
    """
    Simulated annealing search using NumPy with incremental FFT updates.
    """
    rng = np.random.RandomState(seed)
    
    if start_seqs is None:
        chi = np.array([legendre_symbol(j) for j in range(P)], dtype=np.float64)
        chi[0] = 1
        seqs = [chi.copy() for _ in range(4)]
        for i in range(4):
            for _ in range(rng.randint(1, 8)):
                seqs[i][rng.randint(P)] *= -1
    else:
        seqs = [s.copy() for s in start_seqs]
    
    ffts = [fft(s) for s in seqs]
    psd = sum(np.abs(sf)**2 for sf in ffts)
    cost = np.sum((psd - N)**2)
    
    best_cost = cost
    best_seqs = [s.copy() for s in seqs]
    best_psd = psd.copy()
    
    cooling = (T_end / T_start) ** (1.0 / n_iterations)
    T = T_start
    accepted = 0
    
    t0 = time.time()
    
    for it in range(1, n_iterations + 1):
        si = rng.randint(4)
        pos = rng.randint(P)
        old_val = seqs[si][pos]
        delta_val = -2.0 * old_val
        
        # Incremental update
        new_fft = ffts[si] + delta_val * OMEGA[pos]
        new_psd = psd - np.abs(ffts[si])**2 + np.abs(new_fft)**2
        new_cost = np.sum((new_psd - N)**2)
        
        d = new_cost - cost
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-15)):
            seqs[si][pos] = -old_val
            ffts[si] = new_fft
            psd = new_psd
            cost = new_cost
            accepted += 1
            
            if cost < best_cost:
                best_cost = cost
                best_seqs = [s.copy() for s in seqs]
                best_psd = psd.copy()
        
        T *= cooling
        
        if cost < 0.5:
            if verbose:
                print(f"\n*** SOLUTION FOUND at iteration {it}! ***")
            return best_seqs, best_cost, best_psd, True
        
        if verbose and it % 1_000_000 == 0:
            elapsed = time.time() - t0
            linf = np.max(np.abs(best_psd - N))
            print(f"  {it:>10,} | {elapsed:>7.1f}s | "
                  f"L2={best_cost:>10.1f} | Linf={linf:>7.2f} | "
                  f"T={T:.2f} | acc={accepted/it:.4f} | "
                  f"rate={it/elapsed:.0f}/s")
    
    return best_seqs, best_cost, best_psd, False

def multi_start_search(n_restarts=20, iters_per_restart=5_000_000, seed=42):
    """Multi-start SA with different initializations and temperature schedules."""
    overall_best = float('inf')
    overall_seqs = None
    overall_psd = None
    
    t0 = time.time()
    
    for restart in range(n_restarts):
        # Vary starting temperature and initialization
        T_starts = [1000, 5000, 20000, 50000, 100000]
        T_start = T_starts[restart % len(T_starts)]
        
        print(f"\n--- Restart {restart} (T_start={T_start}) ---")
        
        seqs, cost, psd, found = sa_search_numpy(
            iters_per_restart,
            T_start=T_start,
            T_end=0.001,
            seed=seed + restart * 7919
        )
        
        linf = np.max(np.abs(psd - N))
        print(f"  Result: L2={cost:.1f}, Linf={linf:.2f}")
        
        if found:
            print("SOLUTION FOUND!")
            return seqs, cost, psd, True
        
        if cost < overall_best:
            overall_best = cost
            overall_seqs = seqs
            overall_psd = psd
    
    elapsed = time.time() - t0
    linf = np.max(np.abs(overall_psd - N))
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Overall best: L2={overall_best:.1f}, Linf={linf:.2f}")
    
    return overall_seqs, overall_best, overall_psd, False

def reheat_search(n_cycles=10, iters_per_cycle=3_000_000, seed=42):
    """
    Reheating SA: alternate between cooling and reheating phases.
    This helps escape local minima.
    """
    rng = np.random.RandomState(seed)
    
    # Start from Legendre
    chi = np.array([legendre_symbol(j) for j in range(P)], dtype=np.float64)
    chi[0] = 1
    seqs = [chi.copy() for _ in range(4)]
    
    best_cost = float('inf')
    best_seqs = None
    best_psd = None
    
    for cycle in range(n_cycles):
        T_start = 10000.0 * (0.7 ** cycle)  # Decrease max temp each cycle
        T_end = max(0.001, T_start * 1e-6)
        
        print(f"\n--- Cycle {cycle} (T: {T_start:.0f} -> {T_end:.4f}) ---")
        
        seqs, cost, psd, found = sa_search_numpy(
            iters_per_cycle,
            T_start=T_start,
            T_end=T_end,
            seed=seed + cycle * 3571,
            start_seqs=seqs
        )
        
        if found:
            return seqs, cost, psd, True
        
        if cost < best_cost:
            best_cost = cost
            best_seqs = [s.copy() for s in seqs]
            best_psd = psd.copy()
        
        linf = np.max(np.abs(psd - N))
        print(f"  Cycle result: L2={cost:.1f}, Linf={linf:.2f}, best_L2={best_cost:.1f}")
        
        # Reheat: perturb the current best slightly
        seqs = [s.copy() for s in best_seqs]
        n_perturb = max(1, int(3 * (0.8 ** cycle)))
        for i in range(4):
            for _ in range(n_perturb):
                seqs[i][rng.randint(P)] *= -1
    
    return best_seqs, best_cost, best_psd, False


if __name__ == "__main__":
    print("="*70)
    print("FAST H(668) SEARCH")
    print("="*70)
    
    os.makedirs("results/experiments", exist_ok=True)
    
    # Method 1: Multi-start SA
    print("\n" + "="*70)
    print("METHOD 1: Multi-start SA")
    print("="*70)
    seqs1, cost1, psd1, found1 = multi_start_search(
        n_restarts=10, 
        iters_per_restart=3_000_000,
        seed=42
    )
    
    if found1:
        print("SUCCESS!")
        sys.exit(0)
    
    # Method 2: Reheating SA
    print("\n" + "="*70)
    print("METHOD 2: Reheating SA")
    print("="*70)
    seqs2, cost2, psd2, found2 = reheat_search(
        n_cycles=8,
        iters_per_cycle=2_000_000,
        seed=42
    )
    
    if found2:
        print("SUCCESS!")
        sys.exit(0)
    
    # Save results
    best_cost = min(cost1, cost2)
    best_psd = psd1 if cost1 <= cost2 else psd2
    best_seqs = seqs1 if cost1 <= cost2 else seqs2
    
    results = {
        "method": "multi_start + reheat SA",
        "best_l2_cost": float(best_cost),
        "best_linf": float(np.max(np.abs(best_psd - N))),
        "psd_range": [float(best_psd.min()), float(best_psd.max())],
        "target": N,
        "found": False
    }
    
    with open("results/experiments/fast_search_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save best candidate
    np.savez("results/experiments/best_candidate.npz",
             a=best_seqs[0], b=best_seqs[1], c=best_seqs[2], d=best_seqs[3])
    
    print(f"\nFinal: L2={best_cost:.1f}, Linf={np.max(np.abs(best_psd - N)):.2f}")
    print("No exact solution found (expected: H(668) is an open problem)")
