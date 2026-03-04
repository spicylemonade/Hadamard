#!/usr/bin/env python3
"""
High-performance search for Hadamard matrix H(668) = H(4×167).
Uses Goethals-Seidel array with four ±1 circulant blocks of order 167.

APPROACH: Parallel tempering with DFT-guided neighborhood moves.
The key condition: |DFT(a)|^2 + |DFT(b)|^2 + |DFT(c)|^2 + |DFT(d)|^2 = 668
at every frequency k = 0, 1, ..., 166.

Optimization: Use incremental FFT updates for O(n) per move instead of O(n log n).
"""

import numpy as np
from numpy.fft import fft
import time
import json
import os
import sys

P = 167
N = 4 * P  # 668

def legendre_symbol(a, p=P):
    """Compute Legendre symbol (a/p)."""
    if a % p == 0:
        return 0
    val = pow(a, (p - 1) // 2, p)
    return 1 if val == 1 else -1

def compute_psd(seqs):
    """Compute power spectral density sum for four sequences."""
    psd = np.zeros(P, dtype=np.float64)
    for s in seqs:
        sf = fft(s.astype(np.float64))
        psd += np.abs(sf)**2
    return psd

def compute_l2_cost(psd):
    """L2 cost = sum of (PSD - N)^2."""
    return np.sum((psd - N)**2)

def compute_linf(psd):
    """L-infinity = max |PSD - N|."""
    return np.max(np.abs(psd - N))

# Precompute the DFT basis vectors for incremental updates
# omega[j,k] = exp(-2*pi*i*j*k/P)
_omega_matrix = np.exp(-2j * np.pi * np.outer(np.arange(P), np.arange(P)) / P)

def incremental_flip(psd, seq_ffts, seq_idx, pos, old_val):
    """
    Update PSD after flipping seqs[seq_idx][pos] from old_val to -old_val.
    
    delta = -2 * old_val
    new_fft[k] = old_fft[k] + delta * omega[pos, k]
    new_|fft[k]|^2 = |old_fft[k]|^2 + 2*delta*Re(conj(old_fft[k])*omega[pos,k]) + delta^2
    
    PSD change = new_|fft|^2 - old_|fft|^2
    """
    delta = -2.0 * old_val
    omega_row = _omega_matrix[pos]  # omega[pos, :] = exp(-2*pi*i*pos*k/P) for k=0..P-1
    
    old_fft = seq_ffts[seq_idx]
    
    # New FFT for this sequence
    new_fft = old_fft + delta * omega_row
    
    # Update PSD: subtract old contribution, add new
    psd_new = psd - np.abs(old_fft)**2 + np.abs(new_fft)**2
    
    return psd_new, new_fft

class ReplicaState:
    """State for a single replica in parallel tempering."""
    def __init__(self, seqs, temperature):
        self.seqs = [s.copy() for s in seqs]
        self.T = temperature
        self.ffts = [fft(s.astype(np.float64)) for s in seqs]
        self.psd = sum(np.abs(sf)**2 for sf in self.ffts)
        self.cost = compute_l2_cost(self.psd)
    
    def try_flip(self, seq_idx, pos, rng):
        """Try flipping seqs[seq_idx][pos] and accept/reject."""
        old_val = self.seqs[seq_idx][pos]
        
        new_psd, new_fft = incremental_flip(self.psd, self.ffts, seq_idx, pos, old_val)
        new_cost = compute_l2_cost(new_psd)
        
        delta = new_cost - self.cost
        
        if delta <= 0 or rng.random() < np.exp(-delta / max(self.T, 1e-15)):
            # Accept
            self.seqs[seq_idx][pos] = -old_val
            self.ffts[seq_idx] = new_fft
            self.psd = new_psd
            self.cost = new_cost
            return True
        return False

def make_legendre_seqs(seed=42):
    """Create initial sequences based on Legendre symbol."""
    rng = np.random.RandomState(seed)
    chi = np.array([legendre_symbol(j) for j in range(P)], dtype=np.float64)
    chi[0] = 1  # Replace 0 with 1
    
    seqs = []
    for i in range(4):
        s = chi.copy()
        # Random perturbation
        mask = rng.random(P) < 0.08
        s[mask] *= -1
        seqs.append(s)
    return seqs

def make_random_seqs(seed=42):
    """Create random ±1 sequences."""
    rng = np.random.RandomState(seed)
    return [rng.choice([-1.0, 1.0], size=P) for _ in range(4)]

def parallel_tempering(n_replicas=16, n_iterations=20_000_000, 
                       exchange_interval=50, log_interval=500_000, seed=42):
    """
    Parallel tempering search for H(668) sequences.
    """
    rng = np.random.RandomState(seed)
    
    # Temperature ladder
    T_min, T_max = 0.05, 1000.0
    temps = np.geomspace(T_min, T_max, n_replicas)
    
    # Initialize replicas with different starting points
    replicas = []
    for i in range(n_replicas):
        if i < n_replicas // 2:
            seqs = make_legendre_seqs(seed=seed + i * 100)
        else:
            seqs = make_random_seqs(seed=seed + i * 100)
        replicas.append(ReplicaState(seqs, temps[i]))
    
    # Track global best
    best_cost = min(r.cost for r in replicas)
    best_replica = min(range(n_replicas), key=lambda i: replicas[i].cost)
    best_seqs = [s.copy() for s in replicas[best_replica].seqs]
    best_psd = replicas[best_replica].psd.copy()
    
    t0 = time.time()
    total_flips = 0
    accepted = 0
    
    print(f"Parallel Tempering Search for H({N})")
    print(f"  Replicas: {n_replicas}, Temps: [{T_min:.2f}, {T_max:.2f}]")
    print(f"  Iterations: {n_iterations:,}")
    print(f"  Initial best L2 cost: {best_cost:.1f}")
    print(f"  Initial best Linf: {compute_linf(best_psd):.2f}")
    print()
    
    log_entries = []
    
    for it in range(1, n_iterations + 1):
        # Single flip per replica
        for r_idx, rep in enumerate(replicas):
            seq_idx = rng.randint(4)
            pos = rng.randint(P)
            if rep.try_flip(seq_idx, pos, rng):
                accepted += 1
                if rep.cost < best_cost:
                    best_cost = rep.cost
                    best_seqs = [s.copy() for s in rep.seqs]
                    best_psd = rep.psd.copy()
            total_flips += 1
        
        # Replica exchange
        if it % exchange_interval == 0:
            for i in range(n_replicas - 1):
                j = i + 1
                beta_diff = 1.0/replicas[i].T - 1.0/replicas[j].T
                energy_diff = replicas[j].cost - replicas[i].cost
                log_prob = beta_diff * energy_diff
                
                if log_prob < 0 or rng.random() < np.exp(-min(log_prob, 500)):
                    # Swap states
                    replicas[i].seqs, replicas[j].seqs = replicas[j].seqs, replicas[i].seqs
                    replicas[i].ffts, replicas[j].ffts = replicas[j].ffts, replicas[i].ffts
                    replicas[i].psd, replicas[j].psd = replicas[j].psd, replicas[i].psd
                    replicas[i].cost, replicas[j].cost = replicas[j].cost, replicas[i].cost
        
        # Logging
        if it % log_interval == 0:
            elapsed = time.time() - t0
            linf = compute_linf(best_psd)
            rate = total_flips / elapsed
            
            entry = {
                "iter": it, "elapsed_s": round(elapsed, 1),
                "best_l2": round(float(best_cost), 1),
                "best_linf": round(float(linf), 2),
                "flips_per_sec": round(rate, 0),
                "accept_rate": round(accepted / total_flips, 3),
                "coldest_cost": round(float(replicas[0].cost), 1)
            }
            log_entries.append(entry)
            
            print(f"  {it:>12,} | {elapsed:>7.1f}s | "
                  f"L2={best_cost:>10.1f} | Linf={linf:>7.2f} | "
                  f"rate={rate:>8.0f}/s | accept={accepted/total_flips:.3f} | "
                  f"cold={replicas[0].cost:>10.1f}")
            
            if best_cost < 0.5:
                print(f"\n*** EXACT SOLUTION FOUND at iteration {it}! ***")
                break
    
    elapsed = time.time() - t0
    linf = compute_linf(best_psd)
    print(f"\nSearch complete: {elapsed:.1f}s, {total_flips:,} flips")
    print(f"Best: L2={best_cost:.1f}, Linf={linf:.2f}")
    
    return best_seqs, best_cost, best_psd, log_entries


def build_and_verify(seqs):
    """Build Goethals-Seidel matrix and verify."""
    from hadamard_core import goethals_seidel_array, verify_hadamard, export_csv
    
    seqs_int = [s.astype(np.int8) for s in seqs]
    H = goethals_seidel_array(seqs_int[0], seqs_int[1], seqs_int[2], seqs_int[3])
    valid, msg = verify_hadamard(H)
    print(f"Verification: {msg}")
    
    if valid:
        path = os.path.join(os.path.dirname(__file__), '..', 'hadamard_668.csv')
        export_csv(H, path)
        print(f"SAVED to {path}")
    
    return H, valid


def main():
    print("="*70)
    print("H(668) SEARCH - Parallel Tempering with DFT-guided moves")
    print("="*70)
    
    # Run search
    best_seqs, best_cost, best_psd, log_entries = parallel_tempering(
        n_replicas=16,
        n_iterations=10_000_000,
        exchange_interval=50,
        log_interval=500_000,
        seed=42
    )
    
    # Save results
    os.makedirs("results/experiments", exist_ok=True)
    results = {
        "method": "parallel_tempering",
        "n_replicas": 16,
        "n_iterations": 10_000_000,
        "best_l2_cost": float(best_cost),
        "best_linf": float(compute_linf(best_psd)),
        "psd_at_best": best_psd.tolist(),
        "log": log_entries
    }
    
    with open("results/experiments/pt_search_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # If exact solution found, build and verify
    if best_cost < 0.5:
        build_and_verify(best_seqs)
        return True
    
    # Save best candidate sequences
    np.savez("results/experiments/best_candidate.npz",
             a=best_seqs[0], b=best_seqs[1], c=best_seqs[2], d=best_seqs[3])
    
    print(f"\nBest candidate saved. PSD range: [{best_psd.min():.1f}, {best_psd.max():.1f}]")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
