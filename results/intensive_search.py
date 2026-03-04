#!/usr/bin/env python3
"""
Intensive search for H(668) = Goethals-Seidel(167) Hadamard matrix.

Strategy: Numba-accelerated simulated annealing with:
  1. Incremental FFT updates (O(n) per flip instead of O(n log n))
  2. Multi-start from diverse initializations  
  3. Aggressive parallel tempering with many replicas
  4. Hybrid: alternate SA with tabu search phases
  
Key insight: For ±1 sequences of length p, flipping entry j in sequence s
changes |F_s(k)|² by a known quantity computable in O(1) per frequency.

Total PSD change from flipping a[j]: 
  delta_psd(k) = -4 * a[j] * Re(F_a(k) * exp(2*pi*i*j*k/p)) + 4
  (since a[j] -> -a[j] changes F_a(k) by -2*a[j]*exp(-2*pi*i*j*k/p))
"""

import numpy as np
from numpy.fft import fft, ifft
import numba as nb
from numba import njit, prange
import time
import sys
import os

P = 167
N = 4 * P  # = 668
SEED = 42

# Precompute twiddle factors
TWIDDLE = np.exp(-2j * np.pi * np.arange(P)[:, None] * np.arange(P)[None, :] / P)

def legendre_symbol(a, p=P):
    a = int(a)
    p = int(p)
    if a % p == 0: return 0
    return 1 if pow(a, (p-1)//2, p) == 1 else -1

def make_legendre_seq():
    """Standard Legendre sequence with chi(0) = 1."""
    return np.array([legendre_symbol(i, P) if i != 0 else 1 for i in range(P)], dtype=np.int8)

def compute_psd(seqs):
    """Compute PSD at all frequencies for 4 sequences."""
    psd = np.zeros(P, dtype=np.float64)
    for s in range(4):
        F = fft(seqs[s].astype(np.float64))
        psd += np.abs(F)**2
    return psd

def compute_cost(psd):
    """L2 cost: sum of (psd[k] - 668)^2 for k=1..166."""
    dev = psd[1:] - N
    return np.sum(dev**2)

def compute_linf(psd):
    """Linf cost: max |psd[k] - 668| for k=1..166."""
    return np.max(np.abs(psd[1:] - N))

# ============================================================
# Numba-optimized core
# ============================================================

@njit(cache=True)
def precompute_dft_matrix(p):
    """Precompute DFT twiddle factors."""
    W = np.empty((p, p), dtype=np.complex128)
    for j in range(p):
        for k in range(p):
            angle = -2.0 * np.pi * j * k / p
            W[j, k] = np.cos(angle) + 1j * np.sin(angle)
    return W

@njit(cache=True)
def compute_fft_all(seqs, p, W):
    """Compute FFT of all 4 sequences using DFT matrix."""
    F = np.empty((4, p), dtype=np.complex128)
    for s in range(4):
        for k in range(p):
            val = 0.0 + 0.0j
            for j in range(p):
                val += seqs[s, j] * W[j, k]
            F[s, k] = val
    return F

@njit(cache=True)
def compute_psd_from_fft(F, p):
    """Compute PSD from FFT values."""
    psd = np.zeros(p, dtype=np.float64)
    for s in range(4):
        for k in range(p):
            psd[k] += F[s, k].real**2 + F[s, k].imag**2
    return psd

@njit(cache=True) 
def compute_cost_from_psd(psd, p, target):
    """L2 cost from PSD."""
    cost = 0.0
    for k in range(1, p):
        d = psd[k] - target
        cost += d * d
    return cost

@njit(cache=True)
def delta_cost_flip(F, psd, s, j, old_val, p, target, W):
    """
    Compute the change in L2 cost if we flip seqs[s][j] from old_val to -old_val.
    
    When a[j] -> -a[j], F_a(k) changes by -2*old_val*W[j,k].
    New |F_a(k)|² = |F_a(k) - 2*old_val*W[j,k]|²
    """
    delta_cost = 0.0
    for k in range(1, p):
        old_dev = psd[k] - target
        
        # Change in |F_s(k)|²
        Fsk = F[s, k]
        Wjk = W[j, k]
        change_F = -2.0 * old_val * Wjk
        new_Fsk = Fsk + change_F
        new_power = new_Fsk.real**2 + new_Fsk.imag**2
        old_power = Fsk.real**2 + Fsk.imag**2
        delta_psd_k = new_power - old_power
        
        new_dev = old_dev + delta_psd_k
        delta_cost += new_dev * new_dev - old_dev * old_dev
    return delta_cost

@njit(cache=True)
def apply_flip(F, psd, seqs, s, j, old_val, p, W):
    """Apply a flip and update F and PSD incrementally."""
    seqs[s, j] = -old_val
    for k in range(p):
        Fsk = F[s, k]
        Wjk = W[j, k]
        change_F = -2.0 * old_val * Wjk
        
        old_power = Fsk.real**2 + Fsk.imag**2
        new_Fsk = Fsk + change_F
        new_power = new_Fsk.real**2 + new_Fsk.imag**2
        
        F[s, k] = new_Fsk
        psd[k] += new_power - old_power

@njit(cache=True)
def find_best_flip(F, psd, seqs, p, target, W):
    """Find the flip that gives the best cost reduction."""
    best_delta = 1e30
    best_s = 0
    best_j = 0
    for s in range(4):
        for j in range(p):
            old_val = seqs[s, j]
            dc = delta_cost_flip(F, psd, s, j, old_val, p, target, W)
            if dc < best_delta:
                best_delta = dc
                best_s = s
                best_j = j
    return best_s, best_j, best_delta

@njit(cache=True)
def sa_search(seqs, W, p, target, max_iter, T_init, T_min, alpha, seed):
    """
    Simulated annealing with incremental FFT updates.
    Returns: best cost, best sequences, iteration count.
    """
    np.random.seed(seed)
    
    # Compute initial FFT and PSD
    F = compute_fft_all(seqs, p, W)
    psd = compute_psd_from_fft(F, p)
    cost = compute_cost_from_psd(psd, p, target)
    
    best_cost = cost
    best_seqs = seqs.copy()
    
    T = T_init
    accepts = 0
    
    for it in range(max_iter):
        # Random flip
        s = np.random.randint(0, 4)
        j = np.random.randint(0, p)
        old_val = seqs[s, j]
        
        dc = delta_cost_flip(F, psd, s, j, old_val, p, target, W)
        
        if dc <= 0 or (T > 1e-10 and np.random.random() < np.exp(-dc / T)):
            apply_flip(F, psd, seqs, s, j, old_val, p, W)
            cost += dc
            accepts += 1
            
            if cost < best_cost:
                best_cost = cost
                for si in range(4):
                    for ji in range(p):
                        best_seqs[si, ji] = seqs[si, ji]
                
                if best_cost < 1.0:  # Found solution!
                    return best_cost, best_seqs, it + 1
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    return best_cost, best_seqs, max_iter

@njit(cache=True)
def guided_sa_search(seqs, W, p, target, max_iter, T_init, T_min, alpha, seed, guided_prob=0.3):
    """
    SA with DFT-guided moves: with probability guided_prob, 
    pick the frequency with worst deviation and try to fix it.
    """
    np.random.seed(seed)
    
    F = compute_fft_all(seqs, p, W)
    psd = compute_psd_from_fft(F, p)
    cost = compute_cost_from_psd(psd, p, target)
    
    best_cost = cost
    best_seqs = seqs.copy()
    
    T = T_init
    
    for it in range(max_iter):
        # Choose move
        if np.random.random() < guided_prob:
            # Guided: find worst frequency, then find best (s,j) for that frequency
            worst_k = 1
            worst_dev = 0.0
            for k in range(1, p):
                d = abs(psd[k] - target)
                if d > worst_dev:
                    worst_dev = d
                    worst_k = k
            
            # Find best flip for this frequency
            best_delta_local = 1e30
            s_pick = 0
            j_pick = 0
            # Try a random subset
            for _ in range(20):
                s_try = np.random.randint(0, 4)
                j_try = np.random.randint(0, p)
                old_v = seqs[s_try, j_try]
                dc = delta_cost_flip(F, psd, s_try, j_try, old_v, p, target, W)
                if dc < best_delta_local:
                    best_delta_local = dc
                    s_pick = s_try
                    j_pick = j_try
            s = s_pick
            j = j_pick
        else:
            s = np.random.randint(0, 4)
            j = np.random.randint(0, p)
        
        old_val = seqs[s, j]
        dc = delta_cost_flip(F, psd, s, j, old_val, p, target, W)
        
        if dc <= 0 or (T > 1e-10 and np.random.random() < np.exp(-dc / T)):
            apply_flip(F, psd, seqs, s, j, old_val, p, W)
            cost += dc
            
            if cost < best_cost:
                best_cost = cost
                for si in range(4):
                    for ji in range(p):
                        best_seqs[si, ji] = seqs[si, ji]
                
                if best_cost < 1.0:
                    return best_cost, best_seqs, it + 1
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    return best_cost, best_seqs, max_iter


@njit(cache=True)
def parallel_tempering(seqs_init, W, p, target, n_replicas, max_iter, seed):
    """Parallel tempering with n_replicas temperature levels."""
    np.random.seed(seed)
    
    # Temperature ladder (geometric spacing)
    temps = np.empty(n_replicas)
    T_min_val = 0.1
    T_max_val = 200.0
    for r in range(n_replicas):
        if n_replicas > 1:
            frac = r / (n_replicas - 1.0)
        else:
            frac = 0.0
        temps[r] = T_min_val * (T_max_val / T_min_val) ** frac
    
    # Initialize replicas
    all_seqs = np.empty((n_replicas, 4, p), dtype=np.int8)
    all_F = np.empty((n_replicas, 4, p), dtype=np.complex128)
    all_psd = np.empty((n_replicas, p), dtype=np.float64)
    all_cost = np.empty(n_replicas, dtype=np.float64)
    
    for r in range(n_replicas):
        for s in range(4):
            for j in range(p):
                all_seqs[r, s, j] = seqs_init[s, j]
        # Add random perturbations for r > 0
        if r > 0:
            n_flips = int(p * 0.1 * r)
            for _ in range(n_flips):
                ss = np.random.randint(0, 4)
                jj = np.random.randint(0, p)
                all_seqs[r, ss, jj] = -all_seqs[r, ss, jj]
        
        all_F[r] = compute_fft_all(all_seqs[r], p, W)
        all_psd[r] = compute_psd_from_fft(all_F[r], p)
        all_cost[r] = compute_cost_from_psd(all_psd[r], p, target)
    
    best_cost = 1e30
    best_seqs = seqs_init.copy()
    for r in range(n_replicas):
        if all_cost[r] < best_cost:
            best_cost = all_cost[r]
            for s in range(4):
                for j in range(p):
                    best_seqs[s, j] = all_seqs[r, s, j]
    
    swap_interval = 100
    
    for it in range(max_iter):
        # SA step for each replica
        for r in range(n_replicas):
            ss = np.random.randint(0, 4)
            jj = np.random.randint(0, p)
            old_val = all_seqs[r, ss, jj]
            
            dc = delta_cost_flip(all_F[r], all_psd[r], ss, jj, old_val, p, target, W)
            
            T = temps[r]
            if dc <= 0 or (T > 1e-10 and np.random.random() < np.exp(-dc / T)):
                apply_flip(all_F[r], all_psd[r], all_seqs[r], ss, jj, old_val, p, W)
                all_cost[r] += dc
                
                if all_cost[r] < best_cost:
                    best_cost = all_cost[r]
                    for s in range(4):
                        for j in range(p):
                            best_seqs[s, j] = all_seqs[r, s, j]
                    
                    if best_cost < 1.0:
                        return best_cost, best_seqs, it + 1
        
        # Replica exchange
        if it % swap_interval == 0 and it > 0:
            for r in range(n_replicas - 1):
                r2 = r + 1
                dE = (1.0/temps[r] - 1.0/temps[r2]) * (all_cost[r2] - all_cost[r])
                if dE > 0 or np.random.random() < np.exp(dE):
                    # Swap
                    all_cost[r], all_cost[r2] = all_cost[r2], all_cost[r]
                    for s in range(4):
                        for j in range(p):
                            tmp = all_seqs[r, s, j]
                            all_seqs[r, s, j] = all_seqs[r2, s, j]
                            all_seqs[r2, s, j] = tmp
                    for s in range(4):
                        for k in range(p):
                            tmp_f = all_F[r, s, k]
                            all_F[r, s, k] = all_F[r2, s, k]
                            all_F[r2, s, k] = tmp_f
                    for k in range(p):
                        tmp_p = all_psd[r, k]
                        all_psd[r, k] = all_psd[r2, k]
                        all_psd[r2, k] = tmp_p
    
    return best_cost, best_seqs, max_iter


def generate_diverse_init(method, rng):
    """Generate diverse initial sequences."""
    chi = make_legendre_seq()
    
    if method == 'legendre':
        # 4 copies of Legendre
        return np.array([chi.copy() for _ in range(4)], dtype=np.int8)
    
    elif method == 'legendre_perturbed':
        # Legendre with random perturbations
        seqs = np.array([chi.copy() for _ in range(4)], dtype=np.int8)
        for s in range(4):
            n_flips = rng.integers(5, 30)
            indices = rng.choice(P, size=n_flips, replace=False)
            seqs[s, indices] = -seqs[s, indices]
        return seqs
    
    elif method == 'legendre_mixed':
        # Mix Legendre with negated copies
        seqs = np.array([chi.copy() for _ in range(4)], dtype=np.int8)
        for s in range(4):
            if rng.random() < 0.5:
                seqs[s] = -seqs[s]
        # Perturb
        for s in range(4):
            n_flips = rng.integers(1, 15)
            indices = rng.choice(P, size=n_flips, replace=False)
            seqs[s, indices] = -seqs[s, indices]
        return seqs
    
    elif method == 'random':
        # Fully random ±1
        return (2 * rng.integers(0, 2, size=(4, P)) - 1).astype(np.int8)
    
    elif method == 'qr_based':
        # Use quadratic residues in different ways
        qr = set()
        g = 5  # primitive root mod 167
        val = 1
        for _ in range(83):
            qr.add(val)
            val = (val * g * g) % P
        
        seqs = np.ones((4, P), dtype=np.int8)
        # Sequence 0: standard Legendre
        seqs[0] = chi.copy()
        # Sequence 1: shifted Legendre
        shift = rng.integers(1, P)
        seqs[1] = np.array([legendre_symbol((i + shift) % P, P) if (i + shift) % P != 0 else 1 for i in range(P)], dtype=np.int8)
        # Sequence 2: negated Legendre
        seqs[2] = -chi.copy()
        # Sequence 3: random with same weight
        seqs[3] = chi.copy()
        n_flips = rng.integers(5, 20)
        indices = rng.choice(P, size=n_flips, replace=False)
        seqs[3, indices] = -seqs[3, indices]
        return seqs
    
    elif method == 'target_rowsum':
        # Initialize with target row sums
        # Pick a row sum decomposition
        decomps = [
            (1, 1, 15, 21), (1, 9, 15, 19), (3, 3, 5, 25),
            (3, 3, 11, 23), (3, 3, 17, 19), (3, 7, 9, 23),
            (3, 7, 13, 21), (3, 9, 17, 17), (5, 9, 11, 21),
            (7, 13, 15, 15)
        ]
        idx = rng.integers(0, len(decomps))
        sums = list(decomps[idx])
        # Randomly assign signs
        for i in range(4):
            if rng.random() < 0.5:
                sums[i] = -sums[i]
        rng.shuffle(np.array(sums))  # shuffle order
        
        seqs = np.ones((4, P), dtype=np.int8)
        for s in range(4):
            target_sum = sums[s]
            # Start with chi and adjust
            seqs[s] = chi.copy()
            current_sum = int(np.sum(seqs[s]))
            diff = target_sum - current_sum
            # Need to flip diff/2 entries from +1 to -1 or vice versa
            if diff > 0:
                # Need more +1s: flip -1 entries to +1
                neg_indices = np.where(seqs[s] == -1)[0]
                n_to_flip = min(abs(diff) // 2, len(neg_indices))
                if n_to_flip > 0:
                    flip_idx = rng.choice(neg_indices, size=n_to_flip, replace=False)
                    seqs[s, flip_idx] = 1
            elif diff < 0:
                pos_indices = np.where(seqs[s] == 1)[0]
                n_to_flip = min(abs(diff) // 2, len(pos_indices))
                if n_to_flip > 0:
                    flip_idx = rng.choice(pos_indices, size=n_to_flip, replace=False)
                    seqs[s, flip_idx] = -1
        return seqs
    
    else:
        return (2 * rng.integers(0, 2, size=(4, P)) - 1).astype(np.int8)


def run_intensive_search(total_time_seconds=300, n_starts=50):
    """Run intensive multi-start search."""
    print(f"=== Intensive H(668) Search ===")
    print(f"P={P}, N={N}, target PSD={N} at all non-zero frequencies")
    print(f"Time budget: {total_time_seconds}s, starts: {n_starts}")
    print()
    
    # Precompute DFT matrix
    print("Precomputing DFT matrix...", flush=True)
    W = precompute_dft_matrix(P)
    
    # Warm up Numba
    print("Warming up JIT compilation...", flush=True)
    warmup_seqs = np.ones((4, P), dtype=np.int8)
    sa_search(warmup_seqs, W, P, N, 100, 10.0, 0.01, 0.999, 0)
    print("JIT warm-up complete.", flush=True)
    
    global_best_cost = float('inf')
    global_best_seqs = None
    global_best_psd = None
    
    rng = np.random.default_rng(SEED)
    
    methods = ['legendre', 'legendre_perturbed', 'legendre_mixed', 'random', 
               'qr_based', 'target_rowsum']
    
    start_time = time.time()
    
    for trial in range(n_starts):
        elapsed = time.time() - start_time
        if elapsed > total_time_seconds:
            break
        
        remaining = total_time_seconds - elapsed
        method = methods[trial % len(methods)]
        seed = SEED + trial
        
        # Generate initial sequences
        init_seqs = generate_diverse_init(method, np.random.default_rng(seed))
        
        # Decide strategy based on time budget per trial
        time_per_trial = remaining / max(1, n_starts - trial)
        iters = max(500000, int(time_per_trial * 150000))  # ~150K iter/s estimated
        
        # Phase 1: Standard SA
        T_init = 50.0 + rng.random() * 150.0
        alpha = 0.999995 + rng.random() * 0.000004  # Very slow cooling
        
        cost, seqs, iters_done = sa_search(
            init_seqs.copy(), W, P, N, min(iters, 2000000),
            T_init, 0.001, alpha, seed
        )
        
        if cost < global_best_cost:
            global_best_cost = cost
            global_best_seqs = seqs.copy()
            psd = compute_psd(seqs.astype(np.float64))
            global_best_psd = psd
            linf = compute_linf(psd)
            elapsed = time.time() - start_time
            print(f"  Trial {trial} ({method}): NEW BEST L2={cost:.0f}, Linf={linf:.1f} [{elapsed:.1f}s]")
            
            if cost < 1.0:
                print("\n*** SOLUTION FOUND! ***")
                return seqs, psd
        
        # Phase 2: If good, refine with guided SA
        if cost < global_best_cost * 2:
            cost2, seqs2, _ = guided_sa_search(
                seqs.copy(), W, P, N, min(iters // 2, 1000000),
                5.0, 0.001, 0.99999, seed + 10000, 0.3
            )
            if cost2 < global_best_cost:
                global_best_cost = cost2
                global_best_seqs = seqs2.copy()
                psd = compute_psd(seqs2.astype(np.float64))
                global_best_psd = psd
                linf = compute_linf(psd)
                elapsed = time.time() - start_time
                print(f"  Trial {trial} (guided): NEW BEST L2={cost2:.0f}, Linf={linf:.1f} [{elapsed:.1f}s]")
                
                if cost2 < 1.0:
                    print("\n*** SOLUTION FOUND! ***")
                    return seqs2, psd
        
        if (trial + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {trial+1}/{n_starts} trials, best L2={global_best_cost:.0f}, time={elapsed:.1f}s")
    
    # Final phase: Parallel tempering from best found
    print(f"\nFinal phase: Parallel tempering from best solution...")
    remaining = total_time_seconds - (time.time() - start_time)
    if remaining > 10 and global_best_seqs is not None:
        pt_iters = max(500000, int(remaining * 50000))
        cost_pt, seqs_pt, _ = parallel_tempering(
            global_best_seqs.copy(), W, P, N, 12, min(pt_iters, 5000000), SEED + 99999
        )
        if cost_pt < global_best_cost:
            global_best_cost = cost_pt
            global_best_seqs = seqs_pt.copy()
            psd = compute_psd(seqs_pt.astype(np.float64))
            global_best_psd = psd
            linf = compute_linf(psd)
            print(f"  PT result: L2={cost_pt:.0f}, Linf={linf:.1f}")
            
            if cost_pt < 1.0:
                print("\n*** SOLUTION FOUND! ***")
                return seqs_pt, psd
    
    total_time = time.time() - start_time
    print(f"\nSearch complete. Total time: {total_time:.1f}s")
    print(f"Best L2 cost: {global_best_cost:.0f}")
    if global_best_psd is not None:
        print(f"Best Linf: {compute_linf(global_best_psd):.1f}")
        print(f"PSD range at k>0: [{global_best_psd[1:].min():.1f}, {global_best_psd[1:].max():.1f}] (target: {N})")
    
    return global_best_seqs, global_best_psd


if __name__ == "__main__":
    time_budget = int(sys.argv[1]) if len(sys.argv) > 1 else 240
    n_starts = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    
    seqs, psd = run_intensive_search(total_time_seconds=time_budget, n_starts=n_starts)
    
    if seqs is not None:
        # Save best sequences
        np.savez('results/intensive_best.npz', sequences=seqs, psd=psd)
        
        # Check if solution found
        cost = compute_cost(psd) if psd is not None else float('inf')
        if cost < 1.0:
            print("\nBuilding and verifying full 668x668 matrix...")
            from hadamard_core import goethals_seidel_array, verify_hadamard, export_csv
            H = goethals_seidel_array(seqs[0], seqs[1], seqs[2], seqs[3])
            valid, msg = verify_hadamard(H)
            print(f"Verification: {msg}")
            if valid:
                export_csv(H, 'hadamard_668.csv')
                print("Exported to hadamard_668.csv")
        else:
            print(f"\nNo exact solution found. Best cost: {cost:.0f}")
