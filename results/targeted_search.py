#!/usr/bin/env python3
"""
Targeted search for H(668) Goethals-Seidel sequences.

Key mathematical constraints:
  - 4 sequences a,b,c,d of length 167, entries in {-1,+1}
  - |A(k)|^2 + |B(k)|^2 + |C(k)|^2 + |D(k)|^2 = 668 for k=0,1,...,166
  - For k=0: s_a^2 + s_b^2 + s_c^2 + s_d^2 = 668 where s_x = sum(x)
  - For k>0: spectral flatness condition

Approach: Initialize with correct row sums, then optimize spectral condition
using SA with single-bit flips that preserve row sums (swap +1 and -1 positions).
"""

import numpy as np
from numpy.fft import fft
import numba as nb
from numba import njit
import time
import sys

P = 167
N = 4 * P  # 668

# Row sum decompositions: s1^2 + s2^2 + s3^2 + s4^2 = 668, all odd
ROW_SUM_DECOMPS = [
    (1, 1, 15, 21), (1, 9, 15, 19), (3, 3, 5, 25),
    (3, 3, 11, 23), (3, 3, 17, 19), (3, 7, 9, 23),
    (3, 7, 13, 21), (3, 9, 17, 17), (5, 9, 11, 21),
    (7, 13, 15, 15)
]


@njit(cache=True)
def precompute_dft_matrix(p):
    W = np.empty((p, p), dtype=np.complex128)
    for j in range(p):
        for k in range(p):
            angle = -2.0 * np.pi * j * k / p
            W[j, k] = np.cos(angle) + 1j * np.sin(angle)
    return W


@njit(cache=True)
def compute_fft_all(seqs, p, W):
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
    psd = np.zeros(p, dtype=np.float64)
    for s in range(4):
        for k in range(p):
            psd[k] += F[s, k].real**2 + F[s, k].imag**2
    return psd


@njit(cache=True)
def compute_cost_l2(psd, p, target):
    cost = 0.0
    for k in range(1, p):
        d = psd[k] - target
        cost += d * d
    return cost


@njit(cache=True)
def compute_cost_linf(psd, p, target):
    cost = 0.0
    for k in range(1, p):
        d = abs(psd[k] - target)
        if d > cost:
            cost = d
    return cost


@njit(cache=True)
def apply_flip(F, psd, seqs, s, j, old_val, p, W):
    """Apply flip and update F and PSD."""
    seqs[s, j] = -old_val
    for k in range(p):
        Fsk = F[s, k]
        change = -2.0 * old_val * W[j, k]
        new_Fsk = Fsk + change
        old_power = Fsk.real**2 + Fsk.imag**2
        new_power = new_Fsk.real**2 + new_Fsk.imag**2
        F[s, k] = new_Fsk
        psd[k] += new_power - old_power


@njit(cache=True)
def delta_cost_swap(F, psd, s, j1, j2, p, target, W):
    """
    Compute cost change for swapping entries seqs[s][j1] and seqs[s][j2].
    This preserves the row sum if seqs[s][j1] != seqs[s][j2].
    Equivalent to flipping both j1 and j2.
    """
    v1 = -1  # placeholder, will be set from seqs
    v2 = 1   # placeholder
    delta = 0.0
    
    # Temporary: compute new PSD after both flips
    for k in range(1, p):
        Fsk = F[s, k]
        old_power = Fsk.real**2 + Fsk.imag**2
        
        # After flipping j1 (val v1->-v1) and j2 (val v2->-v2)
        change = -2.0 * v1 * W[j1, k] - 2.0 * v2 * W[j2, k]
        new_Fsk = Fsk + change
        new_power = new_Fsk.real**2 + new_Fsk.imag**2
        
        old_dev = psd[k] - target
        new_dev = old_dev + (new_power - old_power)
        delta += new_dev**2 - old_dev**2
    
    return delta


@njit(cache=True)
def delta_cost_flip_l2(F, psd, s, j, old_val, p, target, W):
    """Compute L2 cost change for flipping seqs[s][j]."""
    delta = 0.0
    for k in range(1, p):
        Fsk = F[s, k]
        change = -2.0 * old_val * W[j, k]
        new_Fsk = Fsk + change
        old_power = Fsk.real**2 + Fsk.imag**2
        new_power = new_Fsk.real**2 + new_Fsk.imag**2
        
        old_dev = psd[k] - target
        new_dev = old_dev + (new_power - old_power)
        delta += new_dev**2 - old_dev**2
    return delta


@njit(cache=True)
def delta_cost_double_flip(F, psd, s1, j1, v1, s2, j2, v2, p, target, W):
    """Cost change for flipping seqs[s1][j1] AND seqs[s2][j2] simultaneously."""
    delta = 0.0
    for k in range(1, p):
        old_dev = psd[k] - target
        
        # Changes from both flips
        change1 = -2.0 * v1 * W[j1, k]
        change2 = -2.0 * v2 * W[j2, k]
        
        if s1 == s2:
            Fsk = F[s1, k]
            old_power = Fsk.real**2 + Fsk.imag**2
            new_Fsk = Fsk + change1 + change2
            new_power = new_Fsk.real**2 + new_Fsk.imag**2
            delta_psd = new_power - old_power
        else:
            Fs1k = F[s1, k]
            Fs2k = F[s2, k]
            old_p1 = Fs1k.real**2 + Fs1k.imag**2
            old_p2 = Fs2k.real**2 + Fs2k.imag**2
            new_Fs1k = Fs1k + change1
            new_Fs2k = Fs2k + change2
            new_p1 = new_Fs1k.real**2 + new_Fs1k.imag**2
            new_p2 = new_Fs2k.real**2 + new_Fs2k.imag**2
            delta_psd = (new_p1 - old_p1) + (new_p2 - old_p2)
        
        new_dev = old_dev + delta_psd
        delta += new_dev**2 - old_dev**2
    return delta


@njit(cache=True)
def sa_rowsum_preserving(seqs, W, p, target, max_iter, T_init, T_min, alpha, seed):
    """
    SA with row-sum-preserving swaps.
    Each move swaps a +1 and a -1 in the same sequence.
    """
    np.random.seed(seed)
    
    F = compute_fft_all(seqs, p, W)
    psd = compute_psd_from_fft(F, p)
    cost = compute_cost_l2(psd, p, target)
    
    best_cost = cost
    best_seqs = seqs.copy()
    
    # Precompute positive and negative indices for each sequence
    # We'll track these dynamically
    
    T = T_init
    
    for it in range(max_iter):
        s = np.random.randint(0, 4)
        
        # Pick a +1 position and a -1 position to swap
        j1 = np.random.randint(0, p)
        j2 = np.random.randint(0, p)
        while j2 == j1 or seqs[s, j1] == seqs[s, j2]:
            j1 = np.random.randint(0, p)
            j2 = np.random.randint(0, p)
        
        v1 = seqs[s, j1]
        v2 = seqs[s, j2]
        
        # Cost of flipping both j1 and j2 in sequence s
        dc = delta_cost_double_flip(F, psd, s, j1, v1, s, j2, v2, p, target, W)
        
        if dc <= 0 or (T > 1e-10 and np.random.random() < np.exp(-dc / T)):
            # Apply both flips
            apply_flip(F, psd, seqs, s, j1, v1, p, W)
            apply_flip(F, psd, seqs, s, j2, v2, p, W)
            cost += dc
            
            if cost < best_cost:
                best_cost = cost
                for si in range(4):
                    for ji in range(p):
                        best_seqs[si, ji] = seqs[si, ji]
                
                if best_cost < 1.0:
                    return best_cost, best_seqs, it + 1
        
        T = max(T * alpha, T_min)
    
    return best_cost, best_seqs, max_iter


@njit(cache=True)
def sa_free_flips(seqs, W, p, target, max_iter, T_init, T_min, alpha, seed):
    """SA with unrestricted single-bit flips (does NOT preserve row sums)."""
    np.random.seed(seed)
    
    F = compute_fft_all(seqs, p, W)
    psd = compute_psd_from_fft(F, p)
    
    # Cost = L2 at k>0 + weight * (PSD(0) - target)^2
    cost_k0 = (psd[0] - target)**2
    cost_rest = compute_cost_l2(psd, p, target)
    cost = cost_rest + cost_k0
    
    best_cost = cost
    best_seqs = seqs.copy()
    
    T = T_init
    
    for it in range(max_iter):
        s = np.random.randint(0, 4)
        j = np.random.randint(0, p)
        old_val = seqs[s, j]
        
        # Compute delta cost including k=0
        dc = 0.0
        for k in range(p):
            Fsk = F[s, k]
            change = -2.0 * old_val * W[j, k]
            new_Fsk = Fsk + change
            old_power = Fsk.real**2 + Fsk.imag**2
            new_power = new_Fsk.real**2 + new_Fsk.imag**2
            
            old_dev = psd[k] - target
            new_dev = old_dev + (new_power - old_power)
            dc += new_dev**2 - old_dev**2
        
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
        
        T = max(T * alpha, T_min)
    
    return best_cost, best_seqs, max_iter


@njit(cache=True)
def sa_multi_flip(seqs, W, p, target, max_iter, T_init, T_min, alpha, seed, n_flips=2):
    """SA with multi-bit flips: flip n_flips random entries per move."""
    np.random.seed(seed)
    
    F = compute_fft_all(seqs, p, W)
    psd = compute_psd_from_fft(F, p)
    cost = compute_cost_l2(psd, p, target)
    # Include k=0
    cost += (psd[0] - target)**2
    
    best_cost = cost
    best_seqs = seqs.copy()
    
    T = T_init
    
    for it in range(max_iter):
        # Save state for potential rollback
        saved_flips = np.empty((n_flips, 3), dtype=np.int64)  # (s, j, old_val)
        
        dc_total = 0.0
        
        for f in range(n_flips):
            s = np.random.randint(0, 4)
            j = np.random.randint(0, p)
            old_val = seqs[s, j]
            saved_flips[f, 0] = s
            saved_flips[f, 1] = j
            saved_flips[f, 2] = old_val
            
            # Compute cost change and apply
            dc = 0.0
            for k in range(p):
                Fsk = F[s, k]
                change = -2.0 * old_val * W[j, k]
                new_Fsk = Fsk + change
                old_power = Fsk.real**2 + Fsk.imag**2
                new_power = new_Fsk.real**2 + new_Fsk.imag**2
                old_dev = psd[k] - target
                new_dev = old_dev + (new_power - old_power)
                dc += new_dev**2 - old_dev**2
            
            apply_flip(F, psd, seqs, s, j, old_val, p, W)
            dc_total += dc
        
        if dc_total <= 0 or (T > 1e-10 and np.random.random() < np.exp(-dc_total / T)):
            cost += dc_total
            if cost < best_cost:
                best_cost = cost
                for si in range(4):
                    for ji in range(p):
                        best_seqs[si, ji] = seqs[si, ji]
                if best_cost < 1.0:
                    return best_cost, best_seqs, it + 1
        else:
            # Rollback all flips in reverse
            for f in range(n_flips - 1, -1, -1):
                s = saved_flips[f, 0]
                j = saved_flips[f, 1]
                # Current val is -old_val, so flip back
                cur_val = seqs[s, j]
                apply_flip(F, psd, seqs, s, j, cur_val, p, W)
        
        T = max(T * alpha, T_min)
    
    return best_cost, best_seqs, max_iter


def make_init_seqs_with_rowsums(target_sums, rng):
    """Create ±1 sequences with specified row sums."""
    seqs = np.ones((4, P), dtype=np.int8)
    for s in range(4):
        ts = target_sums[s]
        # We need (P + ts) / 2 entries to be +1 and (P - ts) / 2 to be -1
        n_pos = (P + ts) // 2
        n_neg = P - n_pos
        seq = np.ones(P, dtype=np.int8)
        neg_idx = rng.choice(P, size=n_neg, replace=False)
        seq[neg_idx] = -1
        seqs[s] = seq
    return seqs


def run_targeted_search(time_budget=300, n_starts=100):
    """Main search loop with targeted initializations."""
    print(f"=== Targeted H(668) Search ===")
    print(f"P={P}, N={N}")
    print(f"Time budget: {time_budget}s, starts: {n_starts}")
    print()
    
    W = precompute_dft_matrix(P)
    
    # Warm up
    print("JIT warmup...", flush=True)
    dummy = np.ones((4, P), dtype=np.int8)
    sa_free_flips(dummy, W, P, N, 100, 10.0, 0.01, 0.999, 0)
    sa_rowsum_preserving(dummy, W, P, N, 100, 10.0, 0.01, 0.999, 0)
    sa_multi_flip(dummy, W, P, N, 100, 10.0, 0.01, 0.999, 0, 2)
    print("Ready.", flush=True)
    
    global_best_cost = float('inf')
    global_best_seqs = None
    
    rng = np.random.default_rng(42)
    start_time = time.time()
    
    for trial in range(n_starts):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        seed = 42 + trial * 7
        
        # Choose initialization strategy
        strategy = trial % 8
        
        if strategy == 0:
            # Random with correct row sums
            decomp_idx = rng.integers(0, len(ROW_SUM_DECOMPS))
            sums = list(ROW_SUM_DECOMPS[decomp_idx])
            # Random signs
            for i in range(4):
                if rng.random() < 0.5:
                    sums[i] = -sums[i]
            rng.shuffle(np.array(sums))
            seqs = make_init_seqs_with_rowsums(sums, rng)
            method = f"rowsum{tuple(sums)}"
            
        elif strategy == 1:
            # Fully random
            seqs = (2 * rng.integers(0, 2, size=(4, P)) - 1).astype(np.int8)
            method = "random"
            
        elif strategy in (2, 3):
            # Random with large row sums  
            decomp = ROW_SUM_DECOMPS[rng.integers(0, len(ROW_SUM_DECOMPS))]
            sums = list(decomp)
            for i in range(4):
                if rng.random() < 0.5:
                    sums[i] = -sums[i]
            seqs = make_init_seqs_with_rowsums(sums, rng)
            method = f"rowsum_lg"
            
        elif strategy in (4, 5):
            # Legendre-based with perturbation to get correct row sums
            from hadamard_core import legendre_symbol
            chi = np.array([legendre_symbol(i, P) if i != 0 else 1 for i in range(P)], dtype=np.int8)
            seqs = np.array([chi.copy() for _ in range(4)], dtype=np.int8)
            # Perturb heavily
            for s in range(4):
                n_flips = rng.integers(20, 60)
                idx = rng.choice(P, size=n_flips, replace=False)
                seqs[s, idx] = -seqs[s, idx]
            method = "legendre_perturbed"
            
        else:
            # Near-Legendre: keep structure but vary across sequences
            from hadamard_core import legendre_symbol
            chi = np.array([legendre_symbol(i, P) if i != 0 else 1 for i in range(P)], dtype=np.int8)
            seqs = np.array([chi.copy() for _ in range(4)], dtype=np.int8)
            # Flip different random sets in each sequence
            for s in range(4):
                n_flips = rng.integers(5, 40)
                idx = rng.choice(P, size=n_flips, replace=False)
                seqs[s, idx] = -seqs[s, idx]
            method = "near_legendre"
        
        # Run SA - alternate between methods
        sa_method = trial % 3
        iters = 2000000
        T_init = 20.0 + rng.random() * 180.0
        alpha = 0.999990 + rng.random() * 0.000009
        
        if sa_method == 0:
            cost, best_s, n_it = sa_free_flips(seqs.copy(), W, P, N, iters, T_init, 0.001, alpha, seed)
        elif sa_method == 1:
            cost, best_s, n_it = sa_multi_flip(seqs.copy(), W, P, N, iters, T_init, 0.001, alpha, seed, 2)
        else:
            cost, best_s, n_it = sa_multi_flip(seqs.copy(), W, P, N, iters, T_init, 0.001, alpha, seed, 3)
        
        if cost < global_best_cost:
            global_best_cost = cost
            global_best_seqs = best_s.copy()
            psd = np.zeros(P)
            F_tmp = fft(best_s.astype(np.float64), axis=1)
            for s in range(4):
                psd += np.abs(F_tmp[s])**2
            linf = np.max(np.abs(psd[1:] - N))
            elapsed = time.time() - start_time
            print(f"  Trial {trial:3d} ({method:20s}): L2={cost:.0f}, Linf={linf:.1f}, T0={T_init:.1f}, alpha={alpha:.6f} [{elapsed:.1f}s]")
            
            if cost < 1.0:
                print("\n*** EXACT SOLUTION FOUND! ***")
                return best_s, True
        
        if (trial + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  [{elapsed:.1f}s] {trial+1}/{n_starts} trials, best L2={global_best_cost:.0f}")
    
    elapsed = time.time() - start_time
    print(f"\nDone. Time: {elapsed:.1f}s, Best L2: {global_best_cost:.0f}")
    
    return global_best_seqs, False


if __name__ == "__main__":
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 240
    starts = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    best_seqs, found = run_targeted_search(time_budget=budget, n_starts=starts)
    
    if best_seqs is not None:
        np.savez('results/targeted_best.npz', sequences=best_seqs)
        
        if found:
            print("\nBuilding 668x668 Hadamard matrix...")
            from hadamard_core import goethals_seidel_array, verify_hadamard, export_csv
            H = goethals_seidel_array(best_seqs[0], best_seqs[1], best_seqs[2], best_seqs[3])
            valid, msg = verify_hadamard(H)
            print(f"Verification: {msg}")
            if valid:
                export_csv(H, 'hadamard_668.csv')
                print("Saved to hadamard_668.csv!")
