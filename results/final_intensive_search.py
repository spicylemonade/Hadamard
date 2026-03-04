#!/usr/bin/env python3
"""
Final intensive search for H(668).

This is the last-resort computational attack. Strategy:
1. Use Numba-compiled SA with incremental DFT updates
2. Diverse random restarts from multiple initialization strategies
3. Focus on minimizing the FULL cost (all frequencies including k=0)
4. Track and save the global best continuously
"""

import numpy as np
from numpy.fft import fft
from numba import njit
import time
import sys
import os

P = 167
N = 668
SEED = 42


@njit(cache=True)
def precompute_W(p):
    W = np.empty((p, p), dtype=np.complex128)
    for j in range(p):
        for k in range(p):
            angle = -2.0 * np.pi * j * k / p
            W[j, k] = np.cos(angle) + 1j * np.sin(angle)
    return W


@njit(cache=True)
def compute_fft(seqs, p, W):
    F = np.empty((4, p), dtype=np.complex128)
    for s in range(4):
        for k in range(p):
            val = 0.0 + 0.0j
            for j in range(p):
                val += seqs[s, j] * W[j, k]
            F[s, k] = val
    return F


@njit(cache=True)
def compute_psd(F, p):
    psd = np.zeros(p, dtype=np.float64)
    for s in range(4):
        for k in range(p):
            psd[k] += F[s, k].real**2 + F[s, k].imag**2
    return psd


@njit(cache=True)
def full_cost(psd, p, target):
    """Cost including ALL frequencies k=0..p-1."""
    c = 0.0
    for k in range(p):
        d = psd[k] - target
        c += d * d
    return c


@njit(cache=True)
def apply_flip_update(F, psd, seqs, s, j, old_val, p, W):
    seqs[s, j] = -old_val
    for k in range(p):
        Fsk = F[s, k]
        change = -2.0 * old_val * W[j, k]
        new_Fsk = Fsk + change
        old_pow = Fsk.real**2 + Fsk.imag**2
        new_pow = new_Fsk.real**2 + new_Fsk.imag**2
        F[s, k] = new_Fsk
        psd[k] += new_pow - old_pow


@njit(cache=True)
def delta_full_cost(F, psd, s, j, old_val, p, target, W):
    dc = 0.0
    for k in range(p):
        Fsk = F[s, k]
        change = -2.0 * old_val * W[j, k]
        new_Fsk = Fsk + change
        old_pow = Fsk.real**2 + Fsk.imag**2
        new_pow = new_Fsk.real**2 + new_Fsk.imag**2
        old_dev = psd[k] - target
        new_dev = old_dev + (new_pow - old_pow)
        dc += new_dev * new_dev - old_dev * old_dev
    return dc


@njit(cache=True)
def run_sa(seqs, W, p, target, n_iter, T0, T_min, alpha, seed):
    np.random.seed(seed)
    
    F = compute_fft(seqs, p, W)
    psd = compute_psd(F, p)
    cost = full_cost(psd, p, target)
    
    best_cost = cost
    best_seqs = seqs.copy()
    
    T = T0
    n_accept = 0
    
    for it in range(n_iter):
        s = np.random.randint(0, 4)
        j = np.random.randint(0, p)
        ov = seqs[s, j]
        
        dc = delta_full_cost(F, psd, s, j, ov, p, target, W)
        
        if dc <= 0.0 or (T > 1e-12 and np.random.random() < np.exp(-dc / T)):
            apply_flip_update(F, psd, seqs, s, j, ov, p, W)
            cost += dc
            n_accept += 1
            
            if cost < best_cost:
                best_cost = cost
                for si in range(4):
                    for ji in range(p):
                        best_seqs[si, ji] = seqs[si, ji]
                if best_cost < 0.5:
                    return best_cost, best_seqs, it + 1, n_accept
        
        T = max(T * alpha, T_min)
    
    return best_cost, best_seqs, n_iter, n_accept


@njit(cache=True)
def run_sa_reheating(seqs, W, p, target, n_iter, T0, T_min, alpha, 
                     reheat_interval, reheat_factor, seed):
    """SA with periodic reheating to escape local minima."""
    np.random.seed(seed)
    
    F = compute_fft(seqs, p, W)
    psd = compute_psd(F, p)
    cost = full_cost(psd, p, target)
    
    best_cost = cost
    best_seqs = seqs.copy()
    
    T = T0
    
    for it in range(n_iter):
        s = np.random.randint(0, 4)
        j = np.random.randint(0, p)
        ov = seqs[s, j]
        
        dc = delta_full_cost(F, psd, s, j, ov, p, target, W)
        
        if dc <= 0.0 or (T > 1e-12 and np.random.random() < np.exp(-dc / T)):
            apply_flip_update(F, psd, seqs, s, j, ov, p, W)
            cost += dc
            
            if cost < best_cost:
                best_cost = cost
                for si in range(4):
                    for ji in range(p):
                        best_seqs[si, ji] = seqs[si, ji]
                if best_cost < 0.5:
                    return best_cost, best_seqs, it + 1
        
        T = max(T * alpha, T_min)
        
        # Reheat periodically
        if (it + 1) % reheat_interval == 0:
            T = min(T * reheat_factor, T0 * 0.5)
    
    return best_cost, best_seqs, n_iter


def make_legendre():
    seq = np.zeros(P, dtype=np.int8)
    for i in range(P):
        if i == 0:
            seq[i] = 1
        else:
            seq[i] = 1 if pow(int(i), (P-1)//2, P) == 1 else -1
    return seq


def main():
    time_budget = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    
    print(f"=== FINAL INTENSIVE SEARCH FOR H(668) ===")
    print(f"Time budget: {time_budget}s")
    print(f"P={P}, target PSD={N} at all {P} frequencies")
    print()
    
    W = precompute_W(P)
    
    # Warm up
    print("Compiling...", flush=True)
    dummy = np.ones((4, P), dtype=np.int8)
    run_sa(dummy, W, P, N, 100, 10.0, 0.01, 0.999, 0)
    run_sa_reheating(dummy, W, P, N, 100, 10.0, 0.01, 0.999, 50, 10.0, 0)
    print("Ready.", flush=True)
    
    chi = make_legendre()
    rng = np.random.default_rng(SEED)
    
    global_best = float('inf')
    global_best_seqs = None
    
    start = time.time()
    trial = 0
    
    while time.time() - start < time_budget:
        remaining = time_budget - (time.time() - start)
        if remaining < 2:
            break
        
        seed = SEED + trial * 13 + 7
        
        # Diverse initializations
        strategy = trial % 10
        
        if strategy == 0:
            # Pure random
            seqs = (2 * rng.integers(0, 2, size=(4, P)) - 1).astype(np.int8)
            label = "random"
        elif strategy == 1:
            # Legendre copies, heavily perturbed
            seqs = np.array([chi.copy() for _ in range(4)], dtype=np.int8)
            for s in range(4):
                n_flip = rng.integers(30, 70)
                idx = rng.choice(P, size=n_flip, replace=False)
                seqs[s, idx] = -seqs[s, idx]
            label = "leg_heavy"
        elif strategy == 2:
            # Mixed: some Legendre, some random
            seqs = np.empty((4, P), dtype=np.int8)
            for s in range(4):
                if rng.random() < 0.5:
                    seqs[s] = chi.copy()
                    n_flip = rng.integers(10, 40)
                    idx = rng.choice(P, size=n_flip, replace=False)
                    seqs[s, idx] = -seqs[s, idx]
                else:
                    seqs[s] = (2 * rng.integers(0, 2, size=P) - 1).astype(np.int8)
            label = "mixed"
        elif strategy in (3, 4):
            # Target specific row sums
            decomps = [
                (3, 3, 5, 25), (3, 7, 13, 21), (5, 9, 11, 21),
                (1, 9, 15, 19), (3, 9, 17, 17), (7, 13, 15, 15),
                (3, 3, 17, 19), (3, 7, 9, 23), (1, 1, 15, 21),
                (3, 3, 11, 23),
            ]
            d = decomps[rng.integers(0, len(decomps))]
            sums = list(d)
            for i in range(4):
                if rng.random() < 0.5:
                    sums[i] = -sums[i]
            perm = rng.permutation(4)
            seqs = np.empty((4, P), dtype=np.int8)
            for s in range(4):
                ts = sums[perm[s]]
                n_pos = (P + ts) // 2
                seq = -np.ones(P, dtype=np.int8)
                if n_pos > 0:
                    pos_idx = rng.choice(P, size=min(n_pos, P), replace=False)
                    seq[pos_idx] = 1
                seqs[s] = seq
            label = f"rowsum{tuple(sums)}"
        elif strategy in (5, 6):
            # Legendre with different chi(0) values
            seqs = np.empty((4, P), dtype=np.int8)
            for s in range(4):
                seqs[s] = chi.copy()
                if rng.random() < 0.5:
                    seqs[s, 0] = -1
                # Small random perturbation
                n_flip = rng.integers(5, 25)
                idx = rng.choice(range(1, P), size=n_flip, replace=False)
                seqs[s, idx] = -seqs[s, idx]
            label = "leg_var"
        elif strategy == 7:
            # Start from previous best if available
            if global_best_seqs is not None:
                seqs = global_best_seqs.copy()
                for s in range(4):
                    n_flip = rng.integers(10, 50)
                    idx = rng.choice(P, size=n_flip, replace=False)
                    seqs[s, idx] = -seqs[s, idx]
                label = "perturb_best"
            else:
                seqs = (2 * rng.integers(0, 2, size=(4, P)) - 1).astype(np.int8)
                label = "random"
        elif strategy == 8:
            # Symmetric sequences (Williamson-type)
            seqs = np.empty((4, P), dtype=np.int8)
            for s in range(4):
                half = rng.choice([-1, 1], size=(P+1)//2)
                seqs[s, 0] = half[0]
                for i in range(1, (P+1)//2):
                    seqs[s, i] = half[i]
                    seqs[s, P - i] = half[i]
            label = "symmetric"
        else:
            # All negative with sparse positives
            seqs = -np.ones((4, P), dtype=np.int8)
            for s in range(4):
                n_pos = rng.integers(60, 100)
                idx = rng.choice(P, size=n_pos, replace=False)
                seqs[s, idx] = 1
            label = "sparse"
        
        # Choose SA parameters
        iters_per = min(3000000, int(remaining * 200000))
        T0 = 10.0 + rng.random() * 190.0
        alpha = 0.999990 + rng.random() * 0.000009
        
        # Alternate between standard SA and reheating SA
        if trial % 3 == 0:
            cost, best_s, n_it, n_acc = run_sa(
                seqs.copy(), W, P, N, iters_per, T0, 0.001, alpha, seed
            )
        else:
            cost, best_s, n_it = run_sa_reheating(
                seqs.copy(), W, P, N, iters_per, T0, 0.001, alpha,
                200000, 50.0, seed
            )
        
        if cost < global_best:
            global_best = cost
            global_best_seqs = best_s.copy()
            
            # Compute detailed stats
            F_tmp = np.zeros((4, P), dtype=np.complex128)
            for s in range(4):
                F_tmp[s] = fft(best_s[s].astype(np.float64))
            psd = np.zeros(P)
            for s in range(4):
                psd += np.abs(F_tmp[s])**2
            linf = np.max(np.abs(psd - N))
            linf_nz = np.max(np.abs(psd[1:] - N))
            
            elapsed = time.time() - start
            print(f"  Trial {trial:4d} ({label:16s}): BEST cost={cost:.0f}, "
                  f"Linf={linf:.1f}, Linf(k>0)={linf_nz:.1f}, "
                  f"T0={T0:.1f}, iters={n_it:.0f} [{elapsed:.1f}s]")
            
            if cost < 0.5:
                print("\n!!! EXACT SOLUTION FOUND !!!")
                np.savez('results/SOLUTION.npz', sequences=best_s, psd=psd)
                return best_s
        
        if (trial + 1) % 20 == 0:
            elapsed = time.time() - start
            print(f"  [{elapsed:.0f}s] {trial+1} trials done, best={global_best:.0f}")
        
        trial += 1
    
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Search complete: {trial} trials in {elapsed:.1f}s")
    print(f"Global best full cost: {global_best:.0f}")
    
    if global_best_seqs is not None:
        F_tmp = np.zeros((4, P), dtype=np.complex128)
        for s in range(4):
            F_tmp[s] = fft(global_best_seqs[s].astype(np.float64))
        psd = np.zeros(P)
        for s in range(4):
            psd += np.abs(F_tmp[s])**2
        
        print(f"PSD(0) = {psd[0]:.1f} (target {N})")
        print(f"PSD range k>0: [{psd[1:].min():.1f}, {psd[1:].max():.1f}]")
        print(f"Linf (all k): {np.max(np.abs(psd - N)):.1f}")
        print(f"Linf (k>0): {np.max(np.abs(psd[1:] - N)):.1f}")
        
        np.savez('results/final_best.npz', sequences=global_best_seqs, psd=psd)
    
    return global_best_seqs


if __name__ == "__main__":
    best = main()
