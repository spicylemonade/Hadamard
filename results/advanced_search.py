#!/usr/bin/env python3
"""
Advanced multi-strategy search for Hadamard matrix H(668).
Uses Goethals-Seidel array: find four ±1 sequences of length 167
whose PSD sums to 668 at all non-zero frequencies.

Strategies:
1. Numba-accelerated SA with incremental DFT updates
2. Frequency-targeted guided moves
3. Multi-sequence simultaneous flips
4. Orbit-aware search (Z_167 symmetry)
5. Legendre-perturbed initialization
6. Large neighborhood search (LNS)
"""

import numpy as np
from numba import njit, prange
import time
import sys
import os

P = 167
N = 4 * P  # 668

# ============================================================
# Core functions with Numba JIT
# ============================================================

@njit(cache=True)
def legendre_seq():
    """Compute Legendre symbol sequence for Z_167."""
    seq = np.ones(P, dtype=np.int8)
    for a in range(1, P):
        val = pow(a, (P - 1) // 2, P)
        if val == P - 1:
            seq[a] = -1
    seq[0] = 1  # Convention: chi(0) = 1
    return seq

@njit(cache=True)
def compute_psd(a, b, c, d):
    """Compute PSD at all frequencies using DFT."""
    # Manual DFT computation for exact control
    psd = np.zeros(P, dtype=np.float64)
    omega = 2.0 * np.pi / P
    for k in range(P):
        ar, ai = 0.0, 0.0
        br, bi = 0.0, 0.0
        cr, ci = 0.0, 0.0
        dr, di = 0.0, 0.0
        for j in range(P):
            angle = omega * k * j
            cos_val = np.cos(angle)
            sin_val = np.sin(angle)
            ar += a[j] * cos_val
            ai -= a[j] * sin_val
            br += b[j] * cos_val
            bi -= b[j] * sin_val
            cr += c[j] * cos_val
            ci -= c[j] * sin_val
            dr += d[j] * cos_val
            di -= d[j] * sin_val
        psd[k] = ar*ar + ai*ai + br*br + bi*bi + cr*cr + ci*ci + dr*dr + di*di
    return psd

@njit(cache=True)
def compute_cost_l2(psd):
    """L2 cost: sum of squared deviations from N=668 at non-zero frequencies."""
    cost = 0.0
    for k in range(1, P):
        dev = psd[k] - N
        cost += dev * dev
    return cost

@njit(cache=True)
def compute_cost_linf(psd):
    """Linf cost: max absolute deviation from 668 at non-zero frequencies."""
    mx = 0.0
    for k in range(1, P):
        dev = abs(psd[k] - N)
        if dev > mx:
            mx = dev
    return mx

@njit(cache=True)
def compute_dft_seq(seq):
    """Compute DFT of a single sequence."""
    omega = 2.0 * np.pi / P
    real = np.zeros(P, dtype=np.float64)
    imag = np.zeros(P, dtype=np.float64)
    for k in range(P):
        r, im = 0.0, 0.0
        for j in range(P):
            angle = omega * k * j
            r += seq[j] * np.cos(angle)
            im -= seq[j] * np.sin(angle)
        real[k] = r
        imag[k] = im
    return real, imag

@njit(cache=True)
def incremental_psd_update(dft_real, dft_imag, seq_idx, pos, old_val, new_val, psd):
    """Update PSD after flipping one entry. O(P) update."""
    diff = new_val - old_val  # Either +2 or -2
    omega = 2.0 * np.pi / P
    new_psd = psd.copy()
    for k in range(P):
        angle = omega * k * pos
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)
        # Old power for this sequence at freq k
        old_power = dft_real[seq_idx, k]**2 + dft_imag[seq_idx, k]**2
        # Update DFT
        dft_real[seq_idx, k] += diff * cos_val
        dft_imag[seq_idx, k] -= diff * sin_val
        # New power
        new_power = dft_real[seq_idx, k]**2 + dft_imag[seq_idx, k]**2
        new_psd[k] += new_power - old_power
    return new_psd

@njit(cache=True)
def sa_search_incremental(a0, b0, c0, d0, max_iters, T_start, T_end, seed):
    """
    Simulated annealing with incremental DFT updates.
    Uses geometric cooling schedule.
    """
    np.random.seed(seed)
    
    # Initialize sequences
    a = a0.copy()
    b = b0.copy()
    c = c0.copy()
    d = d0.copy()
    
    seqs = np.zeros((4, P), dtype=np.int8)
    seqs[0] = a
    seqs[1] = b
    seqs[2] = c
    seqs[3] = d
    
    # Compute initial DFTs
    dft_real = np.zeros((4, P), dtype=np.float64)
    dft_imag = np.zeros((4, P), dtype=np.float64)
    for i in range(4):
        r, im = compute_dft_seq(seqs[i])
        dft_real[i] = r
        dft_imag[i] = im
    
    # Initial PSD
    psd = np.zeros(P, dtype=np.float64)
    for k in range(P):
        for i in range(4):
            psd[k] += dft_real[i, k]**2 + dft_imag[i, k]**2
    
    cost = compute_cost_l2(psd)
    best_cost = cost
    best_seqs = seqs.copy()
    best_linf = compute_cost_linf(psd)
    
    cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    accepts = 0
    
    for iteration in range(max_iters):
        # Pick random sequence and position
        seq_idx = np.random.randint(0, 4)
        pos = np.random.randint(0, P)
        
        old_val = seqs[seq_idx, pos]
        new_val = -old_val
        
        # Incremental update
        new_psd = incremental_psd_update(dft_real, dft_imag, seq_idx, pos, old_val, new_val, psd)
        new_cost = compute_cost_l2(new_psd)
        
        delta = new_cost - cost
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            # Accept
            seqs[seq_idx, pos] = new_val
            psd = new_psd
            cost = new_cost
            accepts += 1
            
            if cost < best_cost:
                best_cost = cost
                best_seqs = seqs.copy()
                best_linf = compute_cost_linf(psd)
                
                if best_cost < 1e-6:  # Found exact solution!
                    return best_seqs, best_cost, best_linf, iteration
        else:
            # Reject - revert DFT
            incremental_psd_update(dft_real, dft_imag, seq_idx, pos, new_val, old_val, psd)
        
        T *= cooling_rate
    
    return best_seqs, best_cost, best_linf, max_iters


@njit(cache=True)
def sa_search_guided(a0, b0, c0, d0, max_iters, T_start, T_end, seed):
    """
    SA with DFT-guided moves: preferentially flip positions that reduce 
    the worst-frequency deviation.
    """
    np.random.seed(seed)
    
    seqs = np.zeros((4, P), dtype=np.int8)
    seqs[0] = a0.copy()
    seqs[1] = b0.copy()
    seqs[2] = c0.copy()
    seqs[3] = d0.copy()
    
    # Compute initial DFTs
    dft_real = np.zeros((4, P), dtype=np.float64)
    dft_imag = np.zeros((4, P), dtype=np.float64)
    for i in range(4):
        r, im = compute_dft_seq(seqs[i])
        dft_real[i] = r
        dft_imag[i] = im
    
    psd = np.zeros(P, dtype=np.float64)
    for k in range(P):
        for i in range(4):
            psd[k] += dft_real[i, k]**2 + dft_imag[i, k]**2
    
    cost = compute_cost_l2(psd)
    best_cost = cost
    best_seqs = seqs.copy()
    best_linf = compute_cost_linf(psd)
    
    cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for iteration in range(max_iters):
        guide_prob = 0.3  # 30% guided, 70% random
        
        if np.random.random() < guide_prob:
            # Find worst frequency
            worst_k = 1
            worst_dev = 0.0
            for k in range(1, P):
                dev = abs(psd[k] - N)
                if dev > worst_dev:
                    worst_dev = dev
                    worst_k = k
            
            # Find best flip to reduce this frequency
            best_delta = 1e30
            best_seq = 0
            best_pos = 0
            
            # Sample a few candidates
            for _ in range(10):
                si = np.random.randint(0, 4)
                pi = np.random.randint(0, P)
                old_val = seqs[si, pi]
                diff = -2 * old_val
                angle = 2.0 * np.pi * worst_k * pi / P
                cos_val = np.cos(angle)
                sin_val = np.sin(angle)
                
                old_power = dft_real[si, worst_k]**2 + dft_imag[si, worst_k]**2
                new_r = dft_real[si, worst_k] + diff * cos_val
                new_i = dft_imag[si, worst_k] - diff * sin_val
                new_power = new_r**2 + new_i**2
                
                new_psd_k = psd[worst_k] - old_power + new_power
                delta_dev = abs(new_psd_k - N) - abs(psd[worst_k] - N)
                
                if delta_dev < best_delta:
                    best_delta = delta_dev
                    best_seq = si
                    best_pos = pi
            
            seq_idx = best_seq
            pos = best_pos
        else:
            seq_idx = np.random.randint(0, 4)
            pos = np.random.randint(0, P)
        
        old_val = seqs[seq_idx, pos]
        new_val = -old_val
        
        new_psd = incremental_psd_update(dft_real, dft_imag, seq_idx, pos, old_val, new_val, psd)
        new_cost = compute_cost_l2(new_psd)
        
        delta = new_cost - cost
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            seqs[seq_idx, pos] = new_val
            psd = new_psd
            cost = new_cost
            
            if cost < best_cost:
                best_cost = cost
                best_seqs = seqs.copy()
                best_linf = compute_cost_linf(psd)
                
                if best_cost < 1e-6:
                    return best_seqs, best_cost, best_linf, iteration
        else:
            incremental_psd_update(dft_real, dft_imag, seq_idx, pos, new_val, old_val, psd)
        
        T *= cooling_rate
    
    return best_seqs, best_cost, best_linf, max_iters


@njit(cache=True)
def multi_flip_sa(a0, b0, c0, d0, max_iters, T_start, T_end, seed, max_flips=3):
    """
    SA with 1-3 simultaneous flips per step.
    More exploration power but noisier.
    """
    np.random.seed(seed)
    
    seqs = np.zeros((4, P), dtype=np.int8)
    seqs[0] = a0.copy()
    seqs[1] = b0.copy()
    seqs[2] = c0.copy()
    seqs[3] = d0.copy()
    
    psd = compute_psd(seqs[0], seqs[1], seqs[2], seqs[3])
    cost = compute_cost_l2(psd)
    best_cost = cost
    best_seqs = seqs.copy()
    best_linf = compute_cost_linf(psd)
    
    cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for iteration in range(max_iters):
        # Choose number of flips (1, 2, or 3)
        n_flips = np.random.randint(1, max_flips + 1)
        
        # Store flip info for potential reversal
        flip_info = np.zeros((n_flips, 2), dtype=np.int64)
        
        for f in range(n_flips):
            si = np.random.randint(0, 4)
            pi = np.random.randint(0, P)
            flip_info[f, 0] = si
            flip_info[f, 1] = pi
            seqs[si, pi] = -seqs[si, pi]
        
        new_psd = compute_psd(seqs[0], seqs[1], seqs[2], seqs[3])
        new_cost = compute_cost_l2(new_psd)
        
        delta = new_cost - cost
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            psd = new_psd
            cost = new_cost
            
            if cost < best_cost:
                best_cost = cost
                best_seqs = seqs.copy()
                best_linf = compute_cost_linf(psd)
                
                if best_cost < 1e-6:
                    return best_seqs, best_cost, best_linf, iteration
        else:
            # Revert flips
            for f in range(n_flips):
                si = flip_info[f, 0]
                pi = flip_info[f, 1]
                seqs[si, pi] = -seqs[si, pi]
        
        T *= cooling_rate
    
    return best_seqs, best_cost, best_linf, max_iters


@njit(cache=True)
def symmetric_sa(a0, b0, c0, d0, max_iters, T_start, T_end, seed):
    """
    SA restricted to symmetric sequences: a[j] = a[P-j].
    This is the Williamson constraint. Reduces search space by ~half.
    """
    np.random.seed(seed)
    
    seqs = np.zeros((4, P), dtype=np.int8)
    seqs[0] = a0.copy()
    seqs[1] = b0.copy()
    seqs[2] = c0.copy()
    seqs[3] = d0.copy()
    
    # Enforce symmetry
    half = (P - 1) // 2  # 83
    for i in range(4):
        for j in range(1, half + 1):
            seqs[i, P - j] = seqs[i, j]
    
    psd = compute_psd(seqs[0], seqs[1], seqs[2], seqs[3])
    cost = compute_cost_l2(psd)
    best_cost = cost
    best_seqs = seqs.copy()
    best_linf = compute_cost_linf(psd)
    
    cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for iteration in range(max_iters):
        seq_idx = np.random.randint(0, 4)
        
        # Choose: flip a[0] (free), or flip orbit {j, P-j}
        choice = np.random.randint(0, half + 1)
        
        if choice == 0:
            # Flip a[0]
            old_val = seqs[seq_idx, 0]
            seqs[seq_idx, 0] = -old_val
        else:
            # Flip orbit {choice, P-choice}
            old_val = seqs[seq_idx, choice]
            seqs[seq_idx, choice] = -old_val
            seqs[seq_idx, P - choice] = -old_val
        
        new_psd = compute_psd(seqs[0], seqs[1], seqs[2], seqs[3])
        new_cost = compute_cost_l2(new_psd)
        
        delta = new_cost - cost
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            psd = new_psd
            cost = new_cost
            
            if cost < best_cost:
                best_cost = cost
                best_seqs = seqs.copy()
                best_linf = compute_cost_linf(psd)
                
                if best_cost < 1e-6:
                    return best_seqs, best_cost, best_linf, iteration
        else:
            # Revert
            if choice == 0:
                seqs[seq_idx, 0] = -seqs[seq_idx, 0]
            else:
                seqs[seq_idx, choice] = -seqs[seq_idx, choice]
                seqs[seq_idx, P - choice] = -seqs[seq_idx, choice]
        
        T *= cooling_rate
    
    return best_seqs, best_cost, best_linf, max_iters


def generate_initializations(n_starts, seed=42):
    """Generate diverse initial sequences."""
    np.random.seed(seed)
    leg = legendre_seq()
    inits = []
    
    for i in range(n_starts):
        init_type = i % 6
        
        if init_type == 0:
            # Legendre baseline
            a, b, c, d = leg.copy(), leg.copy(), leg.copy(), leg.copy()
        elif init_type == 1:
            # Legendre with random perturbation (5%)
            a, b, c, d = leg.copy(), leg.copy(), leg.copy(), leg.copy()
            for seq in [a, b, c, d]:
                mask = np.random.random(P) < 0.05
                seq[mask] = -seq[mask]
        elif init_type == 2:
            # Legendre with random perturbation (10%)
            a, b, c, d = leg.copy(), leg.copy(), leg.copy(), leg.copy()
            for seq in [a, b, c, d]:
                mask = np.random.random(P) < 0.10
                seq[mask] = -seq[mask]
        elif init_type == 3:
            # Fully random
            a = np.random.choice(np.array([-1, 1], dtype=np.int8), P)
            b = np.random.choice(np.array([-1, 1], dtype=np.int8), P)
            c = np.random.choice(np.array([-1, 1], dtype=np.int8), P)
            d = np.random.choice(np.array([-1, 1], dtype=np.int8), P)
        elif init_type == 4:
            # Mixed: 2 Legendre + 2 negated Legendre
            a, b = leg.copy(), leg.copy()
            c, d = -leg.copy(), -leg.copy()
            for seq in [a, b, c, d]:
                mask = np.random.random(P) < 0.05
                seq[mask] = -seq[mask]
        else:
            # QR-based: use quadratic residue structure differently
            a = leg.copy()
            # Shift the Legendre sequence
            shift = np.random.randint(1, P)
            b = np.roll(leg, shift).copy()
            shift2 = np.random.randint(1, P)
            c = np.roll(leg, shift2).copy()
            d = np.random.choice(np.array([-1, 1], dtype=np.int8), P)
        
        inits.append((a, b, c, d, i * 1000 + seed))
    
    return inits


def run_parallel_search(strategy, n_starts=20, max_iters=5_000_000, 
                        T_start=1000.0, T_end=0.01):
    """Run multiple search instances and return best result."""
    inits = generate_initializations(n_starts)
    
    best_overall_cost = 1e30
    best_overall_seqs = None
    best_overall_linf = 1e30
    
    for idx, (a, b, c, d, seed) in enumerate(inits):
        t0 = time.time()
        
        if strategy == "incremental":
            seqs, cost, linf, iters = sa_search_incremental(
                a, b, c, d, max_iters, T_start, T_end, seed)
        elif strategy == "guided":
            seqs, cost, linf, iters = sa_search_guided(
                a, b, c, d, max_iters, T_start, T_end, seed)
        elif strategy == "multi_flip":
            seqs, cost, linf, iters = multi_flip_sa(
                a, b, c, d, max_iters, T_start, T_end, seed)
        elif strategy == "symmetric":
            seqs, cost, linf, iters = symmetric_sa(
                a, b, c, d, max_iters, T_start, T_end, seed)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        elapsed = time.time() - t0
        
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_seqs = seqs.copy()
            best_overall_linf = linf
        
        print(f"  Start {idx}: L2={cost:.1f}, Linf={linf:.1f}, "
              f"iters={iters}, time={elapsed:.1f}s")
        
        if cost < 1e-6:
            print("  *** EXACT SOLUTION FOUND! ***")
            return seqs, cost, linf
    
    return best_overall_seqs, best_overall_cost, best_overall_linf


def verify_and_export(seqs, filename="hadamard_668.csv"):
    """Build GS matrix from sequences and verify."""
    from hadamard_core import goethals_seidel_array, verify_hadamard, export_csv
    
    a, b, c, d = seqs[0], seqs[1], seqs[2], seqs[3]
    H = goethals_seidel_array(a, b, c, d)
    valid, msg = verify_hadamard(H)
    
    if valid:
        export_csv(H, filename)
        print(f"VALID Hadamard matrix exported to {filename}")
    else:
        print(f"Not a valid Hadamard matrix: {msg}")
    
    return valid, msg, H


if __name__ == "__main__":
    print("="*60)
    print("Advanced Multi-Strategy Search for H(668)")
    print("="*60)
    
    # First, warm up JIT compilation with a tiny run
    print("\nWarmup (JIT compilation)...")
    leg = legendre_seq()
    seqs, cost, linf, _ = sa_search_incremental(
        leg, leg.copy(), leg.copy(), leg.copy(), 100, 100.0, 0.1, 42)
    print(f"  Warmup complete. Legendre baseline L2={cost:.1f}")
    
    strategies = ["incremental", "guided", "multi_flip"]
    n_starts = 10
    max_iters = 2_000_000
    
    overall_best_cost = 1e30
    overall_best_seqs = None
    overall_best_linf = 1e30
    
    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        print(f"  {n_starts} starts, {max_iters} iterations each")
        
        seqs, cost, linf = run_parallel_search(
            strategy, n_starts=n_starts, max_iters=max_iters,
            T_start=500.0, T_end=0.001)
        
        print(f"  Best: L2={cost:.1f}, Linf={linf:.1f}")
        
        if cost < overall_best_cost:
            overall_best_cost = cost
            overall_best_seqs = seqs.copy()
            overall_best_linf = linf
        
        if cost < 1e-6:
            print("\n*** EXACT SOLUTION FOUND! ***")
            break
    
    print(f"\n{'='*60}")
    print(f"Overall best: L2={overall_best_cost:.1f}, Linf={overall_best_linf:.1f}")
    
    if overall_best_cost < 1e-6:
        # Save the solution
        np.savez("results/solution_sequences.npz",
                 a=overall_best_seqs[0], b=overall_best_seqs[1],
                 c=overall_best_seqs[2], d=overall_best_seqs[3])
        verify_and_export(overall_best_seqs)
    else:
        print("No exact solution found. Saving best candidate...")
        np.savez("results/best_candidate_seqs.npz",
                 a=overall_best_seqs[0], b=overall_best_seqs[1],
                 c=overall_best_seqs[2], d=overall_best_seqs[3],
                 l2_cost=overall_best_cost, linf=overall_best_linf)
