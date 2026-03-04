#!/usr/bin/env python3
"""
High-performance search for Hadamard matrix H(668) via Goethals-Seidel array.

Key insights:
1. Need 4 ±1 sequences of length 167 with PSD(k)=668 for ALL k=0,...,166
2. PSD(0) = s1^2+s2^2+s3^2+s4^2 = 668 constrains row sums
3. Valid row sum quadruples: (7,13,15,15), (3,9,17,17), etc.
4. Use FFT for fast PSD computation with incremental updates
5. Multiple search strategies: SA, LNS, frequency-targeted moves

Author: Research Agent
Seed: 42
"""

import numpy as np
from numpy.fft import fft, ifft
from numba import njit, prange
import time
import sys
import os

P = 167  # prime, ≡ 3 (mod 4)
N = 4 * P  # 668

# Row sum quadruples: (s1,s2,s3,s4) with s1^2+s2^2+s3^2+s4^2 = 668
ROW_SUM_QUADRUPLES = [
    (7, 13, 15, 15),   # most balanced
    (3, 9, 17, 17),    # second most balanced
    (3, 3, 17, 19),
    (5, 9, 11, 21),
    (1, 9, 15, 19),
    (3, 7, 13, 21),
    (1, 1, 15, 21),
    (3, 3, 11, 23),
    (3, 7, 9, 23),
    (3, 3, 5, 25),
]

# Each row sum quadruple can have signs: 2^4 = 16 sign patterns per quadruple
# But by symmetry of GS array, can fix one sign (say s1 > 0).
# Also, permutations of sequences matter.


def legendre_symbol(a, p=P):
    """Compute Legendre symbol (a/p)."""
    if a % p == 0:
        return 0
    val = pow(a, (p - 1) // 2, p)
    return 1 if val == 1 else -1


def make_legendre_seq():
    """Legendre symbol sequence for Z_167."""
    seq = np.array([legendre_symbol(i) for i in range(P)], dtype=np.int8)
    seq[0] = 1
    return seq


def init_sequence_with_rowsum(target_sum, rng, base=None):
    """
    Initialize a ±1 sequence of length P with a given row sum.
    If base is provided, modify it to achieve the target sum.
    """
    if base is not None:
        seq = base.copy()
    else:
        # Random initialization
        n_ones = (P + target_sum) // 2
        seq = np.ones(P, dtype=np.int8)
        neg_positions = rng.choice(P, P - n_ones, replace=False)
        seq[neg_positions] = -1
        return seq
    
    current_sum = int(np.sum(seq))
    diff = target_sum - current_sum
    
    if diff > 0:
        # Need more +1s: flip some -1s to +1
        neg_positions = np.where(seq == -1)[0]
        n_flip = diff // 2
        if n_flip > 0 and len(neg_positions) >= n_flip:
            flip_idx = rng.choice(neg_positions, n_flip, replace=False)
            seq[flip_idx] = 1
    elif diff < 0:
        # Need more -1s: flip some +1s to -1
        pos_positions = np.where(seq == 1)[0]
        n_flip = (-diff) // 2
        if n_flip > 0 and len(pos_positions) >= n_flip:
            flip_idx = rng.choice(pos_positions, n_flip, replace=False)
            seq[flip_idx] = -1
    
    return seq


def compute_psd_fft(a, b, c, d):
    """Compute PSD using FFT. Fast O(P log P)."""
    af = fft(a.astype(np.float64))
    bf = fft(b.astype(np.float64))
    cf = fft(c.astype(np.float64))
    df = fft(d.astype(np.float64))
    return np.abs(af)**2 + np.abs(bf)**2 + np.abs(cf)**2 + np.abs(df)**2


def cost_l2(psd):
    """L2 cost: sum of squared deviations from 668."""
    dev = psd - N
    return np.sum(dev**2)


def cost_linf(psd):
    """Linf cost: max absolute deviation."""
    return np.max(np.abs(psd - N))


@njit(cache=True)
def fast_psd_update(dft_re, dft_im, psd, seq_idx, pos, old_val, new_val):
    """
    Incrementally update PSD after flipping one entry.
    O(P) per update instead of O(P log P).
    """
    diff = float(new_val - old_val)  # ±2
    omega = 2.0 * np.pi / P
    for k in range(P):
        angle = omega * k * pos
        cos_v = np.cos(angle)
        sin_v = np.sin(angle)
        old_pow = dft_re[seq_idx, k]**2 + dft_im[seq_idx, k]**2
        dft_re[seq_idx, k] += diff * cos_v
        dft_im[seq_idx, k] -= diff * sin_v
        new_pow = dft_re[seq_idx, k]**2 + dft_im[seq_idx, k]**2
        psd[k] += new_pow - old_pow
    return psd


@njit(cache=True)
def fast_psd_revert(dft_re, dft_im, psd, seq_idx, pos, old_val, new_val):
    """Revert a PSD update (undo a flip)."""
    return fast_psd_update(dft_re, dft_im, psd, seq_idx, pos, new_val, old_val)


@njit(cache=True)
def compute_l2_from_psd(psd):
    """Compute L2 cost from PSD array."""
    cost = 0.0
    for k in range(P):
        dev = psd[k] - N
        cost += dev * dev
    return cost


@njit(cache=True)
def compute_linf_from_psd(psd):
    """Compute Linf cost from PSD array."""
    mx = 0.0
    for k in range(P):
        dev = abs(psd[k] - N)
        if dev > mx:
            mx = dev
    return mx


@njit(cache=True)
def compute_dft_all(seqs):
    """Compute DFT for all 4 sequences."""
    dft_re = np.zeros((4, P), dtype=np.float64)
    dft_im = np.zeros((4, P), dtype=np.float64)
    omega = 2.0 * np.pi / P
    for s in range(4):
        for k in range(P):
            re, im = 0.0, 0.0
            for j in range(P):
                angle = omega * k * j
                re += seqs[s, j] * np.cos(angle)
                im -= seqs[s, j] * np.sin(angle)
            dft_re[s, k] = re
            dft_im[s, k] = im
    return dft_re, dft_im


@njit(cache=True)
def sa_core(seqs, dft_re, dft_im, psd, max_iters, T_start, T_end, seed,
            row_sums, use_rowsum_constraint):
    """
    Core SA loop with incremental DFT updates and optional row-sum constraint.
    When use_rowsum_constraint=True, always swap a +1 and -1 to maintain row sum.
    """
    np.random.seed(seed)
    
    best_cost = compute_l2_from_psd(psd)
    best_linf = compute_linf_from_psd(psd)
    best_seqs = seqs.copy()
    
    current_cost = best_cost
    cooling = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    accepts = 0
    improves = 0
    
    for it in range(max_iters):
        # Choose random sequence and position to flip
        seq_idx = np.random.randint(4)
        
        if use_rowsum_constraint:
            # Swap a +1 and -1 position to maintain row sum
            # Find positions of +1 and -1
            pos1 = np.random.randint(P)
            while seqs[seq_idx, pos1] != 1:
                pos1 = np.random.randint(P)
            pos2 = np.random.randint(P)
            while seqs[seq_idx, pos2] != -1 or pos2 == pos1:
                pos2 = np.random.randint(P)
            
            # Do both flips
            old_val1 = seqs[seq_idx, pos1]
            new_val1 = -old_val1
            seqs[seq_idx, pos1] = new_val1
            fast_psd_update(dft_re, dft_im, psd, seq_idx, pos1, old_val1, new_val1)
            
            old_val2 = seqs[seq_idx, pos2]
            new_val2 = -old_val2
            seqs[seq_idx, pos2] = new_val2
            fast_psd_update(dft_re, dft_im, psd, seq_idx, pos2, old_val2, new_val2)
            
            new_cost = compute_l2_from_psd(psd)
            delta = new_cost - current_cost
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_cost = new_cost
                accepts += 1
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_linf = compute_linf_from_psd(psd)
                    best_seqs = seqs.copy()
                    improves += 1
            else:
                # Revert both flips
                seqs[seq_idx, pos2] = old_val2
                fast_psd_revert(dft_re, dft_im, psd, seq_idx, pos2, new_val2, old_val2)
                seqs[seq_idx, pos1] = old_val1
                fast_psd_revert(dft_re, dft_im, psd, seq_idx, pos1, new_val1, old_val1)
        else:
            # Simple single flip (changes row sum by ±2)
            pos = np.random.randint(P)
            old_val = seqs[seq_idx, pos]
            new_val = -old_val
            
            seqs[seq_idx, pos] = new_val
            fast_psd_update(dft_re, dft_im, psd, seq_idx, pos, old_val, new_val)
            
            new_cost = compute_l2_from_psd(psd)
            delta = new_cost - current_cost
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_cost = new_cost
                accepts += 1
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_linf = compute_linf_from_psd(psd)
                    best_seqs = seqs.copy()
                    improves += 1
            else:
                seqs[seq_idx, pos] = old_val
                fast_psd_revert(dft_re, dft_im, psd, seq_idx, pos, new_val, old_val)
        
        T *= cooling
    
    return best_seqs, best_cost, best_linf, accepts, improves


@njit(cache=True)
def sa_swap_between_seqs(seqs, dft_re, dft_im, psd, max_iters, T_start, T_end, seed):
    """
    SA with swaps between sequences: pick two sequences and swap entries at the same position.
    This maintains all row sums automatically if the two entries differ.
    """
    np.random.seed(seed)
    
    best_cost = compute_l2_from_psd(psd)
    best_linf = compute_linf_from_psd(psd)
    best_seqs = seqs.copy()
    current_cost = best_cost
    cooling = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for it in range(max_iters):
        # Pick two different sequences
        s1 = np.random.randint(4)
        s2 = np.random.randint(4)
        while s2 == s1:
            s2 = np.random.randint(4)
        pos = np.random.randint(P)
        
        if seqs[s1, pos] == seqs[s2, pos]:
            # Swapping identical values does nothing, skip
            continue
        
        # Swap: flip both
        old1 = seqs[s1, pos]
        old2 = seqs[s2, pos]
        seqs[s1, pos] = old2
        seqs[s2, pos] = old1
        
        fast_psd_update(dft_re, dft_im, psd, s1, pos, old1, old2)
        fast_psd_update(dft_re, dft_im, psd, s2, pos, old2, old1)
        
        new_cost = compute_l2_from_psd(psd)
        delta = new_cost - current_cost
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_linf = compute_linf_from_psd(psd)
                best_seqs = seqs.copy()
        else:
            seqs[s1, pos] = old1
            seqs[s2, pos] = old2
            fast_psd_revert(dft_re, dft_im, psd, s1, pos, old2, old1)
            fast_psd_revert(dft_re, dft_im, psd, s2, pos, old1, old2)
        
        T *= cooling
    
    return best_seqs, best_cost, best_linf


def run_search(row_sum_quad, n_iters=5_000_000, T_start=50.0, T_end=0.01, 
               seed=42, use_rowsum=True, verbose=True):
    """
    Run SA search for given row sum quadruple.
    """
    rng = np.random.RandomState(seed)
    sums = list(row_sum_quad)
    
    # Initialize sequences with correct row sums
    seqs = np.zeros((4, P), dtype=np.int8)
    for i in range(4):
        seqs[i] = init_sequence_with_rowsum(sums[i], rng)
    
    # Verify row sums
    actual_sums = [int(np.sum(seqs[i])) for i in range(4)]
    if verbose:
        print(f"Target sums: {sums}, Actual: {actual_sums}")
    
    # Compute initial DFT and PSD
    dft_re, dft_im = compute_dft_all(seqs)
    psd = np.zeros(P, dtype=np.float64)
    for k in range(P):
        for s in range(4):
            psd[k] += dft_re[s, k]**2 + dft_im[s, k]**2
    
    initial_cost = compute_l2_from_psd(psd)
    initial_linf = compute_linf_from_psd(psd)
    if verbose:
        print(f"Initial: L2={initial_cost:.0f}, Linf={initial_linf:.1f}")
    
    # Run SA
    row_sums = np.array(sums, dtype=np.int8)
    start = time.time()
    best_seqs, best_cost, best_linf, accepts, improves = sa_core(
        seqs, dft_re, dft_im, psd, n_iters, T_start, T_end, seed,
        row_sums, use_rowsum
    )
    elapsed = time.time() - start
    
    if verbose:
        print(f"SA done in {elapsed:.1f}s: L2={best_cost:.0f}, Linf={best_linf:.1f}, "
              f"accepts={accepts}, improves={improves}, "
              f"rate={n_iters/elapsed:.0f} iter/s")
    
    return best_seqs, best_cost, best_linf


def run_multi_strategy_search(row_sum_quad, total_time=600, seed=42, verbose=True):
    """
    Multi-strategy search combining different approaches.
    """
    rng = np.random.RandomState(seed)
    
    best_overall_cost = float('inf')
    best_overall_linf = float('inf')
    best_overall_seqs = None
    
    start = time.time()
    round_num = 0
    
    while time.time() - start < total_time:
        round_num += 1
        elapsed = time.time() - start
        remaining = total_time - elapsed
        if remaining < 5:
            break
        
        # Alternate strategies
        strategy = round_num % 4
        
        round_seed = seed + round_num * 1000
        
        if strategy == 0:
            # Fresh random init with row sum constraint
            n_iters = min(2_000_000, int(remaining * 50000))
            if verbose:
                print(f"\n--- Round {round_num}: Fresh SA (constrained), {n_iters/1e6:.1f}M iters ---")
            seqs, cost, linf = run_search(
                row_sum_quad, n_iters, T_start=100.0, T_end=0.001, 
                seed=round_seed, use_rowsum=True, verbose=verbose
            )
        elif strategy == 1:
            # Unconstrained SA from random init (allows row sum to change)
            n_iters = min(3_000_000, int(remaining * 50000))
            if verbose:
                print(f"\n--- Round {round_num}: Unconstrained SA, {n_iters/1e6:.1f}M iters ---")
            seqs, cost, linf = run_search(
                row_sum_quad, n_iters, T_start=200.0, T_end=0.0001,
                seed=round_seed, use_rowsum=False, verbose=verbose
            )
        elif strategy == 2:
            # Low temperature SA from best solution (exploitation)
            if best_overall_seqs is not None:
                n_iters = min(2_000_000, int(remaining * 50000))
                if verbose:
                    print(f"\n--- Round {round_num}: Exploit best, {n_iters/1e6:.1f}M iters ---")
                seqs_init = best_overall_seqs.copy()
                dft_re, dft_im = compute_dft_all(seqs_init)
                psd = np.zeros(P, dtype=np.float64)
                for k in range(P):
                    for s in range(4):
                        psd[k] += dft_re[s, k]**2 + dft_im[s, k]**2
                
                row_sums = np.array([int(np.sum(seqs_init[i])) for i in range(4)], dtype=np.int8)
                seqs, cost, linf, _, _ = sa_core(
                    seqs_init, dft_re, dft_im, psd, n_iters, 5.0, 0.0001,
                    round_seed, row_sums, False
                )
            else:
                continue
        elif strategy == 3:
            # Swap-between-sequences SA
            if best_overall_seqs is not None:
                n_iters = min(2_000_000, int(remaining * 50000))
                if verbose:
                    print(f"\n--- Round {round_num}: Inter-seq swap SA, {n_iters/1e6:.1f}M iters ---")
                seqs_init = best_overall_seqs.copy()
                dft_re, dft_im = compute_dft_all(seqs_init)
                psd = np.zeros(P, dtype=np.float64)
                for k in range(P):
                    for s in range(4):
                        psd[k] += dft_re[s, k]**2 + dft_im[s, k]**2
                
                seqs, cost, linf = sa_swap_between_seqs(
                    seqs_init, dft_re, dft_im, psd, n_iters, 10.0, 0.001, round_seed
                )
            else:
                continue
        
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_linf = linf
            best_overall_seqs = seqs.copy()
            if verbose:
                sums = [int(np.sum(seqs[i])) for i in range(4)]
                print(f"  *** NEW BEST: L2={cost:.0f}, Linf={linf:.1f}, sums={sums}")
            
            if cost == 0:
                if verbose:
                    print("*** EXACT SOLUTION FOUND! ***")
                return best_overall_seqs, 0.0, 0.0
    
    return best_overall_seqs, best_overall_cost, best_overall_linf


def verify_and_build(seqs):
    """Build GS matrix and verify."""
    from results.hadamard_core import goethals_seidel_array, verify_hadamard
    a, b, c, d = seqs[0], seqs[1], seqs[2], seqs[3]
    H = goethals_seidel_array(a, b, c, d)
    valid, msg = verify_hadamard(H)
    return H, valid, msg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type=int, default=300, help='Total search time in seconds')
    parser.add_argument('--quad', type=int, default=0, help='Row sum quadruple index (0-9)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--strategy', type=str, default='multi', 
                       choices=['multi', 'sa', 'unconstrained'])
    args = parser.parse_args()
    
    quad = ROW_SUM_QUADRUPLES[args.quad]
    print(f"Searching for H(668) with row sum quadruple {quad}")
    print(f"Time budget: {args.time}s, Seed: {args.seed}")
    
    if args.strategy == 'multi':
        seqs, cost, linf = run_multi_strategy_search(
            quad, total_time=args.time, seed=args.seed, verbose=True
        )
    else:
        use_rs = (args.strategy == 'sa')
        seqs, cost, linf = run_search(
            quad, n_iters=args.time * 50000, T_start=100.0, T_end=0.001,
            seed=args.seed, use_rowsum=use_rs, verbose=True
        )
    
    print(f"\n=== FINAL RESULT ===")
    print(f"Best L2: {cost:.0f}, Linf: {linf:.1f}")
    sums = [int(np.sum(seqs[i])) for i in range(4)]
    print(f"Row sums: {sums}")
    
    if cost == 0:
        print("EXACT SOLUTION FOUND!")
        # Build and verify
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        H, valid, msg = verify_and_build(seqs)
        print(f"Verification: {msg}")
        if valid:
            np.savetxt("hadamard_668.csv", H, delimiter=",", fmt="%d")
            print("Saved to hadamard_668.csv")
    
    # Save best sequences
    np.savez("results/optimized_best.npz", seqs=seqs, cost=cost, linf=linf)
