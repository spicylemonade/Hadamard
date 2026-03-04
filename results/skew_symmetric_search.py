#!/usr/bin/env python3
"""
Search for H(668) using mixed skew+symmetric GS sequences.

Key insight: Use Legendre (skew) as sequence a, and search over 3 symmetric
sequences b, c, d. This reduces the search space from 668 to 249 binary variables.

For skew a: a[0]=1, a[j]=-a[P-j] for j>0 (83 free vars, but we FIX it as Legendre)
For symmetric b,c,d: b[j]=b[P-j] for j>0 (84 free vars each: b[0] and 83 pairs)

PSD constraint:
  |a_hat(k)|^2 = 168 for k>0 (fixed, from Legendre)
  |b_hat(k)|^2 + |c_hat(k)|^2 + |d_hat(k)|^2 = 500 for k>0
  sum(a)=1, sum(b)^2+sum(c)^2+sum(d)^2 = 667

Row sum triples for (b,c,d): (1,15,21) or (9,15,19)
"""

import numpy as np
from numpy.fft import fft
from numba import njit
import time
import sys

P = 167
N = 668
TARGET_BCD = 500  # PSD target for b+c+d at non-zero frequencies

def leg_sym(a, p=P):
    if a % p == 0: return 0
    return 1 if pow(a, (p-1)//2, p) == 1 else -1

def make_leg():
    s = np.array([leg_sym(i) for i in range(P)], dtype=np.int8)
    s[0] = 1
    return s

def make_symmetric_seq(half, row_sum_target, rng):
    """
    Create a symmetric ±1 sequence of length P from the first 84 positions.
    half: values for positions 0, 1, ..., 83 (84 values)
    The sequence satisfies seq[j] = seq[P-j] for j > 0.
    """
    seq = np.zeros(P, dtype=np.int8)
    seq[0] = half[0]
    for j in range(1, 84):
        seq[j] = half[j]
        seq[P - j] = half[j]
    # P = 167, so P//2 = 83. Position 84 maps to P-84=83. Wait:
    # Pairs: (1,166), (2,165), ..., (83,84). That's 83 pairs.
    # So positions 0,...,83 determine the whole sequence.
    # seq[j] for j=0: standalone
    # seq[j] for j=1,...,83: paired with seq[P-j] = seq[167-j]
    # 167-1=166, 167-83=84. So seq[84],...,seq[166] are determined.
    return seq

def init_symmetric_with_rowsum(target_sum, rng):
    """
    Initialize a symmetric ±1 sequence with target row sum.
    Row sum of symmetric sequence: seq[0] + 2*sum_{j=1}^{83} seq[j]
    So sum = seq[0] + 2*S where S = sum of seq[1],...,seq[83].
    Target: target_sum = seq[0] + 2*S
    If seq[0] = 1: S = (target_sum - 1) / 2. Need target_sum odd.
    If seq[0] = -1: S = (target_sum + 1) / 2. Need target_sum odd.
    S ranges from -83 to 83.
    """
    # Both seq[0] = 1 and seq[0] = -1 are valid; choose based on S range
    for a0 in [1, -1]:
        S = (target_sum - a0) / 2
        if S != int(S) or abs(S) > 83:
            continue
        S = int(S)
        # S = num_plus - num_minus where num_plus + num_minus = 83
        # num_plus = (83 + S) / 2
        n_plus = (83 + S) // 2
        if n_plus < 0 or n_plus > 83 or (83 + S) % 2 != 0:
            continue
        
        half = np.ones(84, dtype=np.int8)
        half[0] = a0
        # Positions 1-83: n_plus positive, (83-n_plus) negative
        n_minus = 83 - n_plus
        if n_minus > 0:
            neg_idx = rng.choice(np.arange(1, 84), n_minus, replace=False)
            half[neg_idx] = -1
        
        seq = np.zeros(P, dtype=np.int8)
        seq[0] = half[0]
        for j in range(1, 84):
            seq[j] = half[j]
            seq[P - j] = half[j]
        
        actual_sum = int(np.sum(seq))
        assert actual_sum == target_sum, f"Sum mismatch: {actual_sum} != {target_sum}"
        return seq
    
    raise ValueError(f"Cannot achieve row sum {target_sum} with symmetric sequence")


@njit(cache=True)
def build_cos_sin_tables():
    omega = 2.0 * np.pi / P
    cos_t = np.empty((P, P), dtype=np.float64)
    sin_t = np.empty((P, P), dtype=np.float64)
    for k in range(P):
        for j in range(P):
            angle = omega * k * j
            cos_t[k, j] = np.cos(angle)
            sin_t[k, j] = np.sin(angle)
    return cos_t, sin_t


@njit(cache=True)
def compute_bcd_dft(seqs_bcd, cos_t, sin_t):
    """Compute DFT for 3 symmetric sequences (b, c, d)."""
    dft_re = np.zeros((3, P), dtype=np.float64)
    # Symmetric sequences have real DFT, but let's compute it fully
    for s in range(3):
        for k in range(P):
            re = 0.0
            for j in range(P):
                re += seqs_bcd[s, j] * cos_t[k, j]
            dft_re[s, k] = re
    return dft_re


@njit(cache=True)
def psd_bcd(dft_re_bcd):
    """PSD of just b+c+d sequences."""
    psd = np.zeros(P, dtype=np.float64)
    for k in range(P):
        for s in range(3):
            psd[k] += dft_re_bcd[s, k]**2
    return psd


@njit(cache=True)
def l2_bcd(psd, target):
    """L2 cost for b+c+d PSD deviation from target at non-zero frequencies."""
    cost = 0.0
    for k in range(1, P):
        d = psd[k] - target
        cost += d * d
    # Also include k=0: psd_bcd(0) should be 667
    d0 = psd[0] - 667.0
    cost += d0 * d0
    return cost


@njit(cache=True)
def linf_bcd(psd, target):
    mx = 0.0
    for k in range(1, P):
        d = abs(psd[k] - target)
        if d > mx:
            mx = d
    return mx


@njit(cache=True)
def sa_symmetric(seqs_bcd, cos_t, sin_t, max_iters, T_start, T_end, seed):
    """
    SA for 3 symmetric sequences.
    Moves: swap a symmetric pair (positions j and P-j) in one sequence.
    For position 0: just flip it.
    This maintains the symmetric structure.
    Row sum changes by ±2 (for pos 0 flip) or ±4 (for pair flip).
    To maintain row sum, we do TWO pair swaps: one +1 pair → -1, one -1 pair → +1.
    """
    np.random.seed(seed)
    
    dft_re = compute_bcd_dft(seqs_bcd, cos_t, sin_t)
    psd = psd_bcd(dft_re)
    
    current_cost = l2_bcd(psd, TARGET_BCD)
    best_cost = current_cost
    best_linf = linf_bcd(psd, TARGET_BCD)
    best_seqs = seqs_bcd.copy()
    
    log_ratio = np.log(T_end / T_start)
    inv_max = 1.0 / max_iters
    
    # Build position lists for symmetric pairs
    # Each "orbit" is either position 0 (singleton) or pair (j, P-j) for j=1,...,83
    # Total orbits: 84 per sequence, 252 total
    
    for it in range(max_iters):
        T = T_start * np.exp(log_ratio * it * inv_max)
        
        s = np.random.randint(3)
        
        # Pick two orbits in sequence s: one positive, one negative, and swap
        # For symmetric sequence, orbit j means positions j and P-j have same value.
        # "Positive orbit" means seq[j] = 1 (and seq[P-j] = 1 if j>0).
        # "Negative orbit" means seq[j] = -1.
        
        # Method: pick random orbit to flip (changes row sum by ±2 or ±4)
        # Then find another orbit to flip in opposite direction to maintain row sum.
        
        # Simple approach: pick two orbits with different signs and swap both.
        # For j=0: flip changes sum by 2. For j>0: flip changes sum by 4.
        
        # Fastest: pick random j1 in 0..83 where seq is +1, and j2 where seq is -1.
        # Swap both. If both are j>0, sum changes by 0.
        # If one is j=0, we need a compensating flip.
        
        # For simplicity: only swap pairs (j>0), both from positions 1-83.
        j1 = np.random.randint(1, 84)
        j2 = np.random.randint(1, 84)
        while j2 == j1 or seqs_bcd[s, j1] == seqs_bcd[s, j2]:
            j2 = np.random.randint(1, 84)
            if j2 == j1:
                continue
        
        # j1 has one sign, j2 has the opposite. Flip both pairs.
        p1a, p1b = j1, P - j1
        p2a, p2b = j2, P - j2
        
        old_val1 = seqs_bcd[s, j1]  # same as seqs_bcd[s, P-j1]
        old_val2 = seqs_bcd[s, j2]
        
        # Compute new cost
        new_cost = 0.0
        new_dft = np.empty(P, dtype=np.float64)
        for k in range(P):
            # DFT update: flipping positions p1a, p1b, p2a, p2b
            diff1 = -2.0 * old_val1  # flip value at j1 and P-j1
            diff2 = -2.0 * old_val2  # flip value at j2 and P-j2
            
            new_re = dft_re[s, k]
            new_re += diff1 * (cos_t[k, p1a] + cos_t[k, p1b])
            new_re += diff2 * (cos_t[k, p2a] + cos_t[k, p2b])
            new_dft[k] = new_re
            
            # New PSD
            old_pow = dft_re[s, k]**2
            new_pow = new_re**2
            new_psd_k = psd[k] + new_pow - old_pow
            
            if k > 0:
                d = new_psd_k - TARGET_BCD
            else:
                d = new_psd_k - 667.0
            new_cost += d * d
        
        delta = new_cost - current_cost
        
        if delta < 0 or (T > 1e-30 and np.random.random() < np.exp(-delta / T)):
            # Accept
            for k in range(P):
                old_pow = dft_re[s, k]**2
                dft_re[s, k] = new_dft[k]
                new_pow = dft_re[s, k]**2
                psd[k] += new_pow - old_pow
            
            seqs_bcd[s, p1a] = -old_val1
            seqs_bcd[s, p1b] = -old_val1
            seqs_bcd[s, p2a] = -old_val2
            seqs_bcd[s, p2b] = -old_val2
            
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_linf = linf_bcd(psd, TARGET_BCD)
                best_seqs[:] = seqs_bcd[:]
                if best_cost < 1.0:
                    return best_seqs, best_cost, best_linf
    
    return best_seqs, best_cost, best_linf


@njit(cache=True)
def sa_symmetric_free(seqs_bcd, cos_t, sin_t, max_iters, T_start, T_end, seed):
    """
    SA for 3 symmetric sequences with FREE orbit flips (no row sum constraint).
    Flip a single orbit (pair of positions) in one sequence.
    """
    np.random.seed(seed)
    
    dft_re = compute_bcd_dft(seqs_bcd, cos_t, sin_t)
    psd = psd_bcd(dft_re)
    
    current_cost = l2_bcd(psd, TARGET_BCD)
    best_cost = current_cost
    best_linf = linf_bcd(psd, TARGET_BCD)
    best_seqs = seqs_bcd.copy()
    
    log_ratio = np.log(T_end / T_start)
    inv_max = 1.0 / max_iters
    
    for it in range(max_iters):
        T = T_start * np.exp(log_ratio * it * inv_max)
        
        s = np.random.randint(3)
        j = np.random.randint(0, 84)  # orbit index: 0 is singleton, 1-83 are pairs
        
        old_val = seqs_bcd[s, j]
        diff = -2.0 * old_val
        
        if j == 0:
            # Singleton: just flip position 0
            new_cost = 0.0
            new_dft_vals = np.empty(P, dtype=np.float64)
            for k in range(P):
                new_re = dft_re[s, k] + diff * cos_t[k, 0]
                new_dft_vals[k] = new_re
                old_pow = dft_re[s, k]**2
                new_pow = new_re**2
                new_psd_k = psd[k] + new_pow - old_pow
                if k > 0:
                    d = new_psd_k - TARGET_BCD
                else:
                    d = new_psd_k - 667.0
                new_cost += d * d
            
            delta = new_cost - current_cost
            if delta < 0 or (T > 1e-30 and np.random.random() < np.exp(-delta / T)):
                for k in range(P):
                    old_pow = dft_re[s, k]**2
                    dft_re[s, k] = new_dft_vals[k]
                    psd[k] += dft_re[s, k]**2 - old_pow
                seqs_bcd[s, 0] = -old_val
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_linf = linf_bcd(psd, TARGET_BCD)
                    best_seqs[:] = seqs_bcd[:]
        else:
            # Pair: flip positions j and P-j
            p1, p2 = j, P - j
            new_cost = 0.0
            new_dft_vals = np.empty(P, dtype=np.float64)
            for k in range(P):
                new_re = dft_re[s, k] + diff * (cos_t[k, p1] + cos_t[k, p2])
                new_dft_vals[k] = new_re
                old_pow = dft_re[s, k]**2
                new_pow = new_re**2
                new_psd_k = psd[k] + new_pow - old_pow
                if k > 0:
                    d = new_psd_k - TARGET_BCD
                else:
                    d = new_psd_k - 667.0
                new_cost += d * d
            
            delta = new_cost - current_cost
            if delta < 0 or (T > 1e-30 and np.random.random() < np.exp(-delta / T)):
                for k in range(P):
                    old_pow = dft_re[s, k]**2
                    dft_re[s, k] = new_dft_vals[k]
                    psd[k] += dft_re[s, k]**2 - old_pow
                seqs_bcd[s, p1] = -old_val
                seqs_bcd[s, p2] = -old_val
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_linf = linf_bcd(psd, TARGET_BCD)
                    best_seqs[:] = seqs_bcd[:]
        
        if best_cost < 1.0:
            return best_seqs, best_cost, best_linf
    
    return best_seqs, best_cost, best_linf


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    leg = make_leg()
    cos_t, sin_t = build_cos_sin_tables()
    
    # Row sum triples for (b,c,d): (1,15,21) or (9,15,19)
    # We try all permutations and sign combinations
    base_triples = [(1, 15, 21), (9, 15, 19)]
    
    best_ever_cost = float('inf')
    best_ever_linf = float('inf')
    best_ever_seqs_bcd = None
    
    start = time.time()
    run = 0
    
    print(f"=== SKEW+SYMMETRIC SEARCH for H(668) ===")
    print(f"Fixed sequence a = Legendre (skew, sum=1)")
    print(f"Searching for 3 symmetric sequences b,c,d")
    print(f"Need PSD_bcd(k) = 500 at k=1,...,166 and 667 at k=0")
    print(f"Time budget: {args.time}s")
    print(f"{'='*80}")
    
    while time.time() - start < args.time:
        run += 1
        elapsed = time.time() - start
        remaining = args.time - elapsed
        if remaining < 5:
            break
        
        rseed = args.seed + run * 73
        rng = np.random.RandomState(rseed)
        
        # Choose triple
        triple = base_triples[run % 2]
        sums = list(triple)
        rng.shuffle(sums)
        # Random signs
        for i in range(3):
            if rng.random() < 0.5:
                sums[i] = -sums[i]
        
        # Initialize 3 symmetric sequences
        seqs_bcd = np.zeros((3, P), dtype=np.int8)
        try:
            for i in range(3):
                seqs_bcd[i] = init_symmetric_with_rowsum(sums[i], rng)
        except ValueError:
            continue
        
        # Choose SA variant
        n_iters = min(5_000_000, max(500_000, int(remaining * 100_000)))
        
        strategy = run % 4
        if strategy < 2:
            # Row-sum preserving
            T_start = 50.0 if strategy == 0 else 200.0
            result, cost, linf_val = sa_symmetric(
                seqs_bcd, cos_t, sin_t, n_iters, T_start, 0.001, rseed
            )
        else:
            # Free orbit flips
            T_start = 100.0 if strategy == 2 else 500.0
            result, cost, linf_val = sa_symmetric_free(
                seqs_bcd, cos_t, sin_t, n_iters, T_start, 0.001, rseed
            )
        
        improved = cost < best_ever_cost
        if improved:
            best_ever_cost = cost
            best_ever_linf = linf_val
            best_ever_seqs_bcd = result.copy()
        
        actual_sums = [int(np.sum(result[i])) for i in range(3)]
        marker = " ***" if improved else ""
        stnames = ['RS-lo', 'RS-hi', 'Free-lo', 'Free-hi']
        print(f"R{run:3d} [{stnames[strategy]:7s}] sums_bcd={actual_sums} "
              f"L2={cost:10.0f} Li={linf_val:6.1f} [{elapsed:.0f}s]{marker}")
        
        if cost < 1.0:
            print("\n*** EXACT SOLUTION FOUND! ***")
            break
        
        # Also try exploitation of best
        if best_ever_seqs_bcd is not None and run % 6 == 5:
            exploit_seqs = best_ever_seqs_bcd.copy()
            n_iters_ex = min(3_000_000, int(remaining * 80_000))
            result_ex, cost_ex, linf_ex = sa_symmetric(
                exploit_seqs, cos_t, sin_t, n_iters_ex, 5.0, 0.0001, rseed + 1
            )
            if cost_ex < best_ever_cost:
                best_ever_cost = cost_ex
                best_ever_linf = linf_ex
                best_ever_seqs_bcd = result_ex.copy()
                sums_ex = [int(np.sum(result_ex[i])) for i in range(3)]
                print(f"  EXPLOIT: sums={sums_ex} L2={cost_ex:.0f} Li={linf_ex:.1f} ***")
    
    elapsed = time.time() - start
    print(f"\n{'='*80}")
    print(f"Search complete: {elapsed:.1f}s, {run} runs")
    print(f"Best L2={best_ever_cost:.0f}, Linf={best_ever_linf:.1f}")
    
    if best_ever_seqs_bcd is not None:
        sums = [int(np.sum(best_ever_seqs_bcd[i])) for i in range(3)]
        print(f"Best row sums (b,c,d): {sums}")
        
        # Verify full PSD
        psd_total = np.zeros(P, dtype=np.float64)
        # Add Legendre contribution
        leg_fft = fft(leg.astype(np.float64))
        psd_total += np.abs(leg_fft)**2
        for i in range(3):
            sf = fft(best_ever_seqs_bcd[i].astype(np.float64))
            psd_total += np.abs(sf)**2
        
        dev = psd_total - N
        print(f"Full PSD deviation: L2={np.sum(dev**2):.0f}, Linf={np.max(np.abs(dev)):.1f}")
        
        # Check exact PAF
        all_seqs = np.zeros((4, P), dtype=np.int64)
        all_seqs[0] = leg.astype(np.int64)
        for i in range(3):
            all_seqs[i+1] = best_ever_seqs_bcd[i].astype(np.int64)
        
        max_paf = 0
        for tau in range(1, P):
            paf = 0
            for s in range(4):
                for j in range(P):
                    paf += int(all_seqs[s, j]) * int(all_seqs[s, (j + tau) % P])
            max_paf = max(max_paf, abs(paf))
        print(f"Exact max |PAF_total(tau)|: {max_paf}")
        
        if best_ever_cost < 1.0:
            # Build and verify GS matrix
            sys.path.insert(0, 'results')
            from hadamard_core import goethals_seidel_array, verify_hadamard, export_csv
            H = goethals_seidel_array(leg, best_ever_seqs_bcd[0], 
                                      best_ever_seqs_bcd[1], best_ever_seqs_bcd[2])
            valid, msg = verify_hadamard(H)
            print(f"GS Verification: {msg}")
            if valid:
                export_csv(H, 'hadamard_668.csv')
                np.savez('results/solution_sequences.npz',
                         a=leg, b=best_ever_seqs_bcd[0],
                         c=best_ever_seqs_bcd[1], d=best_ever_seqs_bcd[2])
                print("SAVED hadamard_668.csv!")
        
        np.savez('results/skew_sym_best.npz',
                 leg=leg, bcd=best_ever_seqs_bcd,
                 cost=best_ever_cost, linf=best_ever_linf)

if __name__ == '__main__':
    main()
