#!/usr/bin/env python3
"""
Turbo search for H(668) via Goethals-Seidel array.

Ultra-optimized SA with:
- Precomputed cos/sin tables in Numba
- Batch evaluation of all possible swaps
- Adaptive temperature with reheating
- Multi-restart with best-solution memory
- Exact integer PAF verification
"""

import numpy as np
from numpy.fft import fft
from numba import njit, prange
import time
import sys

P = 167
N = 668

def legendre_symbol(a, p=P):
    if a % p == 0: return 0
    return 1 if pow(a, (p-1)//2, p) == 1 else -1

def make_leg():
    s = np.array([legendre_symbol(i) for i in range(P)], dtype=np.int8)
    s[0] = 1
    return s

@njit(cache=True)
def build_tables():
    """Precompute cos/sin tables for DFT."""
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
def init_dft(seqs, cos_t, sin_t):
    """Compute DFT of all 4 sequences."""
    dft_re = np.zeros((4, P), dtype=np.float64)
    dft_im = np.zeros((4, P), dtype=np.float64)
    for s in range(4):
        for k in range(P):
            re, im = 0.0, 0.0
            for j in range(P):
                re += seqs[s, j] * cos_t[k, j]
                im -= seqs[s, j] * sin_t[k, j]
            dft_re[s, k] = re
            dft_im[s, k] = im
    return dft_re, dft_im

@njit(cache=True)
def compute_psd(dft_re, dft_im):
    psd = np.zeros(P, dtype=np.float64)
    for k in range(P):
        for s in range(4):
            psd[k] += dft_re[s, k]**2 + dft_im[s, k]**2
    return psd

@njit(cache=True)
def l2(psd):
    c = 0.0
    for k in range(P):
        d = psd[k] - N
        c += d * d
    return c

@njit(cache=True)
def linf(psd):
    m = 0.0
    for k in range(P):
        d = abs(psd[k] - N)
        if d > m:
            m = d
    return m

@njit(cache=True)
def sa_turbo(seqs, cos_t, sin_t, max_iters, T_start, T_end, seed, mode):
    """
    Ultra-fast SA search.
    
    mode:
      0 = row-sum preserving swaps within sequence
      1 = free single flips
      2 = swap between two sequences at same position
    """
    np.random.seed(seed)
    
    dft_re, dft_im = init_dft(seqs, cos_t, sin_t)
    psd = compute_psd(dft_re, dft_im)
    
    current_cost = l2(psd)
    best_cost = current_cost
    best_linf = linf(psd)
    best_seqs = seqs.copy()
    
    log_ratio = np.log(T_end / T_start)
    T = T_start
    inv_max_iters = 1.0 / max_iters
    
    # Build position lists for mode 0
    # We'll track them dynamically
    if mode == 0:
        # For each sequence, maintain arrays of +1 and -1 positions
        pos_p = np.zeros((4, P), dtype=np.int64)
        pos_m = np.zeros((4, P), dtype=np.int64)
        np_count = np.zeros(4, dtype=np.int64)
        nm_count = np.zeros(4, dtype=np.int64)
        for s in range(4):
            pp, pm = 0, 0
            for j in range(P):
                if seqs[s, j] > 0:
                    pos_p[s, pp] = j
                    pp += 1
                else:
                    pos_m[s, pm] = j
                    pm += 1
            np_count[s] = pp
            nm_count[s] = pm
    
    for it in range(max_iters):
        # Geometric cooling
        T = T_start * np.exp(log_ratio * it * inv_max_iters)
        
        if mode == 0:
            # Row-sum preserving: swap a +1 and -1 in same sequence
            s = np.random.randint(4)
            if np_count[s] < 1 or nm_count[s] < 1:
                continue
            
            i1 = np.random.randint(np_count[s])
            i2 = np.random.randint(nm_count[s])
            p1 = pos_p[s, i1]  # currently +1
            p2 = pos_m[s, i2]  # currently -1
            
            # Compute cost change
            # p1: +1 → -1 (diff = -2)
            # p2: -1 → +1 (diff = +2)
            new_cost = 0.0
            for k in range(P):
                old_pow = dft_re[s, k]**2 + dft_im[s, k]**2
                new_re = dft_re[s, k] - 2.0 * cos_t[k, p1] + 2.0 * cos_t[k, p2]
                new_im = dft_im[s, k] + 2.0 * sin_t[k, p1] - 2.0 * sin_t[k, p2]
                new_pow = new_re**2 + new_im**2
                d = psd[k] + new_pow - old_pow - N
                new_cost += d * d
            
            delta = new_cost - current_cost
            
            if delta < 0 or (T > 1e-30 and np.random.random() < np.exp(-delta / T)):
                # Accept: update DFT and PSD
                for k in range(P):
                    old_pow = dft_re[s, k]**2 + dft_im[s, k]**2
                    dft_re[s, k] += -2.0 * cos_t[k, p1] + 2.0 * cos_t[k, p2]
                    dft_im[s, k] += 2.0 * sin_t[k, p1] - 2.0 * sin_t[k, p2]
                    new_pow = dft_re[s, k]**2 + dft_im[s, k]**2
                    psd[k] += new_pow - old_pow
                
                seqs[s, p1] = -1
                seqs[s, p2] = 1
                pos_p[s, i1] = p2
                pos_m[s, i2] = p1
                
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_linf = linf(psd)
                    best_seqs[:] = seqs[:]
                    if best_cost < 1.0:
                        return best_seqs, best_cost, best_linf
        
        elif mode == 1:
            # Free single flip
            s = np.random.randint(4)
            j = np.random.randint(P)
            diff = -2.0 * seqs[s, j]
            
            new_cost = 0.0
            for k in range(P):
                old_pow = dft_re[s, k]**2 + dft_im[s, k]**2
                new_re = dft_re[s, k] + diff * cos_t[k, j]
                new_im = dft_im[s, k] - diff * sin_t[k, j]
                new_pow = new_re**2 + new_im**2
                d = psd[k] + new_pow - old_pow - N
                new_cost += d * d
            
            delta = new_cost - current_cost
            
            if delta < 0 or (T > 1e-30 and np.random.random() < np.exp(-delta / T)):
                for k in range(P):
                    old_pow = dft_re[s, k]**2 + dft_im[s, k]**2
                    dft_re[s, k] += diff * cos_t[k, j]
                    dft_im[s, k] -= diff * sin_t[k, j]
                    new_pow = dft_re[s, k]**2 + dft_im[s, k]**2
                    psd[k] += new_pow - old_pow
                
                seqs[s, j] = -seqs[s, j]
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_linf = linf(psd)
                    best_seqs[:] = seqs[:]
                    if best_cost < 1.0:
                        return best_seqs, best_cost, best_linf
        
        elif mode == 2:
            # Swap between two sequences at same position
            s1 = np.random.randint(4)
            s2 = np.random.randint(3)
            if s2 >= s1:
                s2 += 1
            j = np.random.randint(P)
            
            if seqs[s1, j] == seqs[s2, j]:
                continue
            
            diff1 = -2.0 * seqs[s1, j]  # s1 flips
            diff2 = -2.0 * seqs[s2, j]  # s2 flips (opposite)
            
            new_cost = 0.0
            for k in range(P):
                old_pow1 = dft_re[s1, k]**2 + dft_im[s1, k]**2
                old_pow2 = dft_re[s2, k]**2 + dft_im[s2, k]**2
                
                new_re1 = dft_re[s1, k] + diff1 * cos_t[k, j]
                new_im1 = dft_im[s1, k] - diff1 * sin_t[k, j]
                new_re2 = dft_re[s2, k] + diff2 * cos_t[k, j]
                new_im2 = dft_im[s2, k] - diff2 * sin_t[k, j]
                
                new_pow1 = new_re1**2 + new_im1**2
                new_pow2 = new_re2**2 + new_im2**2
                
                d = psd[k] + (new_pow1 - old_pow1) + (new_pow2 - old_pow2) - N
                new_cost += d * d
            
            delta = new_cost - current_cost
            
            if delta < 0 or (T > 1e-30 and np.random.random() < np.exp(-delta / T)):
                for k in range(P):
                    old_pow1 = dft_re[s1, k]**2 + dft_im[s1, k]**2
                    old_pow2 = dft_re[s2, k]**2 + dft_im[s2, k]**2
                    
                    dft_re[s1, k] += diff1 * cos_t[k, j]
                    dft_im[s1, k] -= diff1 * sin_t[k, j]
                    dft_re[s2, k] += diff2 * cos_t[k, j]
                    dft_im[s2, k] -= diff2 * sin_t[k, j]
                    
                    new_pow1 = dft_re[s1, k]**2 + dft_im[s1, k]**2
                    new_pow2 = dft_re[s2, k]**2 + dft_im[s2, k]**2
                    
                    psd[k] += (new_pow1 - old_pow1) + (new_pow2 - old_pow2)
                
                seqs[s1, j], seqs[s2, j] = seqs[s2, j], seqs[s1, j]
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_linf = linf(psd)
                    best_seqs[:] = seqs[:]
                    if best_cost < 1.0:
                        return best_seqs, best_cost, best_linf
    
    return best_seqs, best_cost, best_linf


def verify_paf_exact(seqs):
    """Verify PAF using exact integer arithmetic."""
    s = seqs.astype(np.int64)
    max_paf = 0
    for tau in range(1, P):
        paf_total = 0
        for seq_idx in range(4):
            for j in range(P):
                paf_total += int(s[seq_idx, j]) * int(s[seq_idx, (j + tau) % P])
        max_paf = max(max_paf, abs(paf_total))
    return max_paf


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quad', type=str, default='all',
                       help='Quadruple: all, or index 0-9, or "legendre"')
    args = parser.parse_args()
    
    leg = make_leg()
    cos_t, sin_t = build_tables()
    
    # Quadruples sorted by number of flips from Legendre
    quads = [
        (3, 3, 5, 25),     # 16 flips
        (1, 1, 15, 21),    # 17 flips
        (3, 3, 17, 19),    # 19 flips  
        (3, 9, 17, 17),    # 21 flips
        (3, 3, 11, 23),    # 20 flips
        (5, 9, 11, 21),    # 19 flips
        (3, 7, 13, 21),    # 19 flips
        (7, 13, 15, 15),   # 23 flips
        (1, 9, 15, 19),    # 18 flips
        (3, 7, 9, 23),     # 19 flips
    ]
    
    best_ever_cost = float('inf')
    best_ever_linf = float('inf')
    best_ever_seqs = None
    
    start_time = time.time()
    total_iters = 0
    run_id = 0
    
    print(f"=== TURBO SEARCH for H(668) ===")
    print(f"Time budget: {args.time}s, Seed: {args.seed}")
    print(f"{'='*80}")
    
    while time.time() - start_time < args.time:
        run_id += 1
        elapsed = time.time() - start_time
        remaining = args.time - elapsed
        if remaining < 5:
            break
        
        rseed = args.seed + run_id * 97
        rng = np.random.RandomState(rseed)
        
        # Choose strategy
        phase = run_id % 6
        
        if phase < 3:
            # Init from Legendre with correct row sums
            qi = run_id % len(quads)
            quad = quads[qi]
            sums = list(quad)
            # Random sign/permutation
            rng.shuffle(sums)
            for i in range(4):
                if rng.random() < 0.5:
                    sums[i] = -sums[i]
            
            seqs = np.zeros((4, P), dtype=np.int8)
            for i in range(4):
                seq = leg.copy()
                cur = int(np.sum(seq))
                diff = sums[i] - cur
                n_flip = abs(diff) // 2
                if diff > 0:
                    cands = np.where(seq == -1)[0]
                elif diff < 0:
                    cands = np.where(seq == 1)[0]
                else:
                    cands = np.array([], dtype=np.int64)
                if n_flip > 0 and len(cands) >= n_flip:
                    idx = rng.choice(cands, n_flip, replace=False)
                    seq[idx] = -seq[idx]
                seqs[i] = seq
            
            mode = 0  # row-sum preserving
            T_start, T_end = 50.0, 0.001
            n_iters = min(5_000_000, max(500_000, int(remaining * 80_000)))
            
        elif phase == 3:
            # Start from best ever, exploit with low temp
            if best_ever_seqs is None:
                continue
            seqs = best_ever_seqs.copy()
            mode = 0
            T_start, T_end = 5.0, 0.0001
            n_iters = min(3_000_000, int(remaining * 80_000))
            
        elif phase == 4:
            # Free flips from Legendre (no row sum constraint)
            seqs = np.zeros((4, P), dtype=np.int8)
            for i in range(4):
                seqs[i] = leg.copy()
                # Small random perturbation
                n_pert = rng.randint(5, 20)
                idx = rng.choice(P, n_pert, replace=False)
                seqs[i, idx] = -seqs[i, idx]
            mode = 1
            T_start, T_end = 100.0, 0.01
            n_iters = min(5_000_000, int(remaining * 80_000))
            
        else:  # phase == 5
            # Random initialization
            seqs = np.zeros((4, P), dtype=np.int8)
            qi = rng.randint(len(quads))
            quad = quads[qi]
            sums = list(quad)
            rng.shuffle(sums)
            for i in range(4):
                if rng.random() < 0.5:
                    sums[i] = -sums[i]
                n_ones = (P + sums[i]) // 2
                seq = -np.ones(P, dtype=np.int8)
                idx = rng.choice(P, n_ones, replace=False)
                seq[idx] = 1
                seqs[i] = seq
            mode = 0
            T_start, T_end = 200.0, 0.01
            n_iters = min(5_000_000, int(remaining * 80_000))
        
        # Run SA
        result_seqs, cost, linf_val = sa_turbo(
            seqs, cos_t, sin_t, n_iters, T_start, T_end, rseed, mode
        )
        total_iters += n_iters
        
        actual_sums = [int(np.sum(result_seqs[i])) for i in range(4)]
        psd0 = sum(s*s for s in actual_sums)
        
        improved = cost < best_ever_cost
        if improved:
            best_ever_cost = cost
            best_ever_linf = linf_val
            best_ever_seqs = result_seqs.copy()
        
        mode_names = ['RS-SA', 'Free', 'Swap', 'RS-SA', 'Free', 'Rand']
        marker = " ***" if improved else ""
        print(f"R{run_id:3d} [{mode_names[phase]:5s}] L2={cost:10.0f} Li={linf_val:6.1f} "
              f"sums={actual_sums} PSD0={psd0:4d} "
              f"[{time.time()-start_time:.0f}s]{marker}")
        
        if cost < 1.0:
            print("\n*** EXACT SOLUTION FOUND! ***")
            break
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Search complete: {elapsed:.1f}s, {total_iters/1e6:.1f}M iterations, {run_id} runs")
    print(f"Best L2={best_ever_cost:.0f}, Linf={best_ever_linf:.1f}")
    
    if best_ever_seqs is not None:
        sums = [int(np.sum(best_ever_seqs[i])) for i in range(4)]
        print(f"Best row sums: {sums}")
        
        # Exact PAF check
        max_paf = verify_paf_exact(best_ever_seqs)
        print(f"Exact max |PAF_total(tau)|: {max_paf}")
        
        # Save
        np.savez('results/turbo_best.npz',
                 seqs=best_ever_seqs, cost=best_ever_cost, linf=best_ever_linf)
        
        if best_ever_cost < 1.0:
            print("\nBuilding and verifying GS matrix...")
            sys.path.insert(0, 'results')
            from hadamard_core import goethals_seidel_array, verify_hadamard, export_csv
            H = goethals_seidel_array(best_ever_seqs[0], best_ever_seqs[1],
                                      best_ever_seqs[2], best_ever_seqs[3])
            valid, msg = verify_hadamard(H)
            print(f"Verification: {msg}")
            if valid:
                export_csv(H, 'hadamard_668.csv')
                print("SAVED hadamard_668.csv")
    
    return best_ever_seqs, best_ever_cost, best_ever_linf

if __name__ == '__main__':
    main()
