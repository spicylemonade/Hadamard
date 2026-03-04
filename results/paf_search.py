#!/usr/bin/env python3
"""
Direct PAF-based search for H(668).

Operate directly on the periodic autocorrelation function (PAF).
For 4 sequences a,b,c,d of length 167:
  PAF_total(tau) = sum_i PAF_i(tau) must be 0 for all tau = 1,...,166.
  
With Legendre baseline: PAF_total(tau) = -4 for all tau > 0.

Key: When flipping position j in sequence s (from v to -v):
  delta_PAF_s(tau) = -2v * (s[(j+tau)%P] + s[(j-tau)%P])  for tau > 0
  
This is O(P) per flip evaluation (much faster than DFT-based PSD update).
Total cost = sum_{tau>0} PAF_total(tau)^2.

With Legendre: cost = 166 * 16 = 2656.
"""

import numpy as np
from numba import njit
import time
import sys

P = 167
N = 668

def leg_sym(a, p=P):
    if a % p == 0: return 0
    return 1 if pow(a, (p-1)//2, p) == 1 else -1

def make_leg():
    return np.array([leg_sym(i) for i in range(P)], dtype=np.int8)


@njit(cache=True)
def compute_paf_total(seqs):
    """Compute PAF_total for all shifts tau = 1,...,P-1."""
    paf = np.zeros(P, dtype=np.int64)
    for tau in range(1, P):
        total = np.int64(0)
        for s in range(4):
            for j in range(P):
                total += np.int64(seqs[s, j]) * np.int64(seqs[s, (j + tau) % P])
        paf[tau] = total
    return paf


@njit(cache=True)
def paf_cost_l2(paf):
    """Sum of squared PAF values at nonzero shifts."""
    cost = np.int64(0)
    for tau in range(1, P):
        cost += paf[tau] * paf[tau]
    return cost


@njit(cache=True)
def paf_cost_linf(paf):
    """Max absolute PAF value."""
    mx = np.int64(0)
    for tau in range(1, P):
        v = abs(paf[tau])
        if v > mx:
            mx = v
    return mx


@njit(cache=True)
def delta_paf_single_flip(seqs, s, j, paf):
    """
    Compute the change in PAF_total from flipping seqs[s][j].
    Returns the new paf array (copy).
    Also returns delta_cost = new_cost - old_cost.
    """
    old_val = seqs[s, j]
    new_paf = paf.copy()
    delta_cost = np.int64(0)
    
    for tau in range(1, P):
        # Change in PAF_s at shift tau when flipping position j:
        # delta = -2 * old_val * (seqs[s][(j+tau)%P] + seqs[s][(j-tau)%P])
        jp = (j + tau) % P
        jm = (j - tau + P) % P
        
        if jp == j:  # tau = 0, shouldn't happen since tau >= 1
            continue
        
        neighbor_sum = np.int64(seqs[s, jp]) + np.int64(seqs[s, jm])
        
        # But if jp == jm (happens when 2*tau ≡ 0 mod P, i.e., tau = 0 or P/2, 
        # but P is odd so only tau=0 which we skip):
        # Actually for odd P, jp != jm for tau in 1,...,P-1 since 2*tau not ≡ 0 mod P.
        # Wait: P=167 is odd, so 2*tau mod P = 0 only when tau = 0 or tau = (P+1)/2 = 84.
        # No: 2*84 = 168 ≡ 1 mod 167. So 2*tau ≠ 0 for any tau in 1,...,166.
        # So jp and jm are always distinct.
        
        delta_tau = np.int64(-2) * np.int64(old_val) * neighbor_sum
        
        old_sq = new_paf[tau] * new_paf[tau]
        new_paf[tau] += delta_tau
        new_sq = new_paf[tau] * new_paf[tau]
        delta_cost += new_sq - old_sq
    
    return new_paf, delta_cost


@njit(cache=True)
def sa_paf_direct(seqs_init, max_iters, T_start, T_end, seed, 
                  allow_rowsum_change=True):
    """
    SA search directly on PAF representation.
    
    This is faster than DFT-based approaches because:
    - PAF update per flip: O(P) integer operations
    - No floating-point DFT needed
    - Cost is exact integer arithmetic
    """
    np.random.seed(seed)
    
    seqs = seqs_init.copy()
    paf = compute_paf_total(seqs)
    
    current_cost = paf_cost_l2(paf)
    best_cost = current_cost
    best_linf = paf_cost_linf(paf)
    best_seqs = seqs.copy()
    
    log_ratio = np.log(T_end / T_start) if T_start > 0 and T_end > 0 else 0.0
    inv_max = 1.0 / max_iters
    T = T_start
    
    accepts = np.int64(0)
    
    for it in range(max_iters):
        T = T_start * np.exp(log_ratio * it * inv_max)
        
        if allow_rowsum_change:
            # Single flip
            s = np.random.randint(4)
            j = np.random.randint(P)
            
            new_paf, delta_cost = delta_paf_single_flip(seqs, s, j, paf)
            
            if delta_cost < 0 or (T > 1e-30 and np.random.random() < np.exp(-float(delta_cost) / T)):
                seqs[s, j] = -seqs[s, j]
                paf[:] = new_paf[:]
                current_cost += delta_cost
                accepts += 1
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_linf = paf_cost_linf(paf)
                    best_seqs[:] = seqs[:]
                    
                    if best_cost == 0:
                        return best_seqs, best_cost, best_linf, accepts
        else:
            # Row-sum preserving: swap a +1 and -1 in same sequence
            s = np.random.randint(4)
            j1 = np.random.randint(P)
            while seqs[s, j1] != 1:
                j1 = np.random.randint(P)
            j2 = np.random.randint(P)
            while seqs[s, j2] != -1:
                j2 = np.random.randint(P)
            
            # Two-flip update: flip j1 then j2
            new_paf1, dc1 = delta_paf_single_flip(seqs, s, j1, paf)
            seqs[s, j1] = -seqs[s, j1]
            new_paf2, dc2 = delta_paf_single_flip(seqs, s, j2, new_paf1)
            seqs[s, j1] = -seqs[s, j1]  # revert temporarily
            
            total_delta = dc1 + dc2  # approximate (ignores cross-term)
            # More accurate: compute actual new cost
            actual_new_cost = paf_cost_l2(new_paf2)
            total_delta = actual_new_cost - current_cost
            
            if total_delta < 0 or (T > 1e-30 and np.random.random() < np.exp(-float(total_delta) / T)):
                seqs[s, j1] = -seqs[s, j1]
                seqs[s, j2] = -seqs[s, j2]
                paf[:] = new_paf2[:]
                current_cost = actual_new_cost
                accepts += 1
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_linf = paf_cost_linf(paf)
                    best_seqs[:] = seqs[:]
                    
                    if best_cost == 0:
                        return best_seqs, best_cost, best_linf, accepts
    
    return best_seqs, best_cost, best_linf, accepts


@njit(cache=True) 
def sa_paf_multi_flip(seqs_init, max_iters, T_start, T_end, seed, k_flips=2):
    """
    SA with k simultaneous flips (in different sequences and positions).
    More aggressive exploration of the landscape.
    """
    np.random.seed(seed)
    
    seqs = seqs_init.copy()
    paf = compute_paf_total(seqs)
    
    current_cost = paf_cost_l2(paf)
    best_cost = current_cost
    best_linf = paf_cost_linf(paf)
    best_seqs = seqs.copy()
    
    log_ratio = np.log(T_end / T_start) if T_start > 0 and T_end > 0 else 0.0
    inv_max = 1.0 / max_iters
    
    for it in range(max_iters):
        T = T_start * np.exp(log_ratio * it * inv_max)
        
        # Generate k random flips
        flip_seqs = np.empty(k_flips, dtype=np.int64)
        flip_pos = np.empty(k_flips, dtype=np.int64)
        for k in range(k_flips):
            flip_seqs[k] = np.random.randint(4)
            flip_pos[k] = np.random.randint(P)
        
        # Apply flips sequentially and track PAF changes
        new_paf = paf.copy()
        for k in range(k_flips):
            s = flip_seqs[k]
            j = flip_pos[k]
            old_val = seqs[s, j]
            
            for tau in range(1, P):
                jp = (j + tau) % P
                jm = (j - tau + P) % P
                neighbor_sum = np.int64(seqs[s, jp]) + np.int64(seqs[s, jm])
                delta_tau = np.int64(-2) * np.int64(old_val) * neighbor_sum
                new_paf[tau] += delta_tau
            
            seqs[s, j] = -old_val
        
        new_cost = paf_cost_l2(new_paf)
        delta = new_cost - current_cost
        
        if delta < 0 or (T > 1e-30 and np.random.random() < np.exp(-float(delta) / T)):
            paf[:] = new_paf[:]
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_linf = paf_cost_linf(paf)
                best_seqs[:] = seqs[:]
                
                if best_cost == 0:
                    return best_seqs, best_cost, best_linf
        else:
            # Revert all flips
            for k in range(k_flips - 1, -1, -1):
                seqs[flip_seqs[k], flip_pos[k]] = -seqs[flip_seqs[k], flip_pos[k]]
    
    return best_seqs, best_cost, best_linf


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type=int, default=600)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    leg = make_leg()
    
    best_ever_cost = 999999999
    best_ever_linf = 999
    best_ever_seqs = None
    
    start = time.time()
    run = 0
    
    print(f"=== PAF-DIRECT SEARCH for H(668) ===")
    print(f"Time budget: {args.time}s")
    print(f"{'='*80}")
    
    # First: baseline
    seqs_base = np.array([leg.copy() for _ in range(4)], dtype=np.int8)
    paf_base = compute_paf_total(seqs_base)
    base_cost = paf_cost_l2(paf_base)
    base_linf = paf_cost_linf(paf_base)
    print(f"Legendre baseline: L2={base_cost}, Linf={base_linf}")
    print(f"PAF values: {np.unique(paf_base[1:])}")
    best_ever_cost = base_cost
    best_ever_linf = base_linf
    best_ever_seqs = seqs_base.copy()
    
    while time.time() - start < args.time:
        run += 1
        elapsed = time.time() - start
        remaining = args.time - elapsed
        if remaining < 5:
            break
        
        rseed = args.seed + run * 131
        
        strategy = run % 8
        
        n_iters = min(10_000_000, max(1_000_000, int(remaining * 200_000)))
        
        if strategy == 0:
            # Start from Legendre, free flips, high T
            seqs = np.array([leg.copy() for _ in range(4)], dtype=np.int8)
            result, cost, linf_val, acc = sa_paf_direct(
                seqs, n_iters, 50.0, 0.001, rseed, True
            )
            name = "Leg-Free-Hi"
        elif strategy == 1:
            # Start from Legendre, free flips, low T
            seqs = np.array([leg.copy() for _ in range(4)], dtype=np.int8)
            result, cost, linf_val, acc = sa_paf_direct(
                seqs, n_iters, 5.0, 0.0001, rseed, True
            )
            name = "Leg-Free-Lo"
        elif strategy == 2:
            # Start from best, exploit
            if best_ever_seqs is not None:
                seqs = best_ever_seqs.copy()
            else:
                continue
            result, cost, linf_val, acc = sa_paf_direct(
                seqs, n_iters, 2.0, 0.0001, rseed, True
            )
            name = "Exploit"
        elif strategy == 3:
            # Multi-flip from Legendre
            seqs = np.array([leg.copy() for _ in range(4)], dtype=np.int8)
            result, cost, linf_val = sa_paf_multi_flip(
                seqs, n_iters // 2, 100.0, 0.01, rseed, k_flips=2
            )
            acc = 0
            name = "Multi-2"
        elif strategy == 4:
            # Multi-flip k=3 from Legendre
            seqs = np.array([leg.copy() for _ in range(4)], dtype=np.int8)
            result, cost, linf_val = sa_paf_multi_flip(
                seqs, n_iters // 3, 100.0, 0.01, rseed, k_flips=3
            )
            acc = 0
            name = "Multi-3"
        elif strategy == 5:
            # Start from random init with correct-ish structure
            rng = np.random.RandomState(rseed)
            seqs = np.array([leg.copy() for _ in range(4)], dtype=np.int8)
            # Randomly flip ~10% of positions in each sequence
            for s in range(4):
                n_flip = rng.randint(5, 30)
                pos = rng.choice(P, n_flip, replace=False)
                seqs[s, pos] = -seqs[s, pos]
            result, cost, linf_val, acc = sa_paf_direct(
                seqs, n_iters, 100.0, 0.001, rseed, True
            )
            name = "Perturbed"
        elif strategy == 6:
            # Multi-flip k=4 from best
            if best_ever_seqs is not None:
                seqs = best_ever_seqs.copy()
            else:
                continue
            result, cost, linf_val = sa_paf_multi_flip(
                seqs, n_iters // 4, 10.0, 0.001, rseed, k_flips=4
            )
            acc = 0
            name = "ExplMult4"
        else:
            # Different Legendre variant: use negated Legendre for some sequences
            seqs = np.array([leg.copy() for _ in range(4)], dtype=np.int8)
            # Negate sequences 2 and 3
            seqs[2] = -seqs[2]
            seqs[3] = -seqs[3]
            result, cost, linf_val, acc = sa_paf_direct(
                seqs, n_iters, 50.0, 0.001, rseed, True
            )
            name = "NegLeg"
        
        improved = cost < best_ever_cost
        if improved:
            best_ever_cost = cost
            best_ever_linf = linf_val
            best_ever_seqs = result.copy()
        
        sums = [int(np.sum(result[i])) for i in range(4)]
        marker = " ***" if improved else ""
        print(f"R{run:3d} [{name:10s}] L2={cost:8d} Li={linf_val:4d} sums={sums} "
              f"acc={acc} [{elapsed:.0f}s]{marker}")
        
        if cost == 0:
            print("\n*** EXACT SOLUTION FOUND! ***")
            break
    
    elapsed = time.time() - start
    print(f"\n{'='*80}")
    print(f"Complete: {elapsed:.1f}s, {run} runs")
    print(f"Best L2={best_ever_cost}, Linf={best_ever_linf}")
    
    if best_ever_seqs is not None:
        sums = [int(np.sum(best_ever_seqs[i])) for i in range(4)]
        print(f"Row sums: {sums}")
        
        # Verify exact PAF
        paf = compute_paf_total(best_ever_seqs)
        paf_nonzero = paf[1:]
        print(f"PAF values at nonzero shifts: {np.unique(paf_nonzero)}")
        print(f"PAF=0 count: {np.sum(paf_nonzero == 0)}/{len(paf_nonzero)}")
        
        if best_ever_cost == 0:
            # Build and verify
            sys.path.insert(0, 'results')
            from hadamard_core import goethals_seidel_array, verify_hadamard, export_csv
            H = goethals_seidel_array(best_ever_seqs[0], best_ever_seqs[1],
                                      best_ever_seqs[2], best_ever_seqs[3])
            valid, msg = verify_hadamard(H)
            print(f"Verification: {msg}")
            if valid:
                export_csv(H, 'hadamard_668.csv')
                np.savez('results/solution_sequences.npz',
                         a=best_ever_seqs[0], b=best_ever_seqs[1],
                         c=best_ever_seqs[2], d=best_ever_seqs[3])
                print("SAVED hadamard_668.csv!")
        
        np.savez('results/paf_search_best.npz',
                 seqs=best_ever_seqs, cost=best_ever_cost, linf=best_ever_linf)

if __name__ == '__main__':
    main()
