#!/usr/bin/env python3
"""
Intensive PAF-based SA search for H(668).
Uses exact integer arithmetic and Numba JIT for speed.
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
    s = np.array([leg_sym(i) for i in range(P)], dtype=np.int8)
    s[0] = 1  # Convention: chi(0) = 1 for the Hadamard construction
    return s

@njit
def compute_paf_total(seqs):
    paf = np.zeros(P, dtype=np.int64)
    for tau in range(1, P):
        total = np.int64(0)
        for s in range(4):
            for j in range(P):
                total += np.int64(seqs[s, j]) * np.int64(seqs[s, (j + tau) % P])
        paf[tau] = total
    return paf

@njit
def paf_cost_l2(paf):
    cost = np.int64(0)
    for tau in range(1, P):
        cost += paf[tau] * paf[tau]
    return cost

@njit
def sa_intensive(seqs_init, max_iters, T_start, T_end, seed):
    """Ultra-focused SA with single flips, optimizing PAF L2."""
    np.random.seed(seed)
    seqs = seqs_init.copy()
    
    # Compute initial PAF
    paf = compute_paf_total(seqs)
    current_cost = paf_cost_l2(paf)
    best_cost = current_cost
    best_seqs = seqs.copy()
    
    log_ratio = np.log(T_end / T_start)
    inv_max = 1.0 / max_iters
    
    for it in range(max_iters):
        T = T_start * np.exp(log_ratio * it * inv_max)
        
        s = np.random.randint(4)
        j = np.random.randint(P)
        old_val = seqs[s, j]
        
        # Compute delta cost
        delta_cost = np.int64(0)
        for tau in range(1, P):
            jp = (j + tau) % P
            jm = (j - tau + P) % P
            neighbor_sum = np.int64(seqs[s, jp]) + np.int64(seqs[s, jm])
            delta_tau = np.int64(-2) * np.int64(old_val) * neighbor_sum
            
            old_sq = paf[tau] * paf[tau]
            new_val_tau = paf[tau] + delta_tau
            new_sq = new_val_tau * new_val_tau
            delta_cost += new_sq - old_sq
        
        accept = False
        if delta_cost < 0:
            accept = True
        elif T > 1e-30:
            if np.random.random() < np.exp(-float(delta_cost) / T):
                accept = True
        
        if accept:
            seqs[s, j] = -old_val
            # Update PAF - use original old_val (before flip)
            for tau in range(1, P):
                jp = (j + tau) % P
                jm = (j - tau + P) % P
                # seqs[s, jp] and seqs[s, jm] haven't changed 
                # (since jp != j and jm != j for tau > 0 when P is odd)
                neighbor_sum = np.int64(seqs[s, jp]) + np.int64(seqs[s, jm])
                # Wait: after flip, seqs[s,j] = -old_val. But jp = (j+tau)%P != j and jm != j.
                # However, when computing delta for shift tau, the neighbors are at jp and jm.
                # If at some other shift tau', jp' = j (meaning j+tau' ≡ j mod P → tau'=0, excluded)
                # or jm' = j (meaning j-tau' ≡ j mod P → tau'=0, excluded).
                # So no: for ANY tau>0, the neighbor positions jp and jm are never j.
                # Therefore the delta is exact: delta_tau = -2 * old_val * (seqs[s,jp] + seqs[s,jm])
                delta_tau = np.int64(-2) * np.int64(old_val) * neighbor_sum
                paf[tau] += delta_tau
            
            current_cost += delta_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_seqs[:] = seqs[:]
                if best_cost == 0:
                    return best_seqs, best_cost
    
    return best_seqs, best_cost


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type=int, default=600)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    leg = make_leg()
    
    print("Warming up Numba...")
    # Warmup
    test_seqs = np.array([leg.copy() for _ in range(4)], dtype=np.int8)
    _ = sa_intensive(test_seqs, 100, 10.0, 0.1, 0)
    print("Warmup done.")
    
    best_overall = 2656  # Legendre baseline
    best_seqs_overall = np.array([leg.copy() for _ in range(4)], dtype=np.int8)
    
    start = time.time()
    trial = 0
    
    print(f"\n=== INTENSIVE PAF SEARCH ===")
    print(f"Time budget: {args.time}s")
    print(f"Legendre baseline: L2=2656, Linf=4")
    print(f"{'='*80}")
    
    while time.time() - start < args.time:
        trial += 1
        elapsed = time.time() - start
        remaining = args.time - elapsed
        if remaining < 3:
            break
        
        rseed = args.seed + trial * 7919
        rng = np.random.RandomState(rseed)
        
        # Perturbation: vary the number and distribution of flips
        n_pert = rng.randint(3, 50)
        
        seqs = np.array([leg.copy() for _ in range(4)], dtype=np.int8)
        for s_idx in range(4):
            np_s = rng.randint(0, n_pert + 1)
            if np_s > 0:
                idx = rng.choice(P, min(np_s, P), replace=False)
                seqs[s_idx, idx] = -seqs[s_idx, idx]
        
        # Random temperature schedule
        T_start = rng.uniform(2.0, 300.0)
        T_end = rng.uniform(0.00001, 0.1)
        n_iters = min(8_000_000, max(1_000_000, int(remaining * 200_000)))
        
        result, cost = sa_intensive(seqs, n_iters, T_start, T_end, rseed)
        
        if cost < best_overall:
            best_overall = cost
            best_seqs_overall = result.copy()
            sums = [int(np.sum(result[i])) for i in range(4)]
            paf = compute_paf_total(result)
            n_zero = np.sum(paf[1:] == 0)
            linf = max(abs(paf[tau]) for tau in range(1, P))
            print(f"T{trial:4d} [{elapsed:5.0f}s] NEW BEST L2={cost:5d}, Linf={linf:3d}, "
                  f"PAF=0: {n_zero:3d}/166, sums={sums}, "
                  f"pert={n_pert}, T=[{T_start:.1f},{T_end:.5f}]")
            
            if cost == 0:
                print("\n*** EXACT SOLUTION FOUND! ***")
                break
    
    elapsed = time.time() - start
    print(f"\n{'='*80}")
    print(f"Done in {elapsed:.1f}s, {trial} trials")
    print(f"Best L2={best_overall}")
    
    # Verify best
    paf = compute_paf_total(best_seqs_overall)
    paf_nz = paf[1:]
    vals, counts = np.unique(paf_nz, return_counts=True)
    print(f"PAF distribution:")
    for v, c in zip(vals, counts):
        print(f"  PAF={v:4d}: {c:4d} shifts")
    
    linf = max(abs(paf[tau]) for tau in range(1, P))
    print(f"Linf = {linf}")
    sums = [int(np.sum(best_seqs_overall[i])) for i in range(4)]
    print(f"Row sums: {sums}")
    
    # Build GS matrix and check quality
    print("\nBuilding GS matrix...")
    sys.path.insert(0, 'results')
    from hadamard_core import goethals_seidel_array, verify_hadamard, export_csv
    
    H = goethals_seidel_array(best_seqs_overall[0], best_seqs_overall[1],
                              best_seqs_overall[2], best_seqs_overall[3])
    valid, msg = verify_hadamard(H)
    print(f"Verification: {msg}")
    
    if valid:
        export_csv(H, 'hadamard_668.csv')
        np.savez('results/solution_sequences.npz',
                 a=best_seqs_overall[0], b=best_seqs_overall[1],
                 c=best_seqs_overall[2], d=best_seqs_overall[3])
        print("SAVED exact Hadamard matrix!")
    else:
        # Save best near-miss
        export_csv(H, 'hadamard_668.csv')
        np.savez('results/best_near_miss.npz',
                 seqs=best_seqs_overall, cost=best_overall, linf=linf)
        print(f"Saved best near-miss (L2={best_overall}, Linf={linf})")
    
    return best_overall

if __name__ == '__main__':
    main()
