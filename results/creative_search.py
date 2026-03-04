#!/usr/bin/env python3
"""
Creative multi-strategy search for H(668).

Novel approaches:
1. Genetic Algorithm with crossover on PAF-good sequences
2. Multi-sequence swap moves (swap values between two sequences)
3. Greedy construction with backtracking 
4. Hill-climbing from multiple algebraically-motivated starting points
5. Taboo search to avoid revisiting local minima
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
    s[0] = 1
    return s

def primitive_root(p):
    """Find a primitive root mod p."""
    for g in range(2, p):
        seen = set()
        val = 1
        for _ in range(p-1):
            seen.add(val)
            val = (val * g) % p
        if len(seen) == p - 1:
            return g
    return None

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
def sa_with_swaps(seqs_init, max_iters, T_start, T_end, seed):
    """SA with both single flips AND inter-sequence swaps."""
    np.random.seed(seed)
    seqs = seqs_init.copy()
    
    paf = compute_paf_total(seqs)
    current_cost = paf_cost_l2(paf)
    best_cost = current_cost
    best_seqs = seqs.copy()
    
    log_ratio = np.log(T_end / T_start)
    inv_max = 1.0 / max_iters
    
    for it in range(max_iters):
        T = T_start * np.exp(log_ratio * it * inv_max)
        
        move_type = np.random.randint(3)  # 0: flip, 1: swap between seqs, 2: double flip
        
        if move_type == 0:
            # Single flip
            s = np.random.randint(4)
            j = np.random.randint(P)
            old_val = seqs[s, j]
            
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
                for tau in range(1, P):
                    jp = (j + tau) % P
                    jm = (j - tau + P) % P
                    neighbor_sum = np.int64(seqs[s, jp]) + np.int64(seqs[s, jm])
                    delta_tau = np.int64(-2) * np.int64(old_val) * neighbor_sum
                    paf[tau] += delta_tau
                current_cost += delta_cost
                
        elif move_type == 1:
            # Swap positions between two sequences at same index
            s1 = np.random.randint(4)
            s2 = np.random.randint(4)
            if s1 == s2:
                s2 = (s1 + 1) % 4
            j = np.random.randint(P)
            
            if seqs[s1, j] != seqs[s2, j]:
                # Swapping different values = two flips
                old_v1 = seqs[s1, j]
                old_v2 = seqs[s2, j]
                
                # Compute delta from flipping s1[j] and s2[j]
                delta_cost = np.int64(0)
                delta_paf = np.zeros(P, dtype=np.int64)
                
                for tau in range(1, P):
                    jp = (j + tau) % P
                    jm = (j - tau + P) % P
                    
                    ns1 = np.int64(seqs[s1, jp]) + np.int64(seqs[s1, jm])
                    ns2 = np.int64(seqs[s2, jp]) + np.int64(seqs[s2, jm])
                    
                    dt1 = np.int64(-2) * np.int64(old_v1) * ns1
                    dt2 = np.int64(-2) * np.int64(old_v2) * ns2
                    delta_tau = dt1 + dt2
                    delta_paf[tau] = delta_tau
                    
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
                    seqs[s1, j] = -old_v1
                    seqs[s2, j] = -old_v2
                    for tau in range(1, P):
                        paf[tau] += delta_paf[tau]
                    current_cost += delta_cost
                    
        else:
            # Double flip in same sequence at two positions
            s = np.random.randint(4)
            j1 = np.random.randint(P)
            j2 = np.random.randint(P)
            if j1 == j2:
                j2 = (j1 + 1) % P
            
            old_v1 = seqs[s, j1]
            old_v2 = seqs[s, j2]
            
            # Effect of flipping both j1 and j2 in sequence s
            delta_cost = np.int64(0)
            delta_paf = np.zeros(P, dtype=np.int64)
            
            for tau in range(1, P):
                j1p = (j1 + tau) % P
                j1m = (j1 - tau + P) % P
                j2p = (j2 + tau) % P
                j2m = (j2 - tau + P) % P
                
                # Delta from flipping j1 (using original values)
                ns1 = np.int64(seqs[s, j1p]) + np.int64(seqs[s, j1m])
                # But if j2 == j1p or j2 == j1m, the second flip interacts
                if j2 == j1p:
                    ns1 = np.int64(-seqs[s, j1p]) + np.int64(seqs[s, j1m])
                elif j2 == j1m:
                    ns1 = np.int64(seqs[s, j1p]) + np.int64(-seqs[s, j1m])
                dt1 = np.int64(-2) * np.int64(old_v1) * ns1
                
                # Delta from flipping j2 (using values after j1 is flipped)
                ns2 = np.int64(seqs[s, j2p]) + np.int64(seqs[s, j2m])
                if j1 == j2p:
                    ns2 = np.int64(-seqs[s, j2p]) + np.int64(seqs[s, j2m])
                elif j1 == j2m:
                    ns2 = np.int64(seqs[s, j2p]) + np.int64(-seqs[s, j2m])
                dt2 = np.int64(-2) * np.int64(old_v2) * ns2
                
                delta_tau = dt1 + dt2
                delta_paf[tau] = delta_tau
                
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
                seqs[s, j1] = -old_v1
                seqs[s, j2] = -old_v2
                for tau in range(1, P):
                    paf[tau] += delta_paf[tau]
                current_cost += delta_cost
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_seqs[:] = seqs[:]
            if best_cost == 0:
                return best_seqs, best_cost
    
    return best_seqs, best_cost


@njit
def tabu_sa(seqs_init, max_iters, T_start, T_end, seed, tabu_len=500):
    """SA with tabu list to avoid revisiting similar configurations."""
    np.random.seed(seed)
    seqs = seqs_init.copy()
    
    paf = compute_paf_total(seqs)
    current_cost = paf_cost_l2(paf)
    best_cost = current_cost
    best_seqs = seqs.copy()
    
    # Simple tabu: track recent (seq_idx, position) flips
    tabu_list = np.zeros((tabu_len, 2), dtype=np.int32)
    tabu_ptr = 0
    
    log_ratio = np.log(T_end / T_start)
    inv_max = 1.0 / max_iters
    
    for it in range(max_iters):
        T = T_start * np.exp(log_ratio * it * inv_max)
        
        s = np.random.randint(4)
        j = np.random.randint(P)
        
        # Check tabu
        is_tabu = False
        for t in range(tabu_len):
            if tabu_list[t, 0] == s and tabu_list[t, 1] == j:
                is_tabu = True
                break
        
        if is_tabu and np.random.random() > 0.05:  # Aspiration: 5% chance to override
            continue
            
        old_val = seqs[s, j]
        
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
            for tau in range(1, P):
                jp = (j + tau) % P
                jm = (j - tau + P) % P
                neighbor_sum = np.int64(seqs[s, jp]) + np.int64(seqs[s, jm])
                delta_tau = np.int64(-2) * np.int64(old_val) * neighbor_sum
                paf[tau] += delta_tau
            current_cost += delta_cost
            
            tabu_list[tabu_ptr, 0] = s
            tabu_list[tabu_ptr, 1] = j
            tabu_ptr = (tabu_ptr + 1) % tabu_len
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_seqs[:] = seqs[:]
                if best_cost == 0:
                    return best_seqs, best_cost
    
    return best_seqs, best_cost


def create_cyclotomic_init(class_assignments):
    """
    Create sequences based on cyclotomic class assignments.
    class_assignments: array of 4 values, one per coset of index-2 subgroup.
    
    Z_167* = <5>, QR = <25> (index 2 subgroup, 83 elements).
    We partition Z_167 \ {0} into QR and QNR.
    For each sequence, assign +1 or -1 to each class.
    """
    g = 5  # primitive root mod 167
    qr = set()
    val = 1
    for _ in range(83):
        qr.add(val)
        val = (val * g * g) % P
    
    seqs = np.ones((4, P), dtype=np.int8)
    for s in range(4):
        for j in range(1, P):
            if j in qr:
                seqs[s, j] = class_assignments[s][0]
            else:
                seqs[s, j] = class_assignments[s][1]
    return seqs


def create_power_residue_init(k=2):
    """Create initial sequences using k-th power residues and modifications."""
    g = 5
    # Create sequences where each uses a different "rotation" of QR/QNR
    seqs = np.ones((4, P), dtype=np.int8)
    
    powers = [1]
    val = g
    for _ in range(165):
        powers.append(val)
        val = (val * g) % P
    
    for s in range(4):
        for j in range(1, P):
            # Find discrete log
            for dl in range(166):
                if powers[dl] == j:
                    break
            # Use different partition for each sequence
            if (dl + s * 42) % 166 < 83:
                seqs[s, j] = 1
            else:
                seqs[s, j] = -1
    return seqs


def create_jacobi_sum_init():
    """
    Use Jacobi sums to create algebraically motivated starting sequences.
    For chi the Legendre symbol, Jacobi sum J(chi, chi) = sum_{a+b=1} chi(a)*chi(b).
    """
    leg = make_leg()
    g = 5
    
    # Create 4 sequences using different character-based constructions
    seqs = np.ones((4, P), dtype=np.int8)
    
    # Seq 0: Legendre symbol
    seqs[0] = leg.copy()
    
    # Seq 1: Shifted Legendre (cyclic shift by primitive root)
    seqs[1, 0] = 1
    for j in range(1, P):
        seqs[1, j] = leg[(j * g) % P]
    
    # Seq 2: Product of Legendre with shift
    seqs[2, 0] = 1
    for j in range(1, P):
        seqs[2, j] = leg[j] * leg[(j + 1) % P]
    
    # Seq 3: Negated product  
    seqs[3, 0] = 1
    for j in range(1, P):
        val = leg[j] * leg[(j * g * g) % P]
        seqs[3, j] = val if val != 0 else 1
    
    return seqs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type=int, default=300)
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()
    
    leg = make_leg()
    
    print("Warming up Numba...")
    test_seqs = np.array([leg.copy() for _ in range(4)], dtype=np.int8)
    _ = sa_with_swaps(test_seqs, 100, 10.0, 0.1, 0)
    _ = tabu_sa(test_seqs, 100, 10.0, 0.1, 0, 10)
    print("Warmup done.")
    
    best_overall = 2656
    best_seqs_overall = np.array([leg.copy() for _ in range(4)], dtype=np.int8)
    best_linf_overall = 4
    
    start = time.time()
    trial = 0
    
    # Strategy list: different initializations and search methods
    strategies = [
        ("legendre_perturb_swap", "swap_sa"),
        ("legendre_perturb_tabu", "tabu_sa"),
        ("jacobi_sum_swap", "swap_sa"),
        ("power_residue_swap", "swap_sa"),
        ("random_rowsum_swap", "swap_sa"),
        ("legendre_heavy_perturb_swap", "swap_sa"),
        ("legendre_perturb_tabu_long", "tabu_sa"),
    ]
    
    print(f"\n=== CREATIVE MULTI-STRATEGY SEARCH ===")
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
        
        strat_idx = trial % len(strategies)
        strat_name, method = strategies[strat_idx]
        
        # Create initial sequences based on strategy
        if "legendre_perturb" in strat_name:
            seqs = np.array([leg.copy() for _ in range(4)], dtype=np.int8)
            if "heavy" in strat_name:
                n_pert = rng.randint(20, 83)
            else:
                n_pert = rng.randint(3, 40)
            for s_idx in range(4):
                np_s = rng.randint(0, n_pert + 1)
                if np_s > 0:
                    idx = rng.choice(P, min(np_s, P), replace=False)
                    seqs[s_idx, idx] = -seqs[s_idx, idx]
                    
        elif "jacobi" in strat_name:
            seqs = create_jacobi_sum_init()
            # Add small perturbation
            for s_idx in range(4):
                np_s = rng.randint(0, 10)
                if np_s > 0:
                    idx = rng.choice(P, np_s, replace=False)
                    seqs[s_idx, idx] = -seqs[s_idx, idx]
                    
        elif "power_residue" in strat_name:
            seqs = create_power_residue_init(k=2)
            for s_idx in range(4):
                np_s = rng.randint(0, 15)
                if np_s > 0:
                    idx = rng.choice(P, np_s, replace=False)
                    seqs[s_idx, idx] = -seqs[s_idx, idx]
                    
        elif "random_rowsum" in strat_name:
            # Valid row sum quadruples
            quads = [
                (7, 13, 15, 15), (3, 9, 17, 17), (3, 3, 17, 19),
                (5, 9, 11, 21), (1, 9, 15, 19), (3, 7, 13, 21),
                (1, 1, 15, 21), (3, 3, 11, 23), (3, 7, 9, 23), (3, 3, 5, 25)
            ]
            quad = quads[rng.randint(len(quads))]
            seqs = np.ones((4, P), dtype=np.int8)
            for s_idx in range(4):
                target_sum = quad[s_idx]
                if rng.random() < 0.5:
                    target_sum = -target_sum
                n_neg = (P - target_sum) // 2  # number of -1's
                neg_positions = rng.choice(P, n_neg, replace=False)
                seqs[s_idx, neg_positions] = -1
        else:
            seqs = np.array([leg.copy() for _ in range(4)], dtype=np.int8)
        
        T_start = rng.uniform(5.0, 500.0)
        T_end = rng.uniform(0.00001, 0.05)
        n_iters = min(8_000_000, max(500_000, int(remaining * 150_000)))
        
        if method == "swap_sa":
            result, cost = sa_with_swaps(seqs, n_iters, T_start, T_end, rseed)
        else:
            tabu_l = rng.randint(100, 2000)
            result, cost = tabu_sa(seqs, n_iters, T_start, T_end, rseed, tabu_l)
        
        if cost < best_overall:
            best_overall = cost
            best_seqs_overall = result.copy()
            paf = compute_paf_total(result)
            n_zero = np.sum(paf[1:] == 0)
            linf = int(max(abs(paf[tau]) for tau in range(1, P)))
            best_linf_overall = linf
            sums = [int(np.sum(result[i])) for i in range(4)]
            print(f"T{trial:4d} [{elapsed:5.0f}s] NEW BEST L2={cost:5d}, Linf={linf:3d}, "
                  f"PAF=0: {n_zero:3d}/166, sums={sums}, strat={strat_name}")
            
            if cost == 0:
                print("\n*** EXACT SOLUTION FOUND! ***")
                break
    
    elapsed = time.time() - start
    print(f"\n{'='*80}")
    print(f"Done in {elapsed:.1f}s, {trial} trials")
    print(f"Best L2={best_overall}, Linf={best_linf_overall}")
    
    # Verify best
    paf = compute_paf_total(best_seqs_overall)
    paf_nz = paf[1:]
    vals, counts = np.unique(paf_nz, return_counts=True)
    print(f"PAF distribution:")
    for v, c in zip(vals, counts):
        print(f"  PAF={v:4d}: {c:4d} shifts")
    
    # Save results
    np.savez('results/creative_search_best.npz',
             seqs=best_seqs_overall, cost=best_overall, linf=best_linf_overall)
    
    return best_overall

if __name__ == '__main__':
    main()
