#!/usr/bin/env python3
"""
Algebraic construction attempts for H(668).

Approach: Instead of starting from 4 copies of Legendre and perturbing,
try fundamentally different 4-tuple constructions.

Key insight: We need PAF_a(t) + PAF_b(t) + PAF_c(t) + PAF_d(t) = 0 for all t != 0.
The Legendre gives PAF_total(t) = -4 for all t.
We need to find sequences where the individual PAFs cancel.

Strategy 1: Use character sums / Gauss periods
Strategy 2: Construct from difference sets in Z_167
Strategy 3: Use the Paley conference matrix approach but with 4 sequences
Strategy 4: Exhaustive search over small perturbations of Legendre
"""

import numpy as np
from numba import njit
import time
import sys
import itertools

P = 167
N = 668

def leg_sym(a, p=P):
    if a % p == 0: return 0
    return 1 if pow(a, (p-1)//2, p) == 1 else -1

def make_leg():
    s = np.array([leg_sym(i) for i in range(P)], dtype=np.int8)
    s[0] = 1
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
def compute_single_paf(seq):
    """Compute PAF of a single sequence."""
    paf = np.zeros(P, dtype=np.int64)
    for tau in range(P):
        total = np.int64(0)
        for j in range(P):
            total += np.int64(seq[j]) * np.int64(seq[(j + tau) % P])
        paf[tau] = total
    return paf

@njit
def sa_intensive(seqs_init, max_iters, T_start, T_end, seed):
    """Ultra-focused SA with single flips, optimizing PAF L2."""
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
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_seqs[:] = seqs[:]
                if best_cost == 0:
                    return best_seqs, best_cost
    
    return best_seqs, best_cost


def create_gauss_period_sequences():
    """
    Use Gauss periods to construct 4 sequences.
    
    For Z_167*, primitive root g=5.
    eta_0 = sum_{j in QR} omega^j (quadratic Gauss period)
    eta_1 = sum_{j in QNR} omega^j
    
    Since 166 = 2*83 and 83 is prime, we can also partition into 
    cosets of the unique subgroup of index 83 = {1, 166} = {1, -1 mod 167}.
    
    This gives 83 cosets of size 2: {g^k, g^{k+83}} = {g^k, -g^k mod 167}.
    """
    g = 5
    # Compute powers of g
    powers = [0] * P
    val = 1
    for k in range(166):
        powers[k] = val
        val = (val * g) % P
    
    # QR = {g^{2k}} for k=0,...,82
    qr = set(powers[k] for k in range(0, 166, 2))
    qnr = set(powers[k] for k in range(1, 166, 2))
    
    leg = make_leg()
    
    # Strategy: use 4 different sequences, each with different support patterns
    # but whose combined PAF sums to 0.
    
    # Approach: Let a = Legendre, and construct b,c,d such that
    # PAF_b(t) + PAF_c(t) + PAF_d(t) = -PAF_a(t) = 4 for all t > 0.
    # Each of b,c,d individually has PAF_i(0) = P = 167.
    # We need PAF_b(t) + PAF_c(t) + PAF_d(t) = 4 for all t > 0.
    # This means the 3 sequences need a total PAF of +4 at each shift.
    
    results = []
    
    # Try: b = Legendre shifted, c = modified, d = complement
    for shift_b in range(1, min(10, P)):
        b = np.array([leg[(j + shift_b) % P] for j in range(P)], dtype=np.int8)
        
        # For this b, compute PAF_b(t)
        paf_a = compute_single_paf(leg)
        paf_b = compute_single_paf(b)
        
        # We need PAF_c(t) + PAF_d(t) = 4 - PAF_b(t) for all t > 0
        # PAF of shifted Legendre: since cyclic shift doesn't change PAF,
        # paf_b = paf_a. So we need PAF_c + PAF_d = 4 - (-1) = 5 at each shift.
        # Wait: PAF_a(t) = -1 for Legendre with chi(0)=1 at each t>0.
        # So 4 copies: 4*(-1) = -4, matching the known gap.
        # If we keep a=Legendre and try different b:
        # PAF_total = PAF_a + PAF_b + PAF_c + PAF_d = 0
        # PAF_a(t) = -1 for t > 0
        # If b is also Legendre (shifted), PAF_b(t) = -1 too
        # So PAF_c + PAF_d must equal +2 at every shift.
        
        target_cd = 4 + int(paf_a[1])  # Should be 4 + (-1) = 3 if only a is Legendre
        # Actually: PAF_total = 0 needed. PAF_a = -1 at each shift.
        # If b also contributes PAF_b = -1, then PAF_c + PAF_d = 2 needed.
        
        results.append((shift_b, int(paf_a[1]), int(paf_b[1])))
    
    return results


def try_different_zero_values():
    """
    Key insight: The Legendre symbol has chi(0) = 0, but we set it to 1.
    What if we try different values for position 0 in each sequence?
    
    Actually, position 0 is special. For the GS construction,
    the first row of each circulant A,B,C,D starts with a[0], b[0], c[0], d[0].
    
    The PAF includes the contribution from position 0.
    Let's try: what if one sequence has a[0] = -1 instead of +1?
    """
    leg = make_leg()
    
    best_cost = float('inf')
    best_seqs = None
    best_config = None
    
    # Try all 16 combinations of a[0], b[0], c[0], d[0] in {-1, +1}
    for signs in itertools.product([-1, 1], repeat=4):
        seqs = np.array([leg.copy() for _ in range(4)], dtype=np.int8)
        for s in range(4):
            seqs[s, 0] = signs[s]
        
        paf = compute_paf_total(seqs)
        cost = int(paf_cost_l2(paf))
        linf = int(max(abs(paf[tau]) for tau in range(1, P)))
        
        if cost < best_cost:
            best_cost = cost
            best_seqs = seqs.copy()
            best_config = signs
            print(f"  signs={signs}: L2={cost}, Linf={linf}")
    
    return best_seqs, best_cost


def try_negating_sequences():
    """
    Try: a = chi, b = chi, c = chi, d = -chi, and permutations.
    Also: a = chi, b = -chi, c = chi, d = -chi.
    
    For -chi: PAF_{-chi}(t) = PAF_chi(t) (since (-x_i)(-x_{i+t}) = x_i * x_{i+t}).
    So negating doesn't change PAF. This means all sign combinations give same PAF_total.
    
    But wait: what about DIFFERENT sequences? What if we use chi and a related but 
    different sequence?
    """
    leg = make_leg()
    g = 5
    
    results = []
    
    # 1. Use Legendre with different cyclic shifts
    # Since shift preserves PAF, we need fundamentally different sequences.
    
    # 2. Use products: chi * chi_shift. These have different PAF!
    for shift in range(1, 20):
        seq = np.array([leg[j] * leg[(j + shift) % P] for j in range(P)], dtype=np.int8)
        paf = compute_single_paf(seq)
        results.append(('product_shift_%d' % shift, int(paf[1]), int(paf[2])))
    
    # 3. Use chi(j) * chi(j + k) for various k
    # This is a "multiplicative derivative" of the Legendre symbol
    
    return results


def exhaustive_small_perturbations():
    """
    Try ALL single-position perturbations of the Legendre baseline.
    There are 4 * 167 = 668 single flips. Check if any reduces PAF.
    
    Also try all pairs of flips within one sequence (167 choose 2 * 4).
    """
    leg = make_leg()
    seqs = np.array([leg.copy() for _ in range(4)], dtype=np.int8)
    
    paf = compute_paf_total(seqs)
    base_cost = int(paf_cost_l2(paf))
    print(f"Baseline cost: {base_cost}")
    
    # Try all single flips
    best_single = base_cost
    best_flip = None
    
    for s in range(4):
        for j in range(P):
            test = seqs.copy()
            test[s, j] = -test[s, j]
            test_paf = compute_paf_total(test)
            cost = int(paf_cost_l2(test_paf))
            if cost < best_single:
                best_single = cost
                best_flip = (s, j)
                linf = int(max(abs(test_paf[tau]) for tau in range(1, P)))
                print(f"  Single flip ({s},{j}): L2={cost}, Linf={linf}")
    
    print(f"\nBest single flip: {best_flip}, cost={best_single}")
    
    # Now try all pairs: pick one flip from one sequence, one from another
    best_pair = base_cost
    best_pair_flip = None
    
    print("\nTrying pairs of flips across different sequences...")
    count = 0
    for s1 in range(4):
        for s2 in range(s1 + 1, 4):
            for j1 in range(P):
                for j2 in range(P):
                    count += 1
                    if count % 100000 == 0:
                        print(f"  Progress: {count} pairs checked...")
                    
                    test = seqs.copy()
                    test[s1, j1] = -test[s1, j1]
                    test[s2, j2] = -test[s2, j2]
                    test_paf = compute_paf_total(test)
                    cost = int(paf_cost_l2(test_paf))
                    if cost < best_pair:
                        best_pair = cost
                        best_pair_flip = (s1, j1, s2, j2)
                        linf = int(max(abs(test_paf[tau]) for tau in range(1, P)))
                        print(f"  Pair ({s1},{j1}),({s2},{j2}): L2={cost}, Linf={linf}")
    
    print(f"\nBest pair: {best_pair_flip}, cost={best_pair}")
    
    return best_single, best_pair


def mixed_algebraic_search():
    """
    Try using 4 DIFFERENT sequences (not all Legendre).
    
    Key sequences to try:
    1. Legendre symbol chi(j)
    2. chi(j) * chi(j+k) for various k
    3. chi(j+k) (shifts)
    4. Sequences from ideals in Z[zeta_167]
    
    Since cyclic shift doesn't change PAF, we need truly different sequences.
    Product sequences chi(j)*chi(j+k) have different PAF!
    """
    leg = make_leg()
    
    # Compute product sequences and their PAFs
    print("Computing product sequences chi(j)*chi(j+k)...")
    product_seqs = {}
    product_pafs = {}
    
    for k in range(1, P):
        seq = np.array([int(leg[j] * leg[(j + k) % P]) for j in range(P)], dtype=np.int8)
        # Handle j=0: leg[0]=1, leg[k]=leg[k], so seq[0] = leg[k]
        paf = compute_single_paf(seq)
        product_seqs[k] = seq
        product_pafs[k] = paf
    
    # Also have the Legendre sequence itself
    leg_paf = compute_single_paf(leg)
    
    # Target: find k1, k2, k3 such that
    # PAF_leg(t) + PAF_{k1}(t) + PAF_{k2}(t) + PAF_{k3}(t) = 0 for all t>0
    # i.e., PAF_{k1}(t) + PAF_{k2}(t) + PAF_{k3}(t) = -PAF_leg(t) = 1 for all t>0
    
    print("Searching for good 4-tuples (Legendre + 3 product sequences)...")
    
    best_cost = float('inf')
    best_tuple = None
    
    # Try all triples of product sequences (too many: P^3 ~ 4.6M, but we can sample)
    rng = np.random.RandomState(42)
    n_samples = 500000
    
    for _ in range(n_samples):
        k1, k2, k3 = rng.choice(range(1, P), size=3, replace=False)
        
        cost = 0
        for t in range(1, P):
            total = int(leg_paf[t]) + int(product_pafs[k1][t]) + int(product_pafs[k2][t]) + int(product_pafs[k3][t])
            cost += total * total
        
        if cost < best_cost:
            best_cost = cost
            best_tuple = (k1, k2, k3)
            linf = max(abs(int(leg_paf[t]) + int(product_pafs[k1][t]) + int(product_pafs[k2][t]) + int(product_pafs[k3][t])) for t in range(1, P))
            print(f"  k=({k1},{k2},{k3}): L2={cost}, Linf={linf}")
            
            if cost == 0:
                print("*** EXACT SOLUTION FOUND! ***")
                break
    
    print(f"\nBest: k={best_tuple}, L2={best_cost}")
    
    # Now try: SA starting from the best algebraic 4-tuple
    if best_tuple is not None:
        k1, k2, k3 = best_tuple
        seqs = np.array([leg.copy(), product_seqs[k1].copy(), 
                         product_seqs[k2].copy(), product_seqs[k3].copy()], dtype=np.int8)
        
        print(f"\nRunning SA from best algebraic starting point...")
        result, cost = sa_intensive(seqs, 5_000_000, 100.0, 0.001, 42)
        paf = compute_paf_total(result)
        n_zero = np.sum(paf[1:] == 0)
        linf = int(max(abs(paf[tau]) for tau in range(1, P)))
        print(f"After SA: L2={cost}, Linf={linf}, PAF=0: {n_zero}/166")
        
        return result, cost
    
    return None, best_cost


def try_four_product_sequences():
    """
    Try all combinations of 4 product sequences chi(j)*chi(j+k_i).
    Specifically, sample from the space of 4-tuples (k1,k2,k3,k4).
    """
    leg = make_leg()
    
    print("Precomputing all product sequence PAFs...")
    all_pafs = np.zeros((P, P), dtype=np.int64)  # all_pafs[k][tau]
    
    # k=0 is the Legendre sequence (chi(j)*chi(j+0) = chi(j)^2 = 1 for j!=0, not useful)
    # Actually chi(j)*chi(j) = 1 for j != 0 and chi(0)=1, so the sequence is all 1's. Skip.
    # k=0: the "identity product" gives all-ones sequence (PAF = P at every shift).
    
    for k in range(P):
        seq = np.array([int(leg[j] * leg[(j + k) % P]) for j in range(P)], dtype=np.int8)
        paf = compute_single_paf(seq)
        all_pafs[k] = paf
    
    # Also include the raw Legendre (not a product sequence)
    leg_paf = compute_single_paf(leg)
    
    print("Sampling 4-tuples of sequences...")
    
    rng = np.random.RandomState(123)
    best_cost = float('inf')
    best_config = None
    
    # Strategy: mix Legendre and product sequences
    # Type 0: raw Legendre
    # Type k>0: product sequence chi(j)*chi(j+k)
    
    n_samples = 2_000_000
    for trial in range(n_samples):
        if trial % 500000 == 0:
            print(f"  Trial {trial}/{n_samples}...")
        
        # Choose 4 indices from {0 = Legendre, 1..166 = product sequences}
        # But avoid k=0 product (all ones)
        choices = rng.choice(range(P), size=4, replace=True)  # Allow repeats
        
        cost = 0
        max_paf = 0
        for t in range(1, P):
            total = 0
            for c in choices:
                if c == 0:
                    total += int(leg_paf[t])
                else:
                    total += int(all_pafs[c][t])
            cost += total * total
            if abs(total) > max_paf:
                max_paf = abs(total)
        
        if cost < best_cost:
            best_cost = cost
            best_config = tuple(choices)
            print(f"  Trial {trial}: config={best_config}, L2={cost}, Linf={max_paf}")
            
            if cost == 0:
                print("*** EXACT SOLUTION FOUND! ***")
                break
    
    print(f"\nBest: config={best_config}, L2={best_cost}")
    return best_config, best_cost


def main():
    print("=" * 80)
    print("ALGEBRAIC CONSTRUCTION ATTEMPTS FOR H(668)")
    print("=" * 80)
    
    # Strategy 1: Check all sign combinations for position 0
    print("\n--- Strategy 1: Different zero-position values ---")
    seqs_1, cost_1 = try_different_zero_values()
    
    # Strategy 2: Exhaustive single and double flips from Legendre
    print("\n--- Strategy 2: Exhaustive small perturbations ---")
    best_single, best_pair = exhaustive_small_perturbations()
    
    # Strategy 3: Mixed algebraic search with product sequences  
    print("\n--- Strategy 3: Mixed algebraic (Legendre + product seqs) ---")
    result_3, cost_3 = mixed_algebraic_search()
    
    # Strategy 4: 4 product sequences
    print("\n--- Strategy 4: Four product sequences ---")
    config_4, cost_4 = try_four_product_sequences()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"  Strategy 1 (sign variants): L2={cost_1}")
    print(f"  Strategy 2a (single flip): L2={best_single}")
    print(f"  Strategy 2b (pair flips): L2={best_pair}")
    print(f"  Strategy 3 (mixed algebraic): L2={cost_3}")
    print(f"  Strategy 4 (4 products): L2={cost_4}")
    
    return min(cost_1, best_single, best_pair, cost_3, cost_4)

if __name__ == '__main__':
    main()
