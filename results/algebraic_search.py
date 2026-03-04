#!/usr/bin/env python3
"""
Algebraic search for H(668) using cyclotomic structure of Z_167.

Key structure:
- P = 167 is prime, P ≡ 3 (mod 4)
- Z_167* has order 166 = 2 × 83
- Since 83 is prime, subgroups of Z_167* are: {1}, H_2={1,166}, H_83=QR, Z_167*
- Cyclotomic classes of order 2: C_0 = QR (83 elements), C_1 = QNR (83 elements)
- Cyclotomic classes of order 166: singletons

APPROACH: Search for supplementary difference sets (SDS) by:
1. Use the coset structure of Z_167* under various subgroups
2. Try finer partitions using higher-order cyclotomic classes
3. Use the DFT/Gauss sum structure to guide construction

For Goethals-Seidel: need 4-{167; k1,k2,k3,k4; lambda} SDS where
lambda = k1+k2+k3+k4 - 167.
"""

import numpy as np
from numpy.fft import fft
from itertools import combinations, product
import time
import sys

P = 167
N = 4 * P

def legendre(a, p=P):
    if a % p == 0: return 0
    return 1 if pow(a, (p-1)//2, p) == 1 else -1

def find_primitive_root(p=P):
    for g in range(2, p):
        if pow(g, (p-1)//2, p) != 1:  # not a QR, so potential generator
            # Verify it generates
            val = 1
            order = 0
            while True:
                val = (val * g) % p
                order += 1
                if val == 1:
                    break
            if order == p - 1:
                return g
    return None

g = find_primitive_root()
print(f"Primitive root mod {P}: {g}")

# Build discrete log table
dlog = {}
val = 1
for i in range(P - 1):
    dlog[val] = i
    val = (val * g) % P

# Cyclotomic classes
QR = frozenset(x for x in range(1, P) if dlog[x] % 2 == 0)
QNR = frozenset(x for x in range(1, P) if dlog[x] % 2 == 1)
print(f"|QR| = {len(QR)}, |QNR| = {len(QNR)}")

# Orbits under multiplication by powers of g
# The orbit of an element a under <g^d> is {a * g^(d*k) mod P : k = 0, 1, ...}
# For d = 2 (QR multiplier): orbits are {a, a*g^2, a*g^4, ...}
# Since g^2 has order 83, each orbit has 83 elements = QR or QNR

def get_orbit(a, d, p=P):
    """Get orbit of a under <g^d> in Z_p*."""
    gen = pow(g, d, p)
    orbit = set()
    val = a
    while val not in orbit:
        orbit.add(val)
        val = (val * gen) % p
    return frozenset(orbit)

# Finer orbits using the negation multiplier (-1)
# Since P ≡ 3 mod 4, -1 is a QNR, so -1 = g^83 
neg1 = pow(g, 83, P)
assert neg1 == P - 1, f"-1 = {neg1}"

# Negation orbits: pairs {a, P-a}
neg_orbits = [{0}]
for j in range(1, (P+1)//2):
    neg_orbits.append(frozenset({j, P-j}))
print(f"Negation orbits: {len(neg_orbits)} (1 + {len(neg_orbits)-1} pairs)")

# Now build the search over unions of negation orbits (Williamson approach)
# This gives symmetric sequences: a[j] = a[P-j]
# Each orbit is a binary choice (in set or not)
# 84 free bits per set × 4 sets = 336 bits total

# For efficiency, let's use the DFT structure.
# For symmetric sequences, DFT values are REAL.
# We need: sum of hat[k]^2 = 668 for each k.

# DFT of a symmetric {±1} sequence:
# hat[k] = a[0] + 2*sum_{j=1}^{83} a[j]*cos(2*pi*j*k/P)
# This is real for symmetric sequences.

# Precompute the cosine basis
cos_basis = np.zeros((84, P), dtype=np.float64)
cos_basis[0, :] = 1.0  # orbit {0}: contributes 1 to all frequencies
for j in range(1, 84):
    # Orbit {j, P-j}: contributes 2*cos(2*pi*j*k/P) to hat[k]
    for k in range(P):
        cos_basis[j, k] = 2 * np.cos(2 * np.pi * j * k / P)

# For a symmetric sequence defined by orbit choices x[0..83] ∈ {±1}:
# hat[k] = sum_{i=0}^{83} x[i] * cos_basis[i, k]
# PSD = sum_{seq=1}^{4} hat[k]^2 = 668 for all k

def compute_symmetric_psd(orbit_values_4):
    """
    orbit_values_4: list of 4 arrays of shape (84,) with ±1 values
    Returns PSD array of shape (P,)
    """
    psd = np.zeros(P)
    for x in orbit_values_4:
        hat = cos_basis.T @ x  # shape (P,)
        psd += hat**2
    return psd

def symmetric_cost(orbit_values_4):
    psd = compute_symmetric_psd(orbit_values_4)
    return np.sum((psd - N)**2)

# Williamson search: simulated annealing over orbit choices
def williamson_sa(n_iterations=5_000_000, seed=42):
    """Simulated annealing for Williamson-type matrices."""
    rng = np.random.RandomState(seed)
    n_orbits = 84
    
    # Initialize with Legendre-based orbits
    chi_orbits = np.ones(n_orbits, dtype=np.float64)
    chi_orbits[0] = 1  # orbit {0}
    for j in range(1, n_orbits):
        chi_orbits[j] = float(legendre(j))
    
    # Start with 4 copies of Legendre, each with slight perturbation
    states = []
    for i in range(4):
        x = chi_orbits.copy()
        mask = rng.random(n_orbits) < 0.1
        x[mask] *= -1
        states.append(x)
    
    # Precompute: for each orbit index, the DFT contribution
    # hat_contribution[i, k] = x[i] * cos_basis[i, k]
    # When we flip orbit i of sequence s:
    # delta_hat_s[k] = -2 * x_s[i] * cos_basis[i, k]
    # delta_psd[k] = new_hat_s[k]^2 - old_hat_s[k]^2
    
    # Compute initial hats and PSD
    hats = [cos_basis.T @ x for x in states]  # 4 arrays of shape (P,)
    psd = sum(h**2 for h in hats)
    cost = np.sum((psd - N)**2)
    
    best_cost = cost
    best_states = [x.copy() for x in states]
    best_psd = psd.copy()
    
    T = 500.0
    T_min = 0.001
    cooling = (T_min / T) ** (1.0 / n_iterations)
    
    accepted = 0
    improved = 0
    
    t0 = time.time()
    
    for it in range(1, n_iterations + 1):
        # Pick random sequence and orbit
        s_idx = rng.randint(4)
        o_idx = rng.randint(n_orbits)
        
        # Current orbit value
        old_val = states[s_idx][o_idx]
        
        # Compute new hat for this sequence
        delta_hat = -2 * old_val * cos_basis[o_idx]  # shape (P,)
        new_hat = hats[s_idx] + delta_hat
        
        # New PSD
        new_psd = psd - hats[s_idx]**2 + new_hat**2
        new_cost = np.sum((new_psd - N)**2)
        
        # Metropolis criterion
        d = new_cost - cost
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-15)):
            states[s_idx][o_idx] = -old_val
            hats[s_idx] = new_hat
            psd = new_psd
            cost = new_cost
            accepted += 1
            
            if cost < best_cost:
                best_cost = cost
                best_states = [x.copy() for x in states]
                best_psd = psd.copy()
                improved += 1
        
        T *= cooling
        
        if cost < 0.5:
            print(f"\n*** SOLUTION FOUND at iteration {it}! ***")
            break
        
        if it % 500_000 == 0:
            elapsed = time.time() - t0
            linf = np.max(np.abs(best_psd - N))
            rate = it / elapsed
            print(f"  {it:>10,} | {elapsed:>7.1f}s | L2={best_cost:>10.1f} | "
                  f"Linf={linf:>7.2f} | T={T:.4f} | "
                  f"acc={accepted/it:.3f} | rate={rate:>8.0f}/s")
    
    elapsed = time.time() - t0
    linf = np.max(np.abs(best_psd - N))
    print(f"\nDone: {elapsed:.1f}s, best L2={best_cost:.1f}, Linf={linf:.2f}")
    print(f"  Accepted: {accepted}/{n_iterations} = {accepted/max(it,1):.3f}")
    print(f"  Improved: {improved}")
    
    return best_states, best_cost, best_psd

# Reconstruct full sequences from orbit values
def orbits_to_sequence(orbit_vals):
    """Convert 84 orbit values to a length-167 symmetric sequence."""
    seq = np.zeros(P, dtype=np.float64)
    seq[0] = orbit_vals[0]
    for j in range(1, 84):
        seq[j] = orbit_vals[j]
        seq[P - j] = orbit_vals[j]
    return seq

def build_hadamard_from_williamson(states):
    """Build and verify Hadamard matrix from Williamson orbit states."""
    from hadamard_core import goethals_seidel_array, verify_hadamard, export_csv
    
    seqs = [orbits_to_sequence(x).astype(np.int8) for x in states]
    H = goethals_seidel_array(seqs[0], seqs[1], seqs[2], seqs[3])
    valid, msg = verify_hadamard(H)
    return H, valid, msg


if __name__ == "__main__":
    print("="*70)
    print("ALGEBRAIC SEARCH FOR H(668) - Williamson Type")
    print("="*70)
    print(f"P = {P}, N = {N}")
    print(f"Searching over 84 orbits × 4 sequences = 336 binary variables")
    print()
    
    # Multi-start search
    overall_best_cost = float('inf')
    overall_best = None
    
    for restart in range(5):
        print(f"\n--- Restart {restart} ---")
        states, cost, psd = williamson_sa(
            n_iterations=3_000_000,
            seed=42 + restart * 12345
        )
        
        if cost < overall_best_cost:
            overall_best_cost = cost
            overall_best = (states, psd)
        
        if cost < 0.5:
            print("SOLUTION FOUND!")
            H, valid, msg = build_hadamard_from_williamson(states)
            print(f"Verification: {msg}")
            if valid:
                from hadamard_core import export_csv
                export_csv(H, "../hadamard_668.csv")
                print("SAVED!")
                sys.exit(0)
    
    print(f"\nOverall best L2 cost: {overall_best_cost:.1f}")
    if overall_best:
        linf = np.max(np.abs(overall_best[1] - N))
        print(f"Overall best Linf: {linf:.2f}")
        print(f"PSD range: [{overall_best[1].min():.1f}, {overall_best[1].max():.1f}]")
