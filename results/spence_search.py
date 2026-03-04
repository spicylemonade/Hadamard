#!/usr/bin/env python3
"""
Spence SDS construction analysis for H(668).
Item 012 of research rubric.

Analyzes the Spence construction: 4-{334; 167,167,167,168; 334} SDS in Z_334 = Z_2 x Z_167.
"""

import numpy as np
from numpy.fft import fft
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from hadamard_core import legendre_symbol, P, N

np.random.seed(42)


def analyze_spence_parameters():
    """
    Spence construction (Theorem 7 from Cati-Pasechnik 2024):
    For H(4p) via Spence, need 4-{2p; k1,k2,k3,k4; lambda} SDS in Z_{2p}.
    
    For p=167: 4-{334; k1,k2,k3,k4; lambda} SDS in Z_334.
    The standard parameter set: k1=k2=k3=167, k4=168, lambda=334.
    """
    v = 2 * P  # 334
    k_list = [P, P, P, P + 1]  # [167, 167, 167, 168]
    lam = v  # lambda = 334
    
    # Verify parameter condition: sum(ki(ki-1)) = lambda(v-1)
    lhs = sum(k * (k - 1) for k in k_list)
    rhs = lam * (v - 1)
    
    print(f"Spence SDS parameters for H({N}):")
    print(f"  v = {v} = Z_2 x Z_{P}")
    print(f"  k = {k_list}")
    print(f"  lambda = {lam}")
    print(f"  Parameter check: sum(ki(ki-1)) = {lhs}, lambda(v-1) = {rhs}")
    print(f"  Valid: {lhs == rhs}")
    
    return v, k_list, lam


def analyze_group_structure():
    """Analyze Z_334 = Z_2 x Z_167 structure."""
    v = 2 * P
    
    print(f"\nGroup structure of Z_{v}:")
    print(f"  Z_{v} = Z_2 x Z_{P}")
    print(f"  Order: {v}")
    print(f"  Decomposition: elements (a, b) where a in Z_2, b in Z_{P}")
    
    # Automorphism group
    # Aut(Z_2 x Z_167) = Aut(Z_2) x Aut(Z_167) = {id} x Z_166
    # Since Aut(Z_2) is trivial and Aut(Z_167) = Z_166
    print(f"  Aut(Z_{v}) = Aut(Z_2) x Aut(Z_{P}) = {{id}} x Z_166")
    print(f"  |Aut| = 166")
    
    # Multiplier group for SDS search: use multipliers from Z_167*
    # Acting on Z_334 as (a, b) -> (a, m*b mod 167) for m in Z_167*
    print(f"  Useful multipliers: {{1, -1}} subset Z_{P}*, giving {(P-1)//2} orbits on Z_{P}*")
    
    # Number of orbits on Z_334 \ {0}
    # Elements: (0,1),...,(0,166), (1,0), (1,1),...,(1,166)
    # Under (a,b) -> (a, -b):
    # (0,b) and (0,-b) are paired: 83 pairs from Z_2=0 side
    # (1,0) is fixed
    # (1,b) and (1,-b) are paired: 83 pairs from Z_2=1 side
    # Plus (0,0) excluded
    # Total non-zero: 333. Fixed: (1,0). Pairs: 83+83 = 166. Total orbits: 1 + 166 = 167.
    print(f"  Orbits of Z_{v}\\{{0}} under {{1,-1}}: 167")
    print(f"    - 83 pairs from (0, b)/(0, -b) for b != 0")
    print(f"    - 1 fixed point: (1, 0)")
    print(f"    - 83 pairs from (1, b)/(1, -b) for b != 0")
    
    # Search space comparison
    print(f"\n  Search space comparison:")
    print(f"    GS over Z_{P}: 2^{4*P} ~ 10^{4*P*0.301:.0f} (general)")
    print(f"    Spence over Z_{v}: 2^{4*v} ~ 10^{4*v*0.301:.0f} (general)")
    print(f"    Spence with orbits: comparable to GS approach")
    print(f"    Conclusion: Spence does NOT reduce search complexity for p={P}")


def main():
    print("=" * 60)
    print("SPENCE SDS CONSTRUCTION ANALYSIS FOR H(668)")
    print("=" * 60)
    
    v, k_list, lam = analyze_spence_parameters()
    analyze_group_structure()
    
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print("=" * 60)
    print(f"The Spence construction requires 4-{{334; 167,167,167,168; 334}} SDS in Z_334.")
    print(f"The group Z_334 = Z_2 x Z_167 has richer structure than Z_167 alone,")
    print(f"but the search space is LARGER (334 elements vs 167).")
    print(f"The parameter condition is valid, but no such SDS is known to exist.")
    print(f"This approach is ranked below direct GS search due to larger search space.")
    print(f"Feasibility score: 3/10 (valid construction but no advantage over GS)")


if __name__ == "__main__":
    main()
