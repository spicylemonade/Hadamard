# Construction Feasibility Analysis for H(668)

## Overview
Order 668 = 4 x 167, where p = 167 is prime with p = 3 (mod 4). This document ranks all known construction methods by feasibility.

## Feasibility Table

| Rank | Method | Applicable? | Search Space | Prior Attempts | Recommendation |
|------|--------|-------------|--------------|----------------|----------------|
| 1 | Goethals-Seidel (4 circulants, order 167) | YES | 2^668 general, 2^336 symmetric | Prior branch SA (200K iter, failed) | **HIGH PRIORITY** - best chance |
| 2 | Williamson (4 symmetric circulants) | YES (subset of GS) | 2^336 | Prior branch SA (100K iter, failed) | **HIGH PRIORITY** - reduced space |
| 3 | SAT+CAS Hybrid (Bright-Kotsireas) | YES | Varies with encoding | None at v=167 | **MEDIUM PRIORITY** - proven for similar |
| 4 | Spence SDS 4-{334; 167,167,167,168; 334} | YES (Theorem 7) | ~2^668 in Z_334 | None | **MEDIUM PRIORITY** - different space |
| 5 | T-matrices + Williamson | MAYBE | Depends on factorization | None for v=167 | **LOW PRIORITY** - no clean factoring |
| 6 | Two-circulant core | YES | ~2^668 | None at v=167 | **LOW PRIORITY** - not known to help |
| 7 | Propus (symmetric Hadamard) | YES | ~2^336 | None at v=167 | **LOW PRIORITY** - extra symmetry |
| 8 | Miyamoto (Theorem 6) | **NO** | N/A | N/A | **NOT APPLICABLE** |
| 9 | Paley Type I | **NO** | N/A | N/A | Gives H(168) not H(668) |
| 10 | Paley Type II | **NO** | N/A | N/A | Requires q=1 mod 4 |
| 11 | Kronecker product | **NO** | N/A | N/A | No valid factorization |

## Detailed Analysis

### 1. Goethals-Seidel Array (Rank 1 -- HIGHEST PRIORITY)

**Construction:** Find four +/-1 sequences a, b, c, d of length 167 such that
|a_hat(k)|^2 + |b_hat(k)|^2 + |c_hat(k)|^2 + |d_hat(k)|^2 = 668 for all k != 0.

Then the GS array H = [[A, BR, CR, DR], [-BR, A, D^TR, -C^TR], [-CR, -D^TR, A, B^TR], [-DR, C^TR, -B^TR, A]]
is a Hadamard matrix of order 668.

**Search space:** 2^(4*167) = 2^668 ~ 10^201 for unrestricted sequences.

**Baseline:** Four Legendre sequences give PSD = 672 at all non-zero frequencies (gap of 4).
This is the nearest known near-miss.

**Prior attempts:** Prior branch tried SA with only 200K iterations -- vastly insufficient.
Our improved SA with 50M+ iterations converges to local minima at L2 ~ 390K-444K (Linf ~ 120-165).

**Assessment:** The GS approach has succeeded for most known Hadamard matrices of similar size.
The flat PSD excess of 4 from Legendre sequences suggests a structured obstruction.
Multi-start SA with DFT-guided neighborhoods and parallel tempering is the best available strategy.

### 2. Williamson-Type Symmetric Matrices (Rank 2)

**Construction:** Restrict GS to symmetric circulant sequences: a[i] = a[p-i].
This gives Williamson's equation A^2 + B^2 + C^2 + D^2 = 668*I.

**Search space:** Each symmetric sequence has ceil(167/2) = 84 independent entries, so 2^(4*84) = 2^336 ~ 10^101.

**Assessment:** Williamson matrices are known for many primes but not exhaustively for p=167.
The reduced search space is an advantage, but Williamson matrices do NOT exist for all valid orders.
It is possible that no Williamson-type H(668) exists even if H(668) itself exists.

### 3. SAT+CAS Hybrid (Rank 3)

**Construction:** Encode the 4-SDS existence problem as a Boolean satisfiability instance.
For each element x in Z_167 and each block i in {1,2,3,4}, create a Boolean variable x_i.
Add clauses encoding the intersection condition: for each non-zero difference d, the sum
|D_1 cap (D_1+d)| + |D_2 cap (D_2+d)| + |D_3 cap (D_3+d)| + |D_4 cap (D_4+d)| = lambda.

Use symmetry-breaking from the automorphism group of Z_167 to prune.
Integrate CAS-derived propagation (DFT-based filtering).

**Assessment:** Bright et al. (2021) used this for best matrices up to order 208.
Scaling to v=167 may be challenging (167 variables per block, 668 total).
However, strong symmetry breaking from the multiplier group (order 166) can help.

### 4. Spence SDS (Rank 4)

**Construction:** Find 4-{334; 167,167,167,168; 334} supplementary difference sets in Z_334 = Z_2 x Z_167.
By Theorem 7 of Cati-Pasechnik, this yields H(668).

**Assessment:** The group Z_334 has richer structure than Z_167 alone, potentially opening new search paths.
However, the search space is comparable or larger (334 elements, 4 blocks).
No prior attempts at this specific parameter set are known.

### 5. T-matrices + Williamson (Rank 5)

**Construction:** Use T-matrices of order t combined with Williamson matrices of order 167/t.
Requires 167 = t * s for suitable t, s -- but 167 is prime, so t=1 or t=167.
t=1 reduces to standard Williamson; t=167 requires T-matrices of order 167.

**Assessment:** Not useful for prime p=167 unless T-matrices of order 167 are known (they are not).

### 6-7. Two-Circulant Core and Propus (Rank 6-7)

**Assessment:** These are alternative structural approaches. Two-circulant core uses a different matrix layout.
Propus adds the constraint that H is symmetric. Both have comparable or larger search spaces
and no known advantage for v=167.

### 8. Miyamoto Construction (NOT APPLICABLE)

**Why it fails:** Miyamoto's Theorem 6 requires q = 1 (mod 4) with q prime and H(q-1) existing.
For H(668) = H(4*167), we need q=167, but 167 = 3 (mod 4). **Dead end.**

### 9-11. Paley I, Paley II, Kronecker (NOT APPLICABLE)

- Paley I with q=167 gives H(168), not H(668)
- Paley II requires q = 1 (mod 4); 167 = 3 (mod 4)
- Kronecker: 668 cannot be factored into two valid Hadamard orders

## Summary

The only viable approaches for H(668) are:
1. **Goethals-Seidel with stochastic search** (general or Williamson-restricted)
2. **SAT+CAS hybrid** encoding the SDS problem
3. **Spence SDS** over Z_334

All standard algebraic constructions (Paley, Kronecker, Miyamoto) provably fail for 668.
The fundamental mathematical obstruction is that 167 = 3 (mod 4) eliminates most algebraic shortcuts,
and the multiplicative group Z_167* has order 166 = 2*83 with no useful intermediate subgroups
for cyclotomic decomposition.

## References
- Cati & Pasechnik 2024, arXiv:2411.18897
- Bright et al. 2021, arXiv:1907.04987
- Djokovic & Kotsireas 2018, arXiv:1802.00556
- Eliahou 2025, hal-05393934
- Seberry & Yamada 2020, Hadamard Matrices (monograph)
