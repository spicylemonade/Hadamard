# Complementary Sequences for H(668)

## Context
For the GS array construction of H(668), we need four ±1 sequences of length 167 whose periodic autocorrelations sum to zero at all nonzero lags.

## DC Constraint
The DC components (row sums) must satisfy: a1² + a2² + a3² + a4² = 668.
Valid odd decompositions include:
- (1, 1, 15, 21): 1+1+225+441 = 668
- (1, 9, 15, 19): 1+81+225+361 = 668
- (3, 3, 17, 19): 9+9+289+361 = 668

## Legendre Sequence Analysis
- The Legendre sequence of GF(167) has PAF(j) = -1 for all j ≥ 1
- PSD: |S(0)|² = 1, |S(k)|² = 168 for k ≥ 1
- Four Legendre copies give total PAF = -4 (need 0)

## Implementation Backlog
- [x] Compute Legendre sequence PAF
- [x] Enumerate 4-square decompositions of 668
- [x] Analyze frequency domain constraints
- [ ] Search for near-complementary sequence sets
- [ ] Try Golay pair approach for 2-sequence sub-problem
- [ ] Test with turbo optimization (alternating sequences)

## References
- Golay (1961), "Complementary series"
- Kharaghani & Tayfeh-Rezaie (2005), "A Hadamard matrix of order 428"
