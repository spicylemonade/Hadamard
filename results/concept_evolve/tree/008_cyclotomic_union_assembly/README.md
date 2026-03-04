# Cyclotomic Union Assembly for H(668)

## Context
Cyclotomic classes partition the multiplicative group of a finite field into cosets of a fixed subgroup. When the field is GF(p) for prime p, the N-th cyclotomic classes partition GF(p)* = Z_{p-1} into N classes of equal size (when N | p-1).

For p = 167: p-1 = 166 = 2 × 83. The divisors of 166 are: 1, 2, 83, 166. This gives:
- N=2: two classes (QR and QNR), each of size 83
- N=83: 83 classes, each of size 2
- N=166: 166 classes (individual elements)

Karoui (2026) recently used cyclotomic unions of order 16 over GF(5⁴) to construct a skew-Hadamard matrix of order 1252, demonstrating the viability of this approach for large orders.

## Key Challenge
With only 2 meaningful cyclotomic classes for GF(167), the search space for standard unions is very small and may not contain a valid SDS. Extension to GF(167²) (which has richer cyclotomic structure since 167²-1 = 27888 has many divisors) might be necessary.

## Implementation Backlog
- [ ] Compute primitive root and cyclotomic classes for GF(167)
- [ ] Build Jacobsthal matrix
- [ ] Enumerate all 4-tuples of QR/QNR/QR∪{0}/QNR∪{0} blocks
- [ ] Check SDS condition for each combination
- [ ] Extend to GF(167²) cyclotomic classes
- [ ] Implement Karoui's bordered construction method
- [ ] Test with GF(5⁴) to reproduce the known H(1252) result
