# T-Sequence Decomposition

## Context
T-sequences provide an alternative to circulant sequences for Hadamard construction. They have disjoint supports (each position is nonzero in exactly one of the four sequences) and zero-sum non-periodic autocorrelation.

## Cooper-Wallis Theorem
If T-matrices of order n exist and Williamson-type matrices of order w exist, then H(4nw) exists. For H(668): need nw = 167 (prime), so (n,w) ∈ {(1,167), (167,1)}.

## Kharaghani Construction
T-sequences can be constructed from special properties of Z_p for certain primes p. The key reference is Kharaghani (2005) which used T-sequences for H(428).

## Implementation Backlog
- [ ] Implement Kharaghani's T-sequence construction
- [ ] Test for small primes (7, 11, 23, 47)
- [ ] Attempt construction for p = 167
- [ ] Search for T-sequences via constraint programming

## References
- Kharaghani & Tayfeh-Rezaie (2005)
- Cooper & Wallis (1972), "A construction for Hadamard arrays"
