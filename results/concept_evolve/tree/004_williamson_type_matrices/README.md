# Williamson Type Matrices

## Context
Williamson matrices are four symmetric circulant ±1 matrices of order n with A²+B²+C²+D²=4nI. The Williamson array then gives H(4n). For H(668), we need n=167.

## Why This Is Hard
- Known Williamson matrices exist for many n ≤ 63
- No Williamson matrices of order 167 are known
- The search space is 2^84 (palindromic first rows of length 167)
- The sum-of-squares condition is highly constrained

## Implementation Backlog
- [ ] Implement Williamson matrix verification
- [ ] Enumerate palindromic ±1 sequences of length 167
- [ ] Reduce to SAT problem
- [ ] Try constraint propagation with algebraic structure of Z_167

## References
- Hall (1988), "Combinatorial Theory", Chapter 14
- London (2013), "Constructing New Turyn Type Sequences"
