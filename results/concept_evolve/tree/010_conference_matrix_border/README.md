# Conference Matrix Border Construction

## Context
The Jacobsthal matrix Q_167 is a skew conference matrix: zero diagonal, ±1 off-diagonal, QQ^T = 167I - J. Bordering gives H(168). The challenge: extend from H(168) to H(668) = H(4×167).

## Conference Matrix Properties
- Q_167 is skew-symmetric (167 ≡ 3 mod 4)
- Eigenvalues: ±√167·i (purely imaginary)
- Q represents the Paley tournament on GF(167)
- The Paley graph (undirected version for p ≡ 1 mod 4) gives conference matrices for Paley II

## Implementation Backlog
- [x] Construct Q_167 from Legendre symbols
- [x] Verify QQ^T = 167I - J
- [x] Border to get H(168)
- [ ] Explore generalized bordering for larger matrices
- [ ] Study Mathon's conference matrix construction (order pq²+1)
- [ ] Try combined conference + GS approach

## References
- Mathon (1978), "Symmetric conference matrices of order pq²+1"
