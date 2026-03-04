# Kronecker Tensor Product Construction

## Context
The Kronecker product H_m ⊗ H_n of Hadamard matrices yields H(mn). For 668 = 4×167, no H(167) exists (167 is odd, not 1 or 2). The obstruction is fundamental: Hadamard matrices exist only for orders 1, 2, and multiples of 4.

## Key Results
- H(672) = H(4) ⊗ H(168) verified as 672×672 Hadamard matrix
- Truncated H(672)[:668,:668] has:
  - Diagonal entries: all 668 (correct)
  - Off-diagonal entries: {-4, -2, 0, 2, 4}
  - Max |off-diagonal|: 4
  - Mean |off-diagonal|: 1.48
  - This is a 4-modular Hadamard matrix of order 668

## Files Produced
- `results/hadamard_672.csv`: Verified 672×672 Hadamard matrix
- `results/hadamard_168.csv`: Verified 168×168 Hadamard matrix (Paley I)
- `results/hadamard_668_approximate.csv`: 668×668 approximate (4-modular)

## Implementation Backlog
- [x] Construct H(4)
- [x] Construct H(168) via Paley I
- [x] Compute H(672) = H(4) ⊗ H(168)
- [x] Verify H(672)*H(672)^T = 672*I
- [x] Truncate to 668×668
- [x] Analyze off-diagonal distribution
- [ ] Try column/row selection optimization for better truncation
- [ ] Explore bordered constructions

## References
- Sylvester (1867), "Thoughts on inverse orthogonal matrices"
