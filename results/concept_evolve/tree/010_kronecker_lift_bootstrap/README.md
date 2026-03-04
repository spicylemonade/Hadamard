# Kronecker Lift Bootstrap for H(668)

## Context
The Kronecker (tensor) product of two Hadamard matrices is Hadamard: if H_m and H_n are Hadamard, then H_m ⊗ H_n is Hadamard of order mn. However, 668 = 4 × 167, and no Hadamard matrix of order 167 exists (167 ≡ 3 mod 4, not divisible by 4).

The key insight is that the Paley Type I construction gives a Hadamard matrix of order 168 = 167+1. By removing one row and column, we get a 167×167 near-orthogonal matrix M' with M'M'ᵀ ≈ 167I. The Kronecker product H₄ ⊗ M' gives a 668×668 near-Hadamard matrix that can be refined.

## Key Challenge
The initial defect from the Kronecker scaffold may be large. The entry-flipping refinement must navigate a complex energy landscape. The scaffold provides structure but the combinatorial correction problem may still be hard.

## Implementation Backlog
- [ ] Implement Paley conference matrix for GF(167)
- [ ] Construct H(168) via bordered construction
- [ ] Extract 167×167 submatrix with minimal defect
- [ ] Form H₄ ⊗ M' scaffold
- [ ] Compute initial Frobenius defect
- [ ] Implement SA with entry-flipping moves
- [ ] Test on smaller orders: H(44) from H(4)⊗M'(11)
- [ ] Multi-start SA with different deleted rows/columns
- [ ] Combine with BP for targeted repair of remaining defects
