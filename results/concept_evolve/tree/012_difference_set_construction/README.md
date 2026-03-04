# Difference Set Construction

## Context
Supplementary difference sets (SDS) in abelian groups provide the most successful path to new Hadamard matrices. Djoković used this approach for orders 764, 188, 388, 436, 580, 988, and many others.

## Two SDS Targets for H(668)

### Direct approach: 4-SDS in Z_167
- Parameters: 4-{167; k1,k2,k3,k4; λ}
- Need: sum of k_i(k_i-1) = λ × 166
- Various parameter choices possible

### Spence construction: 4-SDS in Z_166
- Parameters: 4-{166; 83,83,83,84; 166}
- Z_166 ≅ Z_2 × Z_83 (CRT decomposition)
- λ = 166 verified

## Implementation Backlog
- [x] Compute Spence parameters
- [x] Verify lambda equation
- [ ] Implement SDS verification routine
- [ ] Search for SDS in Z_167 using algebraic initialization
- [ ] Search for Spence SDS in Z_166 using CRT structure
- [ ] Encode as SAT/CSP problem
- [ ] Try Djoković's computer search methodology

## References
- Spence (1975), "Hadamard matrices from relative difference sets"
- Djoković (2008), "Hadamard matrices of order 764 exist"
