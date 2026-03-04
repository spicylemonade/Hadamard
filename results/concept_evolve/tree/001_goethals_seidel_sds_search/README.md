# Goethals-Seidel SDS Search for H(668)

## Context
Order 668 = 4 × 167 is the smallest open order in the Hadamard conjecture (as of 2024). The Goethals-Seidel array is a 4×4 block construction that produces Hadamard matrices from four supplementary difference sets (SDS) over a group of order v = n/4. For n = 668, we need SDS over Z_167.

Since 167 is prime with 167 ≡ 7 mod 8, the multiplicative group GF(167)* = Z_166 = Z_2 × Z_83 provides the automorphism structure. Đoković, Golubitsky, and Kotsireas have used this approach successfully for orders up to 2524.

## Key Challenge
The search space for four blocks in Z_167 is approximately (2^167)^4 ≈ 10^201. Even with orbit reduction (factor ~166), the space remains intractable without further structural constraints (symmetry, skewness, or propus-type restrictions).

## Implementation Backlog
- [ ] Implement GS array orthogonality checker with FFT-based autocorrelation
- [ ] Build orbit reduction module for Aut(Z_167)
- [ ] Implement meet-in-the-middle matching with hashing
- [ ] Test with known SDS for v=47, v=59 (known orders)
- [ ] Apply propus constraint (X2=X3, X0 symmetric)
- [ ] GPU-parallelize the matching step
- [ ] Run full search for v=167
