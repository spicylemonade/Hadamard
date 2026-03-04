# Mathematical Structure of the H(668) Problem

## Key Parameters
- n = 668 = 4 × 167
- p = 167 (prime, p ≡ 3 mod 4)
- Multiplicative group Z_167* has order 166 = 2 × 83

## Why Standard Constructions Fail

### Paley Type I
- Gives H(q+1) for q ≡ 3 (mod 4) prime power
- q = 167: H(168), not H(668)
- Cannot reach 668 by Kronecker since 668/168 ∉ Z

### Paley Type II
- Gives H(2(q+1)) for q ≡ 1 (mod 4) prime power
- q = 167 is ≡ 3 (mod 4), so Paley II doesn't apply directly
- q = 333 (not prime power), q = 166 (even), etc. — no valid q

### Kronecker Products
- 668 = 4 × 167; but 167 is not a valid Hadamard order (167 ≡ 3 mod 4, not 0 or 1 or 2)
- 668 = 2 × 334 = 2 × 2 × 167; same issue
- No factorization into two valid Hadamard orders exists

### Miyamoto Construction (Theorem 6)
- Requires q ≡ 1 (mod 4) and H(q-1) to exist
- For H(668) = H(4×167), need q = 167
- But 167 ≡ 3 (mod 4), so Miyamoto does NOT apply

## Goethals-Seidel / SDS Approach

### PSD Condition
For four ±1 sequences a, b, c, d of length 167:
|â(k)|² + |b̂(k)|² + |ĉ(k)|² + |d̂(k)|² = 668 for all k = 0, ..., 166

### Legendre Baseline Analysis
- Gauss sum G = i√167 (for p ≡ 3 mod 4)
- For Legendre sequence x with x[0]=s: |x̂(k)|² = 1+p = 168 for k≠0, |x̂(0)|² = s² = 1
- Four Legendre copies: PSD(k≠0) = 4×168 = 672, PSD(0) = 4
- Gap: need 668 but get 672, excess of 4 at every non-zero frequency
- This is the fundamental obstruction: 4(p+1) = 4×168 = 672 ≠ 4p = 668

### Row Sum Constraint
- PSD(0) = s₁² + s₂² + s₃² + s₄² where sᵢ = Σⱼ aᵢ[j] (row sums)
- Need s₁² + s₂² + s₃² + s₄² = 668
- All sᵢ must be odd (P=167 entries of ±1)
- Valid decompositions: (25,5,3,3), (23,9,7,3), (23,11,3,3), (21,13,7,3), (21,11,9,5), etc. (10 solutions)

## Group Structure of Z_167*
- Cyclic of order 166 = 2 × 83
- Primitive root: g = 5
- Quadratic residues: C₀ = {g^(2k) : k=0,...,82} (83 elements)
- Quadratic non-residues: C₁ = {g^(2k+1) : k=0,...,82} (83 elements)
- Element of order 2: g^83 = -1 (mod 167) = 166
- Since 83 is prime, only subgroups are {1}, {1,-1}, QR, and Z_167*
- No intermediate subgroups ⟹ no finer cyclotomic decomposition

## Conclusion
The existence of H(668) remains an open problem. The fundamental obstruction is the gap between 4(p+1) = 672 and 4p = 668, which requires finding supplementary difference sets in Z_167 that differ from the Legendre-based structure.
