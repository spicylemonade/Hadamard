# Algebraic Geometry Code Evaluation Matrix for H(668)

## Context
Goppa (1982) introduced algebraic geometry codes by evaluating rational functions at points of algebraic curves over finite fields. The resulting generator matrices have controlled distance properties via the Riemann-Roch theorem. The Hermitian curve over GF(q²) has the maximum number of rational points (q³+1) allowed by the Hasse-Weil bound.

For q=167: the Hermitian curve has 167³+1 = 4,657,464 rational points — far more than needed for a 668-dimensional code. By choosing an appropriate divisor G, we can construct a [668, 668] evaluation code whose generator matrix has strong structural properties.

## Key Insight
The orthogonality structure of AG code generator matrices arises from the geometric properties of the underlying curve. If this structure is preserved (approximately) under sign rounding, we get a near-Hadamard matrix that can be refined by local search.

## Implementation Backlog
- [ ] Implement Hermitian curve construction in SageMath
- [ ] Enumerate rational points for small fields (GF(13²), GF(17²))
- [ ] Build evaluation codes of target dimension
- [ ] Implement sign-rounding procedure
- [ ] Measure orthogonality defect
- [ ] Implement local search refinement (entry flipping)
- [ ] Scale to GF(167²)
- [ ] Compare with Reed-Solomon code baseline
