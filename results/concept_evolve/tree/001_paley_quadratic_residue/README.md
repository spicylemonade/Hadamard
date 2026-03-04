# Paley Quadratic Residue Construction

## Context
The Paley Construction I uses quadratic residues in GF(p) for p ≡ 3 mod 4 (prime) to build a skew Hadamard matrix of order p+1. For p=167, this gives H(168), the core building block for any approach to H(668) = H(4×167).

## Key Facts
- 167 is prime, 167 ≡ 3 (mod 4)
- Primitive root mod 167: g = 5
- Z*_167 has order 166 = 2 × 83
- 83 quadratic residues, 83 quadratic non-residues
- Jacobsthal matrix Q is 167×167, skew-symmetric
- Q*Q^T = 167*I - J (conference matrix property)
- H(168) = [[1, 1^T], [-1, Q+I]] is a verified Hadamard matrix

## Implementation Backlog
- [x] Construct Jacobsthal matrix Q of order 167
- [x] Verify Q*Q^T = 167*I - J
- [x] Construct H(168) via Paley I
- [x] Verify H(168)*H(168)^T = 168*I
- [ ] Analyze spectral properties of Q
- [ ] Try modified Paley constructions for H(668)
- [ ] Combine with Goethals-Seidel array approach

## References
- Paley (1933), "On Orthogonal Matrices"
- Cati & Pasechnik (2024), "A database of constructions of Hadamard matrices"
