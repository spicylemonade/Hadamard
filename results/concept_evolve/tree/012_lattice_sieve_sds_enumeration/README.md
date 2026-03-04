# Lattice Sieve SDS Enumeration for H(668)

## Context
Lattice sieving, a key component of the Number Field Sieve for integer factoring, finds smooth values of polynomials by exploiting lattice structure. The SDS autocorrelation conditions for the Goethals-Seidel array define a system of linear constraints that carve out a lattice in high-dimensional integer space.

The connection to lattice-based cryptography is direct: finding a 0/1 vector in a lattice (the SDS problem) is a variant of the Closest Vector Problem (CVP), one of the fundamental problems in computational lattice theory.

## Key Insight
The autocorrelation conditions Σᵢ Nₓᵢ(g) = λ for all g ≠ 0 define 166 linear constraints over the 4×167 = 668 binary variables encoding the four SDS blocks. After LLL/BKZ reduction, the reduced basis vectors may point toward regions of the lattice containing valid 0/1 solutions.

## Implementation Backlog
- [ ] Formulate autocorrelation conditions as integer linear system
- [ ] Construct the autocorrelation lattice for v=23
- [ ] Apply LLL reduction and measure lattice quality
- [ ] Implement CVP solver for 0/1 target
- [ ] Verify that known SDS for v=23 are recoverable
- [ ] Scale to v=47 and measure computational cost
- [ ] Apply BKZ-2.0 for stronger reduction at v=167
- [ ] Combine with Schnorr-Euchner enumeration
- [ ] Integrate cyclotomic constraints to reduce lattice dimension
