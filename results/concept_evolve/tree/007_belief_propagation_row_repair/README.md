# Belief Propagation Row Repair for H(668)

## Context
Belief propagation (BP) is a message-passing algorithm that has been spectacularly successful in decoding LDPC codes and solving constraint satisfaction problems. The Hadamard orthogonality condition — that every pair of rows has inner product zero — is structurally similar to parity-check constraints in coding theory.

The factor graph for H(668) has 668×668 = 446,224 variable nodes (matrix entries) and C(668,2) = 222,778 factor nodes (pairwise orthogonality constraints). Each factor connects to 668 variables (one from each column).

## Key Challenge
The factor graph has many short cycles (every pair of factor nodes shares 668 variable nodes), which degrades BP convergence. Damped/scheduled BP and survey propagation (from spin glass theory) are needed to handle this.

## Implementation Backlog
- [ ] Build factor graph representation of Hadamard orthogonality
- [ ] Implement standard BP message updates
- [ ] Add damping and momentum for cycle handling
- [ ] Implement decimation (fix confident variables)
- [ ] Implement survey propagation variant
- [ ] Test on small known orders (12, 20, 28)
- [ ] Initialize from near-Hadamard (Paley scaffold) for order 668
- [ ] Compare convergence with SA baseline
