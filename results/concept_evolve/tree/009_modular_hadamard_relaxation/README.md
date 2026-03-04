# Modular Hadamard Relaxation

## Context
Since exact H(668) is unknown, researchers study m-modular approximations where HH^T ≡ 668I (mod m). Higher m is closer to exact.

## Timeline of Progress
- 2001: 32-modular Hadamard matrix of order 668 (unknown author)
- 2025: 64-modular Hadamard matrix (Eliahou, Australasian J. Combinatorics)
- Our work: 4-modular via H(672) truncation (trivial construction)

## Hierarchy
4-modular → 8-modular → 16-modular → 32-modular → 64-modular → ... → exact

## Implementation Backlog
- [x] Construct 4-modular matrix via truncation
- [x] Verify off-diagonal entries are {-4,-2,0,2,4}
- [ ] Implement Eliahou's 64-modular construction
- [ ] Search for higher-modulus approximations
- [ ] Study modular lifting techniques

## References
- Eliahou (2025), "A 64-modular Hadamard matrix of order 668"
