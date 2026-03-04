# Spin Glass Ising Relaxation for H(668)

## Context
The connection between Hadamard matrix construction and spin glass physics is deep: the orthogonality energy E(H) = Σ_{r<s}(Σ_k h_{rk}h_{sk})² is a spin-glass Hamiltonian with multi-body interactions and frustration. Finding E=0 ground states is equivalent to constructing a Hadamard matrix.

Parallel tempering (replica-exchange Monte Carlo) is the gold standard for sampling in frustrated systems. By running multiple replicas at different temperatures and exchanging configurations, the algorithm efficiently traverses energy barriers that trap single-temperature MC.

## Key Advantage
Single-spin-flip dynamics is efficient: flipping entry h_{ij} changes only the inner products involving row i, requiring O(n) work per flip. For n=668, this is ~668 operations per update. With structured initialization (from Paley scaffold), the starting energy is already low, dramatically reducing the search effort.

## Implementation Backlog
- [ ] Implement E(H) evaluation with O(n) incremental updates
- [ ] Implement parallel tempering with Metropolis swap criterion
- [ ] Design temperature schedule (geometric, adaptive)
- [ ] Test on orders 44, 92, 156 (known Hadamard matrices)
- [ ] Initialize from Paley scaffold for order 668
- [ ] Monitor replica mixing and adjust temperatures
- [ ] Implement cluster moves (flip entire rows/columns)
- [ ] Run large-scale computation (72+ hours)
