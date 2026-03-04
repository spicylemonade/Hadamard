# Tensor Network Contraction for H(668)

## Context
Tensor networks (TN) have revolutionized quantum many-body physics and are increasingly applied to classical optimization problems. The Hadamard matrix construction can be formulated as finding a zero-energy state of a constraint tensor network where nodes enforce pairwise row orthogonality.

The key insight is that the constraint structure of Hadamard orthogonality has bounded interaction range when rows are treated as sequential objects — each new row must be orthogonal to all previous rows, creating a chain-like dependency structure amenable to MPS representation.

## Key Challenge
The bond dimension required to exactly represent the constraint grows exponentially with the number of constrained rows. Approximate contraction with bounded chi introduces errors that may prevent convergence to an exact solution.

## Implementation Backlog
- [ ] Formulate Hadamard orthogonality as tensor network
- [ ] Implement row-wise MPS representation
- [ ] Build DMRG-like sweep algorithm for row optimization
- [ ] Test on order 28 (Hadamard exists, known construction)
- [ ] Measure bond dimension scaling
- [ ] Implement adaptive chi strategy
- [ ] Scale to order 44, then 92
- [ ] Investigate tree tensor network for better scaling
