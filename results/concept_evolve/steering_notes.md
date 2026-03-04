# Steering Notes for H(668) Construction

## Problem Status
H(668) is the **smallest open case** of the Hadamard conjecture. As of March 2026, no Hadamard matrix of order 668 = 4 × 167 has been constructed. Key facts:
- 167 is prime, 167 ≡ 3 (mod 4)
- SageMath has constructions up to order 664 but NOT 668
- Eliahou (2025) found only a 64-modular Hadamard matrix of order 668

## Three Concrete Steering Directions

### Direction 1: Goethals-Seidel + Advanced Stochastic Search (PRIORITIZED)
**Rubric items:** item_006, item_007, item_008, item_011, item_015, item_016
**Approach:** Find four ±1 sequences of length 167 satisfying PSD = 668 at all frequencies using:
- Parallel tempering with DFT-guided neighborhoods
- Multi-scale moves (single flip, orbit flip, block swap)
- Extremely long runs (10^8+ iterations)
**Rationale:** This is the standard approach that has succeeded for similar-sized orders. The search space is 2^(4×167) ≈ 10^201 for general sequences, but orbit decomposition and symmetry can reduce this.

### Direction 2: Williamson-Type Symmetric Matrices
**Rubric items:** item_006, item_008, item_010, item_012
**Approach:** Restrict to symmetric circulant matrices (Williamson condition: A^2+B^2+C^2+D^2=668I).
Search space reduced to 2^(4×84) ≈ 10^101 (84 independent positions per symmetric sequence).
**Rationale:** Williamson matrices are known for many values of n but NOT exhaustively searched at n=167.

### Direction 3: SAT+CAS Hybrid (Bright-Kotsireas Method)
**Rubric items:** item_010, item_014, item_015
**Approach:** Encode the SDS existence problem as a SAT instance with CAS-derived propagation.
**Rationale:** Has been used successfully for best matrices and Hadamard matrices of other orders.

## Priority Justification
**Direction 1 is prioritized** because:
1. It's the most flexible (no symmetry assumptions)
2. It can leverage DFT-based cost functions for efficient evaluation
3. Parallel tempering has been shown effective for similar combinatorial problems
4. It can be augmented with orbit structure as extra symmetry breaking
