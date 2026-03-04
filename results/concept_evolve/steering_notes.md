# Steering Notes for H(668) Construction

## Problem Status
H(668) is the **smallest open case** of the Hadamard conjecture. As of March 2026, no Hadamard matrix of order 668 = 4 × 167 has been constructed. Key facts:
- 167 is prime, 167 ≡ 3 (mod 4)
- SageMath (Cati-Pasechnik 2024/2025 database) confirms 668 is the smallest unknown order
- Eliahou (2025) found only a 64-modular Hadamard matrix of order 668
- The next unknown orders are 716, 892, 1132
- φ(167) = 166 = 2 × 83 (sparse subgroup structure is a key obstacle)

## Three Concrete Steering Directions

### Direction 1: Goethals-Seidel + Intensive Stochastic Search (PRIORITIZED)
**Rubric items:** item_006, item_007, item_008, item_011, item_015, item_016, item_017
**Approach:** Find four ±1 sequences of length 167 satisfying PSD = 668 at all frequencies.
**Findings (this session):**
- Legendre baseline gives PAF_total(τ) = -4 for ALL nonzero shifts (uniformly flat)
- Best SA result: L2=1984, Linf=8 (48/166 shifts with PAF=0) — but worse Linf than Legendre
- The Legendre GS matrix has HH^T = 668I + E with off-diagonal entries only in {-4, 0}
- 24.9% of off-diagonal entries are -4, 75.1% are 0
- Submatrix extraction from H(672)=H(4)⊗H(168) gives identical quality (max error 4)
- MILP on linearized flip model: solution exists but nonlinear effects dominate
- Row sum constraint s₁²+s₂²+s₃²+s₄²=668 with odd sᵢ: 10 valid quadruples found
**Assessment:** The gap of 4 appears to be a deep algebraic barrier related to Z₁₆₇* structure.

### Direction 2: Mixed Skew/Symmetric Sequences
**Rubric items:** item_008, item_011, item_012
**Approach:** Use Legendre (skew) as one sequence plus 3 symmetric sequences.
**Findings:**
- Reduces search space from 668 to 249 binary variables
- Best result: L2≈2.1M (much worse than Legendre baseline)
- The reduced search space doesn't contain solutions close to the Legendre quality
**Assessment:** Less promising than Direction 1. The symmetry constraints may be too restrictive.

### Direction 3: Algebraic/SAT Approaches
**Rubric items:** item_003, item_010, item_016
**Approach:** SAT encoding, modular lifting, or algebraic number theory.
**Findings:**
- The SDS problem at v=167 has 166 quadratic constraints over 668 binary variables
- 64-modular Hadamard exists (Eliahou 2025) but lifting to exact appears extremely hard
- The flip impact matrix A has rank ≤ 166 with columns in {-1,0,1}^166
- Sum of all columns = (1,...,1) — but the all-ones solution corresponds to flipping all QNR positions, giving a trivial (constant) sequence
**Assessment:** Algebraic methods reveal the structure but don't break the barrier.

## Priority Justification
**Direction 1 remains prioritized** because:
1. The Legendre baseline is provably the closest known starting point (PAF uniformly -4)
2. Any solution must break the Z₁₆₇* algebraic structure in a non-trivial way
3. This is genuinely an open problem — no construction method is known to work for H(668)
4. The most likely path to solution (if one exists computationally) is extensive parallel search with smart initialization

## Key Insight from This Session
The gap between the Legendre baseline (PAF=-4 everywhere) and exact solution (PAF=0) appears to be a **fundamental algebraic barrier** rather than a computational one. The error is not random but perfectly structured: every nonzero shift has the same PAF deviation. This suggests that resolving H(668) may require a genuinely new mathematical idea rather than more computation.
