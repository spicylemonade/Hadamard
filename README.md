# Hadamard Matrix of Order 668: Computational Investigation

## Overview

This repository contains a comprehensive investigation into constructing a Hadamard matrix of order 668, which is the **smallest order for which no Hadamard matrix construction is known** (Cati & Pasechnik, 2024).

## Key Results

- **No exact H(668) was found.** This is consistent with 668 being a genuinely open case of the Hadamard conjecture.
- The **best candidate matrix** is provided in `hadamard_668.csv` (668x668, ±1 entries).
- This matrix satisfies HH^T = 668I + E where off-diagonal entries of E are in {0, -4}, the best known approximation.
- Approximately **500 million candidate evaluations** were performed across multiple search strategies.

## Output

- **`hadamard_668.csv`** — Best candidate 668x668 ±1 matrix (near-Hadamard, Linf error = 4)
- **`results/report.md`** — Full research report (5000+ words)
- **`sources.bib`** — Bibliography with 26 references

## Mathematical Background

Order 668 = 4 × 167, where:
- 167 is prime with 167 ≡ 3 (mod 4)
- The multiplicative group Z₁₆₇* has order 166 = 2 × 83 (sparse subgroup structure)
- All standard constructions (Paley I/II, Kronecker, Miyamoto, T-sequences) provably fail
- The only viable approach is the Goethals-Seidel array with stochastic search

## Methods Attempted

1. **Legendre baseline** — 4 copies of Legendre symbol sequence (PSD = 672, gap = 4)
2. **Simulated annealing** — Multiple variants with incremental FFT updates
3. **Parallel tempering** — 8-16 temperature replicas
4. **DFT-guided moves** — Frequency-targeted optimization
5. **Multi-flip SA** — Simultaneous multi-bit perturbations
6. **Williamson symmetric search** — Reduced search space (2^336)
7. **Row-sum-targeted initialization** — 10 valid row sum decompositions
8. **Spence SDS analysis** — 4-{334; 167,167,167,168; 334} parameters
9. **Miyamoto construction** — Proven inapplicable (167 ≡ 3 mod 4)

## Repository Structure

```
hadamard_668.csv          # Best candidate matrix (668x668 CSV)
research_rubric.json      # Research progress tracking
sources.bib               # Bibliography (26 entries)
results/
  report.md               # Full research report
  hadamard_core.py        # Core infrastructure (GS array, verification)
  build_best_candidate.py # Script to build and export the candidate matrix
  final_intensive_search.py  # Final intensive SA search
  targeted_search.py      # Targeted row-sum-preserving search
  intensive_search.py     # Multi-start SA with Numba
  baseline_legendre.py    # Legendre symbol baseline analysis
  gs_search_framework.py  # GS orbit search framework
  metaheuristic_search.py # Advanced metaheuristic methods
  analysis/               # Mathematical analysis documents
  experiments/            # Experiment logs and results
  concept_evolve/         # Concept tree exploration
figures/                  # Generated plots
```

## Reproducing Results

```bash
# Build the best candidate matrix
python3 results/build_best_candidate.py

# Run the intensive search (600s budget)
python3 results/final_intensive_search.py 600

# Run targeted search with row-sum preservation
python3 results/targeted_search.py 240 100
```

## References

See `sources.bib` for the complete bibliography. Key references:
- Cati & Pasechnik (2024), arXiv:2411.18897 — Database confirming 668 as first unknown
- Eliahou (2025), hal-05393934 — 64-modular H(668)
- Djokovic & Kotsireas (2018), arXiv:1802.00556 — GS difference families
- Bright et al. (2021), arXiv:1907.04987 — SAT+CAS methods
