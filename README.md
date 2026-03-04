# Hadamard Matrix of Order 668: Computational Investigation

## Overview

This repository contains a comprehensive investigation into constructing a Hadamard matrix of order 668, which is the **smallest order for which no Hadamard matrix construction is known** (Cati & Pasechnik, 2024).

## Key Results

- **No exact H(668) was found.** This is consistent with 668 being a genuinely open case of the Hadamard conjecture.
- The **best near-miss matrix** is provided in `near_miss_668.csv` (668x668, ±1 entries).
- This matrix satisfies HH^T = 668I + E where off-diagonal entries of E are in {0, -4} (max error = 4).
- Approximately **3.9 billion candidate evaluations** were performed across 9 search strategies.
- PAF-direct SA achieved the best L2 cost (1,984) but with Linf = 8; the Legendre baseline has the best Linf = 4.

## Output

- **`near_miss_668.csv`** — Best near-miss 668x668 ±1 matrix (Legendre GS, Linf error = 4). **NOT a valid Hadamard matrix.**
- **`results/report.md`** — Full research report
- **`research_paper.tex`** — LaTeX paper documenting the investigation
- **`sources.bib`** — Bibliography with verified references

## Mathematical Background

Order 668 = 4 x 167, where:
- 167 is prime with 167 = 3 (mod 4)
- The multiplicative group Z_167* has order 166 = 2 x 83 (sparse subgroup structure)
- All standard constructions (Paley I/II, Kronecker, Miyamoto, T-sequences) provably fail
- The only viable approach is the Goethals-Seidel array with stochastic search

## Methods Attempted

| Method | Evaluations | Best L2 | Best Linf |
|--------|------------|---------|-----------|
| Legendre baseline | 1 | 2,656 | 4 |
| PAF-direct SA (Numba) | 2.7e9 | 1,984 | 8 |
| Fast SA (random init) | 5.0e7 | 390,000 | 120 |
| Row-sum constrained SA | 4.7e8 | 448,896 | 150 |
| Skew + symmetric SA | 6.1e8 | 2,148,288 | 251 |
| Williamson symmetric | 5.0e6 | 450,000 | 130 |
| Parallel tempering | 5.0e5 | 600,000 | 150 |
| Multi-start SA (Legendre) | 1.0e6 | 623,136 | 154 |
| Submatrix from H(672) | 1.0e3 | 2,656 | 4 |
| MILP flip optimization | 1 | N/A | N/A |
| **Total** | **~3.9e9** | | |

## Repository Structure

```
near_miss_668.csv         # Best near-miss matrix (668x668 CSV, NOT Hadamard)
research_paper.tex        # LaTeX paper
research_rubric.json      # Research progress tracking (26/26 items completed)
sources.bib               # Bibliography
peer_review.md            # Peer review feedback
results/
  report.md               # Full research report
  hadamard_core.py        # Core infrastructure (GS array, verification)
  intensive_paf_search.py # PAF-direct SA with Numba (best L2=1984)
  turbo_search.py         # Multi-strategy SA with DFT updates
  skew_symmetric_search.py # Skew+symmetric GS search
  optimized_search.py     # Row-sum constrained SA
  build_best_candidate.py # Script to build and export the near-miss matrix
  solution_sequences.npz  # Legendre generating sequences
  best_near_miss.npz      # PAF-optimized sequences (L2=1984)
  analysis/               # Mathematical analysis documents
  experiments/            # Experiment logs and results
  concept_evolve/         # Concept tree exploration
figures/                  # Generated plots
```

## Reproducing Results

```bash
# Build the best near-miss matrix (Legendre GS)
python3 results/build_best_candidate.py

# Run PAF-direct search (best single method)
python3 results/intensive_paf_search.py

# Run multi-strategy search
python3 results/turbo_search.py
```

## References

See `sources.bib` for the complete bibliography. Key references:
- Cati & Pasechnik (2024), arXiv:2411.18897 -- Database confirming 668 as first unknown
- Eliahou (2025), AJOC 93(2) -- 64-modular H(668)
- Bright, Djokovic, Kotsireas & Ganesh (2019) -- SAT+CAS methods
- Djokovic & Kotsireas (2018), arXiv:1802.00556 -- GS difference families
