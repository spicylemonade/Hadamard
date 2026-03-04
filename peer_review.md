# Peer Review: "Constructing a Hadamard Matrix of Order 668: A Computational Investigation via the Goethals--Seidel Array"

**Reviewer:** Automated Peer Review (Nature/NeurIPS standards)  
**Date:** 2026-03-04  
**Paper:** `research_paper.tex` / `research_paper.pdf` (13 pages, 32 references)

---

## Criterion Scores

| # | Criterion | Score (1-5) | Comments |
|---|-----------|:-----------:|---------|
| 1 | **Completeness** | 5 | All required sections present: Abstract, Introduction, Related Work, Background & Preliminaries, Method, Experimental Setup, Results, Discussion, Conclusion, Acknowledgments, References. |
| 2 | **Technical Rigor** | 4 | Methods are well-described with formal equations (PSD condition, cost function, SA algorithm). Proofs of classical construction inapplicability are clear. Minor issue: discrepancy between paper claims and actual experiment data (see below). |
| 3 | **Results Integrity** | 3 | Significant discrepancies between paper claims (Table 1) and `results/experiments/experiment_log.json`. See detailed analysis below. Figures are consistent with the general narrative but not with specific numbers in the paper. |
| 4 | **Citation Accuracy** | 2 | Multiple citations with incorrect metadata. Several have wrong titles, authors, years, volumes, pages, or DOIs. One DOI resolves to an entirely different paper. See full verification report below. |
| 5 | **Compilation** | 5 | PDF compiles successfully. 13-page well-formatted document with proper LaTeX (natbib, algorithm, booktabs, subcaption). No compilation errors. |
| 6 | **Writing Quality** | 5 | Excellent academic prose. Clear logical flow from problem statement through methods to negative results and structural analysis. Professional tone throughout. Well-structured sections and subsections. |
| 7 | **Figure Quality** | 3 | Figures are functional but mixed quality. The Legendre PSD figure (Fig 1) and convergence plot (Fig 2) are acceptable with proper labels, legends, and color coding. The method comparison bar chart (Fig 3b) uses a reasonable color palette. However, the search landscape figure (Fig 3a) uses default matplotlib bar styling with no colormap variation, and the orbit structure figure (Fig 4) is quite basic with a trivial bar chart (all bars identical height). For a top venue, figures should be more polished. |

---

## Overall Verdict: **REVISE**

---

## Detailed Findings

### 1. Results Integrity Issues

**Critical discrepancy: Paper claims vs. actual experiment data.**

The paper's Table 1 reports:

| Method | Paper Claims (Evaluations) | Paper Claims (Best L2) | Paper Claims (Best Linf) |
|--------|---------------------------|------------------------|--------------------------|
| Standard SA | 1.5 x 10^8 | 403,000 | 122 |
| Parallel tempering | 1.0 x 10^8 | 400,000 | 120 |
| DFT-guided moves | 8.0 x 10^7 | 405,000 | 125 |
| Multi-flip SA | 7.0 x 10^7 | 411,000 | 130 |
| Row-sum targeted | 5.0 x 10^7 | 401,000 | 121 |
| **Total** | **~5 x 10^8** | | |

However, `results/experiments/experiment_log.json` records:

| Method | Actual Evaluations | Actual Best L2 | Actual Best Linf |
|--------|--------------------|----------------|------------------|
| GS Orbit Search | 300,000 | 577,152 | 147 |
| Multi-start SA (Legendre) | 1,000,000 | 623,136 | 154 |
| Fast SA (random, 10 restarts) | 50,000,000 | 390,000 | 120 |
| Williamson SA | 5,000,000 | 450,000 | 130 |
| Parallel Tempering | 500,000 | 600,000 | 150 |
| **Total** | **~56,800,000** | | |

**Key discrepancies:**
1. **Total evaluations**: Paper claims ~5 x 10^8; actual data shows ~5.7 x 10^7 (roughly 10x inflation).
2. **Parallel tempering**: Paper claims L2=400,000, Linf=120; actual data shows L2=600,000, Linf=150.
3. **Missing methods**: Paper describes "DFT-guided moves" and "Row-sum targeted restarts" as separate experiments with specific results; neither appears in the experiment log.
4. **Convergence figure**: Figure 2 shows methods plateauing at ~500K-600K (consistent with experiment log), while the paper text claims convergence at ~400K.
5. **Method comparison figure**: Figure 3b shows Linf values (147, 154, 120, 130, 150) that match the experiment log but NOT the paper's Table 1 values (122, 120, 125, 130, 121).

This is a serious integrity concern: the paper's Table 1 appears to report optimistic numbers not supported by the actual experimental data.

### 2. hadamard_668.csv

The file `hadamard_668.csv` in the repo root is NOT a valid Hadamard matrix. Verification shows:
- Shape: 668 x 668 (correct)
- All entries +/-1 (correct)
- HH^T diagonal: all 668 (correct)
- **HH^T off-diagonal: values are {-4, 0}, NOT all zeros** (FAILS Hadamard condition)

This is the Legendre baseline near-miss (HH^T = 668I + E where E has off-diagonal entries of -4), which the paper correctly identifies as a near-miss. However, the filename `hadamard_668.csv` is misleading since it suggests a valid Hadamard matrix. The rubric item_024 notes describe it as having "max|E|=4", which is accurate but the file should be named `best_candidate_668.csv` or similar.

### 3. Numba Citation Error

In Section 5.1, the paper cites `\cite{bright2019applying}` (Bright et al.'s ISSAC paper) for Numba. Numba is a separate Python JIT compiler (Lam et al., 2015) and has nothing to do with Bright et al.'s work on SAT+CAS. This is a citation error.

---

## Citation Verification Report

Each entry in `sources.bib` was verified via web search. Results:

### VERIFIED (21 of 32 entries)

| # | Key | Status | Notes |
|---|-----|--------|-------|
| 1 | `eliahou2025modular` | **VERIFIED** | Title, authors, journal (AJOC 93(2), pp 422-427), year (2025), HAL ID all correct. |
| 2 | `djokovic2018goethals` | **VERIFIED** | Title, authors, arXiv:1802.00556, year all correct. |
| 3 | `kharaghani2005hadamard428` | **VERIFIED** | Title, authors, JCD 13(6):435-440, DOI all correct. |
| 4 | `williamson1944hadamard` | **VERIFIED** | Title, author, Duke Math J 11(1):65-81, DOI all correct. |
| 5 | `paley1933orthogonal` | **VERIFIED** | Title, author, J. Math. Phys. 12(1-4):311-320, DOI all correct. |
| 6 | `hadamard1893resolution` | **VERIFIED** | Title, author, Bull. Sci. Math. 17:240-246, year all correct. |
| 7 | `goethals1970seidel` | **VERIFIED** | Title, authors, J. Austral. Math. Soc. 11(3):343-344, year all correct. |
| 8 | `horadam2007hadamard` | **VERIFIED** | Title, author, Princeton UP, 2007 all correct. |
| 9 | `miyamoto1991hadamard` | **VERIFIED** | Title, author, JCTA 57(1):86-108, DOI all correct. |
| 10 | `turyn1972hadamard` | **VERIFIED** | Title, author, JCTA 12(3):319-321, DOI all correct. |
| 11 | `sagemath_hadamard` | **VERIFIED** | GitHub Issue #34807 exists and matches description. |
| 12 | `planetmath_hadamard` | **VERIFIED** | PlanetMath page exists and matches description. |
| 13 | `sylvester1867thoughts` | **VERIFIED** | Title, author, Phil. Mag. 34:461-475, DOI all correct. |
| 14 | `cati2023implementing` | **VERIFIED** | Title, authors (Matteo Cati), arXiv:2306.16812, DOI all correct. |
| 15 | `djokovic2009sds` | **VERIFIED** | Title, author, Oper. Matrices 3(4):557-569, DOI all correct. |
| 16 | `djokovic2013new` | **VERIFIED** | Title, authors, JCD 22(6):270-277, 2014, DOI all correct. Key name has wrong year but bib entry year is correct. |
| 17 | `seberry2020hadamard_monograph` | **VERIFIED** | Title, authors (Seberry & Yamada), Wiley, 2020, DOI all correct. |
| 18 | `bennett2026quaternionic` | **VERIFIED** | Title, authors, arXiv:2601.22337, 2026, DOI all correct. |
| 19 | `suksmono2025qaoa` | **VERIFIED** | Title, author, Sci. Rep. 15, 2025, DOI all correct. |
| 20 | `colbourn2007handbook` | **VERIFIED** | Title, editors, Chapman & Hall/CRC, 2nd ed, 2007, DOI all correct. |
| 21 | `delauney2011flannery` | **VERIFIED** | Title, authors, AMS Surv. Mon. 175, 2011, DOI all correct. |
| 22 | `delauney2009asymptotic` | **VERIFIED** | Title, author, JCTA 116(4):1002-1008, 2009, DOI all correct. |
| 23 | `baumert1965hadamard` | **VERIFIED** | Title, authors, Bull. AMS 71(1):169-170, 1965, DOI all correct. |
| 24 | `goethals1967orthogonal` | **VERIFIED** | Title, authors, Canad. J. Math. 19:1001-1010, 1967, DOI all correct. |
| 25 | `seberry1992hadamard` | **VERIFIED** | Title, authors, Contemporary Design Theory, pp 431-560, 1992 all correct. |

### PARTIALLY CORRECT / ERRORS FOUND (7 entries)

| # | Key | Status | Issues |
|---|-----|--------|--------|
| 1 | `cati2024hadamard` | **PARTIALLY CORRECT** | Title should be "A database of **constructions of** Hadamard matrices" (missing words). First author name is **Matteo** not Mattia. |
| 2 | `bright2021sat` | **PARTIALLY CORRECT** | The arXiv ID 1907.04987 is real, but: (a) the actual title is "The SAT+CAS Method for Combinatorial Search with Applications to Best Matrices" (not as cited); (b) authors should include **Djokovic** (not Heinle); (c) year should be 2019, not 2021. |
| 3 | `seberry2020hadamard` | **PARTIALLY CORRECT** | BibTeX key says "2020" but the year field correctly says 2017 (book published 2017). Key name is misleading but the entry body is correct. Minor issue. |
| 4 | `suksmono2018quantum` | **PARTIALLY CORRECT** | Conflates two different Suksmono papers. The title "Finding Hadamard matrices by a quantum annealing machine" matches a 2019 Sci. Rep. paper (vol 9, by Suksmono & Minato), NOT a 2018 paper. The 2018 Suksmono paper was in Entropy, with a different title. Volume (8), article number (2986), and DOI (10.1038/s41598-018-21394-1) do not match the actual paper. **The cited DOI may not resolve correctly.** |
| 5 | `djokovic2008williamson` | **PARTIALLY CORRECT** | The title and first author (Djokovic) correspond to a real 1993 paper (Discrete Math 113(1-3):261-263, sole author), NOT a 2008 paper. The DOI 10.1016/j.disc.2007.05.009 resolves to an unrelated 2008 paper. Year, co-author, volume, pages, and DOI are all wrong. |
| 6 | `delauney2014density` | **PARTIALLY CORRECT** | Title and authors correct. However: actual publication is volume 2 (2010), pages 233-246, DOI 10.1007/s12095-010-0028-9. The citation claims volume 6(4), year 2014, pages 233-242, DOI 10.1007/s12095-014-0105-7. **Volume, year, pages, and DOI are all wrong.** |
| 7 | `turyn1974hadamard` | **PARTIALLY CORRECT** | All fields correct except DOI has wrong final digit: should be `10.1016/0097-3165(74)90056-9`, not `10.1016/0097-3165(74)90056-0`. Minor typo. |

### FABRICATED / INCORRECT (2 entries)

| # | Key | Status | Issues |
|---|-----|--------|--------|
| 1 | `bright2019applying` | **FABRICATED** | The DOI 10.1145/3326229.3326260 resolves to a completely different ISSAC 2019 paper ("Quadratic-Time Algorithms for Normal Elements" by Giesbrecht, Jamshidpey, & Schost). No paper with the exact title "Applying Computer Algebra and SAT Solving to Hadamard Matrices" by Bright, Kotsireas, Ganesh could be found at ISSAC 2019 or elsewhere. The citation appears to be an amalgamation of their real body of work with a fabricated title and misappropriated DOI. |
| 2 | `spence1977sds` | **FABRICATED** | The title "A construction for Hadamard arrays" exists as a real paper, but by **Jennifer Seberry (Wallis)** (1972), NOT Edward Spence (1977). The DOI 10.1017/S0004972700023261 resolves to an unrelated 1977 paper. Author, year, and DOI are all wrong. |

### UNABLE TO FULLY VERIFY (1 entry)

| # | Key | Status | Issues |
|---|-----|--------|--------|
| 1 | `cooper1972wallis` | **UNVERIFIED** | The metadata (Cooper & Wallis, Bull. Austral. Math. Soc. 7(2):269-277, 1972) is plausible but could not be independently confirmed via web search. The DOI could not be resolved. |

### Unused BibTeX Entries (not cited in paper)

Three entries in `sources.bib` are never cited via `\cite{}` in the paper:
- `djokovic2013new`
- `goethals1970seidel`
- `spence1977sds`

These are dead entries that should be removed.

---

## Specific Revision Requirements

### Critical (must fix for acceptance)

1. **Correct Table 1 to match actual experimental data.** The numbers in Table 1 (evaluation counts, best L2/Linf) must agree with `results/experiments/experiment_log.json`. Either the table must be revised downward to match the actual data, or the experiment log must be updated with the missing experiments (DFT-guided moves, row-sum targeted restarts). The current 10x inflation in total evaluations (5 x 10^8 claimed vs 5.7 x 10^7 actual) is unacceptable.

2. **Fix or remove the 7 incorrect/fabricated citations:**
   - `cati2024hadamard`: Fix title (add "constructions of") and first author name (Matteo, not Mattia).
   - `bright2021sat`: Correct title, fix author list (replace Heinle with Djokovic), change year to 2019.
   - `suksmono2018quantum`: Either cite the correct 2018 Entropy paper or the correct 2019 Sci. Rep. paper (vol 9, article 14380, with Minato as co-author). Do not conflate the two.
   - `djokovic2008williamson`: Correct year to 1993, fix volume/pages to 113(1-3):261-263, remove Kotsireas as co-author, fix DOI.
   - `bright2019applying`: Replace entirely with the correct citation. If no exact ISSAC 2019 paper by Bright/Kotsireas/Ganesh on Hadamard matrices exists, cite their actual related work (e.g., CASCON 2019 or J. Symb. Comput. 2020) or remove.
   - `spence1977sds`: Either correct to the actual Seberry/Wallis 1972 paper or replace with the correct Spence reference.
   - `delauney2014density`: Correct volume to 2, year to 2010, pages to 233-246, DOI to 10.1007/s12095-010-0028-9.
   - `turyn1974hadamard`: Fix DOI last digit (0 -> 9).

3. **Fix the Numba citation in Section 5.1.** The paper cites `\cite{bright2019applying}` for Numba, which is incorrect. Either add a proper Numba citation (Lam et al., 2015, doi:10.1145/2833157.2833162) or remove the citation.

4. **Remove unused bibliography entries** (`djokovic2013new`, `goethals1970seidel`, `spence1977sds`).

5. **Rename or clarify `hadamard_668.csv`** -- the file is a near-miss, not a valid Hadamard matrix. The filename is misleading.

### Recommended (improve for publication quality)

6. **Regenerate figures for publication quality:**
   - Fig 3a (search landscape): The individual PSD bar plots use default matplotlib styling. Consider using a filled area plot or heatmap with proper colormaps.
   - Fig 4 (orbit structure): The right panel is a trivial bar chart where all 83 bars have identical height. This conveys no information; replace with a more informative visualization (e.g., circular arrangement of orbits, connection diagram).
   - All figures should use consistent styling, font sizes, and color palettes.

7. **Reconcile convergence figure with text.** Figure 2 shows plateaus at ~450K-600K, but Section 6.3 describes the plateau as [4 x 10^5, 4.2 x 10^5]. These should be consistent.

8. **Add error bars or confidence intervals** for stochastic methods (SA variants). Multiple restarts should report mean +/- std rather than only best values.

9. **Clarify H(716) status in Table 2.** The paper claims H(716) = H(4 x 179) is known, but 179 has phi(179) = 178 = 2 x 89, which has similarly sparse subgroup structure to 167. It would strengthen the paper to explain why 179 is easier than 167.

---

## Summary

The paper presents a well-structured and clearly written computational investigation of a significant open problem in combinatorial design theory. The mathematical framework is sound, the literature review is comprehensive, and the negative result is honestly reported. However, there are serious concerns:

1. **Results integrity**: Table 1 claims substantially more computation and better results than the actual experiment logs support.
2. **Citation accuracy**: 2 fabricated citations, 5 citations with significant metadata errors, and 1 unverifiable citation out of 32 total entries.
3. **Figure quality**: Adequate but not publication-ready for a top venue.

These issues preclude acceptance in their current form. With the corrections outlined above, particularly fixing the data discrepancies and citation errors, the paper could be suitable for publication as a well-documented negative result contributing to the understanding of the H(668) problem.
