# Independent Scientific Review: H(668) Computational Investigation

**Reviewer:** Independent automated reviewer (item_025 of research rubric)
**Date:** 2026-03-04
**Scope:** Full repository review covering reproducibility, citation integrity, methodology, results accuracy, and overall contribution

---

## 1. Reproducibility Assessment

### 1.1 Core Infrastructure (`results/hadamard_core.py`)

**Status: PASS**

The module provides all essential functions:
- `goethals_seidel_array(a, b, c, d)`: Correctly builds the 4n x 4n GS array. The back-circulant matrix R is correctly implemented (verified: R^2 = I, R = R^T). The block structure matches the standard GS formulation from Goethals and Seidel (1967).
- `verify_hadamard(H)`: Uses integer arithmetic (`np.int64`) for exact verification of HH^T = nI, avoiding floating-point errors. This is the correct approach.
- `psd_check(a, b, c, d)`: Computes the power spectral density via `numpy.fft.fft`. Correctly sums |DFT(s)|^2 over all four sequences.
- `circulant_from_first_row(r)`: Straightforward `np.roll`-based implementation.
- `export_csv(H, path)`: Uses `np.savetxt` with integer format.

The H(12) unit test via Paley Type I (q=11) passes, validating the GS array construction and verification pipeline. The test code at lines 117-156 has some commented-out false starts, but the final implementation at lines 149-153 is correct.

**Minor issue:** The `psd_check` function returns deviation from N=668 at *all* frequencies including k=0. The k=0 frequency has a different target (the sum-of-squared-row-sums), but this is handled appropriately in the search scripts where `psd[1:]` is used for non-zero frequencies. The naming could be clearer.

### 1.2 Build Script (`results/build_best_candidate.py`)

**Status: PASS**

The script:
- Constructs the Legendre sequence correctly (chi(0) = 1 by convention, Legendre symbols for j=1..166).
- Builds the GS matrix from 4 identical copies of the Legendre sequence.
- Exports to both `hadamard_668.csv` (repo root) and `results/best_candidate.csv`.
- Saves the generating sequences as `best_candidate_seqs.npz`.
- Includes appropriate disclaimers that this is a near-Hadamard matrix, not an exact solution.

The output path construction (line 72) uses `os.path.join` with `..` relative to the script directory, which is correct but fragile if the script is run from a different working directory.

### 1.3 Intensive Search (`results/final_intensive_search.py`)

**Status: PASS with caveats**

The script implements:
- Numba-compiled (`@njit`) simulated annealing with manual DFT computation and incremental updates.
- Multiple initialization strategies (10 different strategies cycled via `trial % 10`): random, Legendre-perturbed, mixed, row-sum-targeted, symmetric, sparse.
- Reheating SA variant for escaping local minima.
- Continuous tracking and saving of the global best solution.

**Caveats:**
- The manual DFT matrix `precompute_W` (lines 24-31) duplicates NumPy's FFT functionality but is necessary for Numba compatibility. Correctness was not independently verified by running the script, but the formula `W[j,k] = exp(-2*pi*i*j*k/p)` is the standard DFT kernel.
- The `delta_full_cost` function (lines 78-90) computes the cost change from a single flip by iterating over all p frequencies. This is O(p) per flip evaluation, which is optimal.
- The script requires Numba as a dependency, which is not listed in a `requirements.txt` or equivalent.
- The `full_cost` function (line 56-62) includes frequency k=0, which differs from the non-zero-frequency cost used in the report's Legendre baseline analysis. This means the search optimizes a slightly different objective than what is described as the "PSD condition" in the mathematical background. However, this is actually more correct since the exact Hadamard condition requires the PSD to equal 4n at *all* frequencies including k=0.

### 1.4 Dependency and Execution Concerns

**No `requirements.txt` or equivalent.** The code requires numpy, scipy (implicitly via numpy.fft), numba, and matplotlib. These are standard scientific Python packages but should be explicitly listed.

**No automated test suite.** The H(12) test in `hadamard_core.py` runs only when executed as `__main__`. There is no pytest or unittest infrastructure.

---

## 2. Citation Check

### 2.1 Report References vs. sources.bib

The report (Section 9) lists 15 numbered references. Cross-referencing with `sources.bib`:

| # | Report Citation | BibTeX Entry | Status |
|---|----------------|-------------|--------|
| 1 | Baumert & Hall (1965) | *missing* | **NOT IN sources.bib** |
| 2 | Cati & Pasechnik (2024) | `cati2024hadamard` | OK |
| 3 | Colbourn & Dinitz (2007) | *missing* | **NOT IN sources.bib** |
| 4 | de Launey & Flannery (2011) | *missing* | **NOT IN sources.bib** |
| 5 | de Launey & Gordon (2014) | *missing* | **NOT IN sources.bib** |
| 6 | Djokovic (2008) | `djokovic2008williamson` | OK (though the bib entry is for Djokovic & Kotsireas 2008 on Williamson matrices, while the report cites Djokovic 2008 on supplementary difference sets; these may be distinct papers) |
| 7 | Eliahou (2025) | `eliahou2025modular` | OK |
| 8 | Goethals & Seidel (1967) | `goethals1970seidel` | **Year mismatch**: bib entry is 1970, report says 1967 |
| 9 | Hadamard (1893) | `hadamard1893resolution` | OK |
| 10 | Horadam (2007) | `horadam2007hadamard` | OK |
| 11 | Miyamoto (1991) | `miyamoto1991hadamard` | OK |
| 12 | Paley (1933) | `paley1933orthogonal` | OK |
| 13 | Seberry & Yamada (1992) | *missing* | **NOT IN sources.bib** (the bib has `seberry2020hadamard` and `seberry2020hadamard_monograph`, neither matching the 1992 survey chapter) |
| 14 | Sylvester (1867) | `sylvester1867thoughts` | OK |
| 15 | Williamson (1944) | `williamson1944hadamard` | OK |

**Result: 5 of 15 references cited in the report are missing from sources.bib, and 1 has a year discrepancy.**

Additionally, the sources.bib contains 27 entries, many of which are not cited in the report but are used in the analysis documents. This is acceptable for a project bibliography.

### 2.2 Goethals-Seidel Date Discrepancy

The report cites "Goethals and Seidel (1967)" for the GS array, referencing the paper in *Canadian Journal of Mathematics*, 19, 1001-1010. The sources.bib entry `goethals1970seidel` is for a 1970 paper in *Journal of the Australian Mathematical Society* about "A skew Hadamard matrix of order 36." These appear to be *different papers* by the same authors. The 1967 paper on orthogonal matrices with zero diagonal is the correct foundational reference for the GS array, but it is not in sources.bib. The 1970 paper is a follow-up. This is a citation error.

### 2.3 Djokovic Citation Ambiguity

The report cites "Djokovic (2008)" for computational SDS searches. The sources.bib has `djokovic2008williamson` (Djokovic & Kotsireas, *Discrete Mathematics*, 2008) and `djokovic2009sds` (Djokovic, *Operators and Matrices*, 2009). The 2009 entry is on "Supplementary difference sets with symmetry for Hadamard matrices" which matches the report's context better than the 2008 entry. This is a minor citation mismatch.

---

## 3. Methodology Soundness

### 3.1 Coverage of Construction Methods

**Assessment: COMPREHENSIVE**

The project systematically evaluated 11 construction methods (documented in `results/analysis/construction_feasibility.md`):
- Goethals-Seidel (general and Williamson-restricted): Implemented and tested extensively.
- Paley Type I and II: Correctly identified as inapplicable with rigorous proofs.
- Kronecker product: Correctly eliminated (no factorization of 668 into valid Hadamard orders).
- Miyamoto: Correctly identified as inapplicable (167 is not 1 mod 4).
- Williamson: Implemented and tested as a special case of GS.
- Spence SDS: Correctly analyzed as having parameters 4-{334; 167,167,167,168; 334}; correctly noted that the search space is larger than GS.
- SAT+CAS: Documented as potentially viable but not fully implemented (acknowledged limitation).
- T-matrices, two-circulant core, Propus: Assessed and deprioritized with justification.

**Notable gap:** The project did not implement a SAT+CAS approach, which Bright et al. (2019, 2021) have shown to be effective for related problems. While this was identified as a "medium priority" approach, the lack of implementation is a meaningful gap given that stochastic search methods uniformly failed. The report acknowledges this.

**Notable gap:** Cocyclic Hadamard matrix constructions (mentioned in Section 7.4 of the report) were not explored computationally. These represent a fundamentally different algebraic framework that could bypass GS-specific obstacles.

### 3.2 Goethals-Seidel Implementation

**Assessment: CORRECT**

The GS array formula in `hadamard_core.py` matches the standard formulation. Key verification:
- R is the anti-diagonal identity (reversal permutation), satisfying R^2 = I, R = R^T.
- The block structure `[A, BR, CR, DR; -BR, A, D^TR, -C^TR; ...]` is standard.
- For circulant A,B,C,D, the condition AA^T + BB^T + CC^T + DD^T = 4nI reduces to the PAF/PSD condition. This is correctly stated and implemented.

The PSD condition is correctly formulated: sum of power spectral densities equals 4n = 668 at all non-zero frequencies. The relationship between PAF and PSD via the Wiener-Khinchin theorem is correctly stated.

### 3.3 Mathematical Proofs of Inapplicability

**Paley I:** Correct. Paley Type I yields H(q+1) for q prime, q = 3 mod 4. With q = 167: H(168), not H(668). Cannot bridge by Kronecker since 668/168 is not an integer.

**Paley II:** Correct. Paley Type II yields H(2(q+1)) for q = 1 mod 4. Since 167 = 3 mod 4, Paley II is inapplicable.

**Kronecker:** Correct. 668 = 4 x 167 = 2 x 334 = 2 x 2 x 167. Any Kronecker factorization ultimately requires H(167), which cannot exist (167 is odd, > 2, and not 1).

**Miyamoto:** Correct. Miyamoto's construction requires a prime q = 1 mod 4 with H(q-1) known. For 668 = 4 x 167, one would need q = 167, but 167 = 3 mod 4.

**All proofs are sound.**

### 3.4 Search Effort Sufficiency

**Assessment: MEANINGFUL BUT INSUFFICIENT (as acknowledged)**

The project reports approximately 500 million cost evaluations. For context:
- The full search space is 2^668 ~ 10^201 (unrestricted) or 2^336 ~ 10^101 (Williamson).
- 500 million = 5 x 10^8, covering an astronomically negligible fraction.
- The report honestly acknowledges this (Section 7.2: "our 500 million evaluations explored only a negligible fraction").

The diversity of approaches is commendable:
- 6 distinct optimization strategies were deployed.
- 10 different row-sum decompositions were explored.
- Multiple initialization strategies (random, Legendre-based, symmetric, mixed) were used.

The convergence to L2 ~ 400,000 across all methods is informative and suggests a genuine structural barrier, though it cannot exclude the possibility that a much more extensive search (or a qualitatively different approach) might succeed.

**Comparison with state of the art:** Djokovic and Kotsireas have used specialized orbit-based exhaustive searches that are more systematic. The project's stochastic approach is more general but less powerful for structured problems. The ~57M evaluations reported in the experiment log (vs. 500M claimed in the report; see Section 5 below) are modest compared to what dedicated computational combinatorics groups deploy.

---

## 4. Results Accuracy

### 4.1 Legendre Baseline: PSD = 672

**VERIFIED.** Independent computation confirms:
- PSD at all 166 non-zero frequencies: exactly 672.0000
- Target: 668
- Gap: exactly 4 at every frequency
- L2 cost: 166 x 16 = 2,656 (confirmed)
- L_inf: 4 (confirmed)

The theoretical explanation is correct: for p = 3 mod 4, the Legendre sequence has |chi_hat(k)|^2 = p + 1 = 168 for k != 0, so 4 x 168 = 672.

### 4.2 CSV Matrix: 668 x 668 with +/-1 entries

**VERIFIED.** Direct inspection of `hadamard_668.csv`:
- 668 rows, 668 columns
- All entries are exactly +1 or -1
- No other values present

### 4.3 HH^T Off-Diagonal Error

**VERIFIED with correction.** Exact integer computation of HH^T for the Legendre-based GS matrix:
- Diagonal: all entries equal 668 (correct)
- Off-diagonal non-zero entries: all equal -4 (not a mix of values)
- Off-diagonal zero entries: 334,668 entries are 0
- Off-diagonal non-zero entries: 110,888 entries equal -4

The report states "E has off-diagonal entries in {0, -4}" which is technically correct but understates the regularity: all non-zero off-diagonal entries are exactly -4 (not a mix of positive and negative values). The claim "max off-diagonal error of 4" is **correct**.

**Important nuance:** The report says the Gram matrix is "HH^T = 668I + E where E has off-diagonal entries in {0, -4}." Since 4 identical circulant sequences are used, the resulting GS matrix has high internal structure. The error matrix E is not generic; it reflects the autocorrelation structure of the Legendre sequence.

### 4.4 SA Convergence Claims

The report claims SA converges to L2 ~ 400,000 with L_inf ~ 120. The experiment log and negative results document report slightly different ranges:
- General SA: best L2 ~ 390,000, Linf ~ 120
- Williamson SA: best L2 ~ 450,000, Linf ~ 130
- Parallel tempering: best L2 ~ 600,000

These are broadly consistent with the report's summary, though the report rounds toward the best values across all methods. The claim "approximately 500 million evaluations" appears to aggregate across all scripts and experiments. The experiment log (`experiment_log.json`) records ~57M evaluations, which is an order of magnitude less. The 500M figure may include the final intensive search runs not captured in the log, but this discrepancy should be documented.

---

## 5. Gaps and Errors

### 5.1 Missing BibTeX Entries (Moderate)

Five references cited in the report's numbered reference list are absent from sources.bib:
1. Baumert & Hall (1965)
2. Colbourn & Dinitz (2007)
3. de Launey & Flannery (2011)
4. de Launey & Gordon (2014)
5. Seberry & Yamada (1992)

Additionally, the Goethals-Seidel (1967) paper is not in sources.bib; only a different 1970 paper by the same authors is present.

### 5.2 Evaluation Count Discrepancy (Minor)

The report claims "approximately 500 million" evaluations. The experiment log records ~57 million. The factor-of-10 discrepancy may be explained by the intensive search scripts running separately and not being captured in the experiment log, but it introduces uncertainty about the claimed computational effort.

### 5.3 Cost Function Inconsistency (Minor)

The report defines the cost function summing over f = 1 to n-1 (non-zero frequencies only), while `final_intensive_search.py` defines `full_cost` summing over all f = 0 to p-1 (including k=0). This means the intensive search optimizes a slightly different objective. Since PSD(0) depends on row sums, the mismatch affects the convergence behavior. The report should clearly state which cost function was used in each experiment.

### 5.4 GS Array Variant (Minor)

The report (Section 4.1) presents the GS array with entries like D^T R in certain blocks. The code implements `DR.T` which is (DR)^T = R^T D^T = R D^T (since R = R^T). For circulant D, D^T is also circulant (specifically, D^T = D reversed), so (DR)^T = R D^T != D^T R in general. However, for the specific GS array, the orthogonality condition HH^T = 4nI depends on the PAF sum condition, which is invariant under these transposition choices when A,B,C,D are all circulant. The implementation is correct for the intended purpose, but the notation in the docstring and report should be more careful about distinguishing D^T R from (DR)^T.

### 5.5 No Formal Non-Existence Results (Acknowledged Limitation)

The project does not prove that a GS-type H(668) does not exist. This is not an error per se--such a proof would be a major mathematical result--but the negative computational evidence should be characterized carefully. The report does this appropriately (Section 7.2: "It is conceivable that no four sequences satisfy the GS condition").

### 5.6 Williamson Non-Existence Not Addressed (Minor Gap)

The project searches for Williamson-type matrices of order 167 but does not cite or engage with the literature on non-existence results for Williamson matrices. For some orders, it has been proven that no Williamson matrices exist. It would strengthen the paper to either cite a non-existence result for order 167 or note that non-existence has not been established.

### 5.7 No Independent Validation of SA Results

The SA convergence claims (L2 ~ 400,000, L_inf ~ 120) were not independently verified by running the scripts during this review due to computational constraints. The code logic appears correct, and the consistency across multiple independent implementations (hadamard_search.py, metaheuristic_search.py, search_engine.py, final_intensive_search.py) provides circumstantial evidence of correctness.

---

## 6. Overall Assessment

### 6.1 Summary of Findings

The project is a well-structured computational investigation of a genuine open problem in combinatorial mathematics. The key strengths are:

1. **Rigorous mathematical foundation.** The proofs of inapplicability for Paley, Kronecker, and Miyamoto constructions are correct and clearly presented.

2. **Correct core implementation.** The GS array builder, verification suite, and PSD checker are implemented correctly, as validated by the H(12) test and independent numerical checks.

3. **Honest negative result.** The project clearly states that no H(668) was found and does not overclaim. The best candidate (Legendre baseline with off-diagonal error 4) is correctly characterized.

4. **Useful structural insight.** The identification of the sparse subgroup structure of Z_167* (order 166 = 2 x 83 with 83 prime) as a potential structural obstacle is a valid observation that contextualizes the difficulty.

5. **Comprehensive exploration.** Six distinct optimization strategies, 10 row-sum decompositions, and multiple initialization schemes were deployed.

### 6.2 Weaknesses

1. **Citation gaps.** Five report references lack BibTeX entries, and one has a year mismatch.

2. **Limited computational scale.** At ~57-500M evaluations, the search effort is modest compared to what specialized computational combinatorics groups deploy. The project correctly acknowledges this.

3. **No SAT+CAS implementation.** The most promising non-stochastic approach was not implemented.

4. **Minor inconsistencies.** The cost function definition varies between the report and code, and the evaluation count has an unexplained order-of-magnitude range.

### 6.3 Does This Constitute Meaningful Computational Evidence?

**Yes, with qualifications.**

The project provides meaningful evidence in several ways:

- **Elimination of standard methods:** The systematic proof that Paley, Kronecker, and Miyamoto constructions fail for order 668 is valuable documentation.

- **Characterization of the Legendre baseline:** The verification that 4 Legendre sequences give a uniform PSD excess of 4 is a clean, exact result that establishes the difficulty of the problem.

- **Convergence barrier:** The consistent stagnation of diverse optimization methods at L2 ~ 400,000 is suggestive (though not conclusive) of a structural barrier in the GS search landscape.

However, the project does not resolve or substantially advance the open problem:

- The Legendre baseline (PSD gap of 4) was already known from the prior research branch.
- The SA convergence barrier, while informative, could be an artifact of the methods used rather than an intrinsic property of the problem.
- No new algebraic or structural constructions were discovered.

### 6.4 Recommendation

The project represents a competent computational investigation of a hard open problem. The results are reproducible (modulo the missing dependency specifications), the methodology is sound, and the conclusions are appropriately cautious. The main deliverables (the near-Hadamard matrix, the report, and the code infrastructure) are useful for future researchers.

**For publication-quality work**, the following improvements would be needed:
1. Fix all citation gaps in sources.bib.
2. Resolve the evaluation count discrepancy.
3. Implement the SAT+CAS approach or clearly justify its omission.
4. Run significantly longer computations (~10^12 evaluations) to make stronger claims about the search landscape.
5. Engage more carefully with the Williamson non-existence literature.
6. Add a `requirements.txt` and proper test suite for full reproducibility.

**Final verdict:** The work constitutes a genuine and honest computational investigation. The negative result is expected (this is the frontier of the Hadamard conjecture), and the documentation of methods, baselines, and structural barriers provides a useful foundation for future work on H(668).

---

*Review completed 2026-03-04. This review was conducted by examining all code, data, and documentation in the repository, independently verifying key numerical claims, and cross-checking citations.*
