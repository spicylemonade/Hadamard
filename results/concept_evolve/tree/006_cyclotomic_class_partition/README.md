# Cyclotomic Class Partition

## Context
The multiplicative group Z*_167 has order 166 = 2 × 83. Since 4 does not divide 166, we cannot partition into 4 equal cyclotomic classes. This is the root algebraic obstruction for constructing H(668) via standard methods.

## Group Structure
- Primitive root: g = 5
- Z*_167 ≅ Z_166 = Z_2 × Z_83
- Only subgroups: orders 1, 2, 83, 166
- Quadratic residues (order 83 subgroup): <g²> = <25>
- QR and QNR partition Z*_167 into two equal halves

## Non-uniform Partition
Partition by discrete log mod 4:
- Class 0: 42 elements (indices 0,4,8,...,164)
- Class 1: 42 elements (indices 1,5,9,...,165)
- Class 2: 41 elements (indices 2,6,10,...,162)
- Class 3: 41 elements (indices 3,7,11,...,163)

## Implementation Backlog
- [x] Find primitive root mod 167
- [x] Compute discrete logarithm table
- [x] Partition into cyclotomic-like classes
- [x] Test as initialization for GS search (energy ~598K)
- [ ] Explore hybrid partitions using QR/QNR + index structure
- [ ] Try partitions based on Z_2 × Z_83 CRT structure

## References
- Storer (1967), "Cyclotomy and Difference Sets"
