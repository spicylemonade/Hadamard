#!/usr/bin/env python3
"""
Core infrastructure for Hadamard matrix construction and verification.
Optimized for the H(668) problem using Goethals-Seidel array with 4 circulant blocks of order 167.
"""

import numpy as np
from numpy.fft import fft, ifft

P = 167  # 167 is prime, 167 ≡ 3 (mod 4)
N = 4 * P  # = 668

def legendre_symbol(a, p=P):
    """Compute the Legendre symbol (a/p)."""
    if a % p == 0:
        return 0
    val = pow(a, (p - 1) // 2, p)
    return 1 if val == 1 else -1

def psd_check(a, b, c, d):
    """
    Compute power spectral density: |DFT(a)|^2 + |DFT(b)|^2 + |DFT(c)|^2 + |DFT(d)|^2
    Returns the PSD array and the maximum deviation from N=668.
    """
    af = fft(a.astype(np.float64))
    bf = fft(b.astype(np.float64))
    cf = fft(c.astype(np.float64))
    df = fft(d.astype(np.float64))
    psd = np.abs(af)**2 + np.abs(bf)**2 + np.abs(cf)**2 + np.abs(df)**2
    deviation = psd - N
    return psd, deviation

def psd_cost(a, b, c, d):
    """Sum of squared PSD deviations from N=668."""
    _, dev = psd_check(a, b, c, d)
    return np.sum(dev**2)

def psd_max_dev(a, b, c, d):
    """Maximum absolute PSD deviation."""
    _, dev = psd_check(a, b, c, d)
    return np.max(np.abs(dev))

def circulant_from_first_row(r):
    """Build circulant matrix from first row."""
    n = len(r)
    C = np.zeros((n, n), dtype=r.dtype)
    for i in range(n):
        C[i] = np.roll(r, -i)
    return C

def back_circulant_matrix(n):
    """Build the n×n back-circulant (reversal) permutation matrix R."""
    R = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        R[i, (-i) % n] = 1
    return R

def goethals_seidel_array(a, b, c, d):
    """
    Build the 4n×4n Hadamard matrix using Goethals-Seidel array:
    H = [[ A,  BR,  CR,  DR],
         [-BR,  A,  D^TR, -C^TR],
         [-CR, -D^TR, A,  B^TR],
         [-DR,  C^TR, -B^TR, A]]
    """
    n = len(a)
    A = circulant_from_first_row(np.array(a, dtype=np.int8))
    B = circulant_from_first_row(np.array(b, dtype=np.int8))
    C = circulant_from_first_row(np.array(c, dtype=np.int8))
    D = circulant_from_first_row(np.array(d, dtype=np.int8))
    R = back_circulant_matrix(n)
    
    BR = B @ R
    CR = C @ R
    DR = D @ R
    
    H = np.block([
        [ A,    BR,    CR,    DR   ],
        [-BR,   A,     DR.T, -CR.T ],
        [-CR,  -DR.T,  A,     BR.T ],
        [-DR,   CR.T, -BR.T,  A    ]
    ])
    return H

def verify_hadamard(H):
    """
    Verify H is a Hadamard matrix using exact integer arithmetic.
    Returns (is_valid, message).
    """
    n = H.shape[0]
    if H.shape != (n, n):
        return False, f"Not square: {H.shape}"
    
    # Check ±1 entries
    unique = np.unique(H)
    if not np.array_equal(np.sort(unique), np.array([-1, 1])):
        return False, f"Not ±1 entries: {unique}"
    
    # Check HH^T = nI using integer arithmetic
    H_int = H.astype(np.int64)
    HHt = H_int @ H_int.T
    expected = n * np.eye(n, dtype=np.int64)
    
    if np.array_equal(HHt, expected):
        return True, f"Valid Hadamard matrix of order {n}"
    else:
        max_err = int(np.max(np.abs(HHt - expected)))
        return False, f"H*H^T ≠ {n}*I, max error = {max_err}"

def export_csv(H, path):
    """Export matrix to CSV."""
    np.savetxt(path, H, delimiter=",", fmt="%d")
    return path

# ---- Tests on known matrices ----

def test_hadamard_12():
    """Test using known H(12) from Paley I with q=11."""
    q = 11
    # Paley Type I: q ≡ 3 (mod 4), build (q+1)×(q+1) matrix
    # Conference matrix C of order q+1
    n = q + 1
    chi = [0] + [legendre_symbol(j, q) for j in range(1, q)]
    
    # Build QR conference matrix
    Q = np.zeros((q, q), dtype=np.int8)
    for i in range(q):
        for j in range(q):
            Q[i, j] = legendre_symbol((j - i) % q, q)
    
    # Paley Type I: H = I + S where S = [[0, j^T], [-j, Q]]
    S = np.zeros((n, n), dtype=np.int8)
    S[0, 1:] = 1  # first row: 0, 1, 1, ..., 1
    S[1:, 0] = -1  # first col: 0, -1, -1, ..., -1
    S[1:, 1:] = Q
    
    H = np.eye(n, dtype=np.int8) + S
    # Normalize: replace 0s (shouldn't be any in I+S since diagonal of S is 0)
    # Actually H = I + S, and S has 0 on diagonal, so H has 1 on diagonal + 0 = 1
    # And off-diagonal: S[i,j] + delta[i,j], for i!=j: just S[i,j]
    # Wait, S has non-zero entries off diagonal. Let me check.
    # S[0,0] = 0, so H[0,0] = 1. S[i,j] = chi(j-i) for i,j >= 1.
    # So H[i,j] = delta[i,j] + S[i,j]. For i=j: 1+0=1. For i!=j: S[i,j] in {-1,1}.
    # But S[0, j>0] = 1, S[i>0, 0] = -1. So H has entries in {-1, 0, 1, 2}.
    # Hmm, that's wrong. Need to adjust.
    
    # Standard Paley Type I: H_n = [[1, e^T], [-e, Q+I]]
    # where Q is the Jacobsthal matrix
    H = np.zeros((n, n), dtype=np.int8)
    H[0, :] = 1
    H[1:, 0] = -1
    H[1:, 1:] = Q + np.eye(q, dtype=np.int8)
    
    # All entries should be ±1
    valid, msg = verify_hadamard(H)
    return valid, msg

if __name__ == "__main__":
    print("Testing Hadamard core infrastructure...")
    valid, msg = test_hadamard_12()
    print(f"H(12) test: {msg}")
    assert valid, f"H(12) test failed: {msg}"
    print("All tests passed.")
