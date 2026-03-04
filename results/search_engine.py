#!/usr/bin/env python3
"""
High-performance search engine for Hadamard matrix H(668) via Goethals-Seidel array.

Strategy: Find four ±1 sequences a, b, c, d of length 167 such that
|DFT(a)|^2 + |DFT(b)|^2 + |DFT(c)|^2 + |DFT(d)|^2 = 668 for all frequencies.

Uses parallel tempering with DFT-guided moves.
"""

import numpy as np
from numpy.fft import fft
import time
import sys
import json
import os

P = 167
N = 4 * P  # 668

def legendre_symbol(a, p=P):
    if a % p == 0:
        return 0
    val = pow(a, (p - 1) // 2, p)
    return 1 if val == 1 else -1

def compute_psd(seqs):
    """Compute PSD for four sequences. Returns array of length P."""
    psd = np.zeros(P)
    for s in seqs:
        sf = fft(s.astype(np.float64))
        psd += np.abs(sf)**2
    return psd

def compute_cost_l2(psd):
    """L2 cost: sum of squared deviations."""
    return np.sum((psd - N)**2)

def compute_cost_linf(psd):
    """L-infinity cost: max absolute deviation."""
    return np.max(np.abs(psd - N))

def compute_cost_l1(psd):
    """L1 cost: sum of absolute deviations."""
    return np.sum(np.abs(psd - N))

def incremental_psd_update(old_psd, old_seq_fft, seq_idx, pos, seqs):
    """
    Efficiently recompute PSD after flipping seqs[seq_idx][pos].
    
    When we flip entry j of sequence s:
    New s'[j] = -s[j], so delta[j] = -2*s[j]
    New FFT: s'_hat[k] = s_hat[k] + delta[j] * omega^{jk}
    where omega = exp(-2*pi*i/P)
    
    New |s'_hat[k]|^2 = |s_hat[k] + delta_j * omega^{jk}|^2
    = |s_hat[k]|^2 + 2*Re(s_hat[k]* * delta_j * omega^{jk}) + |delta_j|^2
    = |s_hat[k]|^2 + 2*delta_j * Re(conj(s_hat[k]) * omega^{jk}) + 4
    
    PSD change at each k:
    delta_psd[k] = 2*delta_j * Re(conj(s_hat[k]) * omega^{jk}) + 4
    """
    j = pos
    s = seqs[seq_idx]
    delta_j = -2.0 * s[j]  # The change in the sequence value
    
    # omega^{jk} for k = 0, ..., P-1
    k_arr = np.arange(P)
    omega_jk = np.exp(-2j * np.pi * j * k_arr / P)
    
    old_fft = old_seq_fft[seq_idx]
    
    # New FFT
    new_fft = old_fft + delta_j * omega_jk
    
    # PSD change
    new_psd = old_psd - np.abs(old_fft)**2 + np.abs(new_fft)**2
    
    return new_psd, new_fft

class ParallelTemperingSearch:
    """Parallel tempering (replica exchange) search for GS quadruples."""
    
    def __init__(self, n_replicas=16, seed=42):
        self.n_replicas = n_replicas
        self.rng = np.random.RandomState(seed)
        self.P = P
        self.N = N
        
        # Temperature ladder (geometric)
        t_min, t_max = 0.1, 500.0
        self.temps = np.geomspace(t_min, t_max, n_replicas)
        
        # Initialize replicas
        self.replicas = []
        for i in range(n_replicas):
            seqs = self._init_sequences()
            self.replicas.append(seqs)
        
        # Precompute FFTs and PSDs
        self.ffts = []
        self.psds = []
        self.costs = []
        for seqs in self.replicas:
            seq_ffts = [fft(s.astype(np.float64)) for s in seqs]
            psd = sum(np.abs(sf)**2 for sf in seq_ffts)
            cost = np.sum((psd - N)**2)
            self.ffts.append(seq_ffts)
            self.psds.append(psd)
            self.costs.append(cost)
        
        self.best_cost = min(self.costs)
        self.best_idx = self.costs.index(self.best_cost)
        self.best_seqs = [s.copy() for s in self.replicas[self.best_idx]]
        self.best_psd = self.psds[self.best_idx].copy()
    
    def _init_sequences(self):
        """Initialize with Legendre-based sequences (best known starting point)."""
        chi = np.array([legendre_symbol(j) for j in range(P)], dtype=np.float64)
        chi[0] = 1  # Replace 0 with 1
        
        # Start with 4 copies of the Legendre sequence with random perturbation
        seqs = []
        for i in range(4):
            s = chi.copy()
            # Randomly flip ~5% of entries
            mask = self.rng.random(P) < 0.05
            s[mask] = -s[mask]
            seqs.append(s)
        return seqs
    
    def _guided_flip(self, replica_idx):
        """
        DFT-guided flip: identify the frequency with largest PSD deviation,
        then flip an entry that reduces that deviation.
        """
        psd = self.psds[replica_idx]
        dev = psd - N
        
        # With probability 0.3, do a random flip; otherwise guided
        if self.rng.random() < 0.3:
            seq_idx = self.rng.randint(4)
            pos = self.rng.randint(P)
            return seq_idx, pos
        
        # Find frequency with largest deviation
        worst_k = np.argmax(np.abs(dev))
        
        # For this frequency, find which sequence contributes most
        contrib = np.array([np.abs(self.ffts[replica_idx][i][worst_k])**2 for i in range(4)])
        
        if dev[worst_k] > 0:
            # PSD too high at this frequency - flip in sequence with largest contribution
            seq_idx = int(np.argmax(contrib))
        else:
            # PSD too low - flip in sequence with smallest contribution
            seq_idx = int(np.argmin(contrib))
        
        # Find the position that would help most at this frequency
        # The effect of flipping position j on FFT[k] depends on omega^{jk}
        # Choose position that would reduce |FFT[k]|^2 most for the target sequence
        s = self.replicas[replica_idx][seq_idx]
        sf = self.ffts[replica_idx][seq_idx][worst_k]
        
        # Effect of flipping j: new_fft = sf + (-2*s[j]) * omega^{jk}
        k_arr = np.arange(P)
        omega_jk = np.exp(-2j * np.pi * self.rng.randint(P) * worst_k / P)  # Just pick random for speed
        
        pos = self.rng.randint(P)
        return seq_idx, pos
    
    def step(self, replica_idx):
        """Single SA step for one replica."""
        seq_idx, pos = self._guided_flip(replica_idx)
        
        seqs = self.replicas[replica_idx]
        old_val = seqs[seq_idx][pos]
        
        # Compute new PSD incrementally
        new_psd, new_fft = incremental_psd_update(
            self.psds[replica_idx], 
            self.ffts[replica_idx],
            seq_idx, pos, seqs
        )
        new_cost = np.sum((new_psd - N)**2)
        
        delta = new_cost - self.costs[replica_idx]
        T = self.temps[replica_idx]
        
        if delta < 0 or self.rng.random() < np.exp(-delta / max(T, 1e-10)):
            # Accept
            seqs[seq_idx][pos] = -old_val
            self.ffts[replica_idx][seq_idx] = new_fft
            self.psds[replica_idx] = new_psd
            self.costs[replica_idx] = new_cost
            
            if new_cost < self.best_cost:
                self.best_cost = new_cost
                self.best_seqs = [s.copy() for s in seqs]
                self.best_psd = new_psd.copy()
                self.best_idx = replica_idx
                return True  # New best found
        
        return False
    
    def exchange(self):
        """Attempt replica exchange between adjacent temperatures."""
        for i in range(self.n_replicas - 1):
            j = i + 1
            delta = (1.0/self.temps[i] - 1.0/self.temps[j]) * (self.costs[j] - self.costs[i])
            if delta < 0 or self.rng.random() < np.exp(-delta):
                # Swap replicas
                self.replicas[i], self.replicas[j] = self.replicas[j], self.replicas[i]
                self.ffts[i], self.ffts[j] = self.ffts[j], self.ffts[i]
                self.psds[i], self.psds[j] = self.psds[j], self.psds[i]
                self.costs[i], self.costs[j] = self.costs[j], self.costs[i]
    
    def run(self, n_iterations=10_000_000, log_interval=100_000, exchange_interval=100):
        """Run parallel tempering search."""
        t0 = time.time()
        
        print(f"Parallel Tempering Search for H({N})")
        print(f"  Replicas: {self.n_replicas}")
        print(f"  Temperatures: [{self.temps[0]:.2f}, ..., {self.temps[-1]:.2f}]")
        print(f"  Iterations: {n_iterations:,}")
        print(f"  Initial best cost: {self.best_cost:.1f}")
        print()
        
        log_data = []
        
        for it in range(1, n_iterations + 1):
            # Step each replica
            for r in range(self.n_replicas):
                self.step(r)
            
            # Exchange replicas periodically
            if it % exchange_interval == 0:
                self.exchange()
            
            # Log progress
            if it % log_interval == 0:
                elapsed = time.time() - t0
                max_dev = np.max(np.abs(self.best_psd - N))
                l1_dev = np.sum(np.abs(self.best_psd - N))
                
                entry = {
                    "iteration": it,
                    "elapsed_s": round(elapsed, 1),
                    "best_l2_cost": round(float(self.best_cost), 1),
                    "best_linf_dev": round(float(max_dev), 2),
                    "best_l1_dev": round(float(l1_dev), 1),
                    "replica_costs": [round(float(c), 1) for c in self.costs]
                }
                log_data.append(entry)
                
                print(f"  iter {it:>10,} | elapsed {elapsed:>7.1f}s | "
                      f"L2={self.best_cost:>10.1f} | Linf={max_dev:>6.2f} | "
                      f"L1={l1_dev:>8.1f} | "
                      f"coldest={self.costs[0]:>10.1f}")
                
                if self.best_cost < 0.5:
                    print(f"\n*** SOLUTION FOUND at iteration {it}! ***")
                    break
        
        elapsed = time.time() - t0
        print(f"\nSearch complete in {elapsed:.1f}s")
        print(f"Best L2 cost: {self.best_cost:.1f}")
        print(f"Best Linf deviation: {np.max(np.abs(self.best_psd - N)):.2f}")
        
        return self.best_seqs, self.best_cost, log_data


class SmartSearch:
    """
    Smarter search using multi-scale moves and adaptive temperature.
    Focuses on the Williamson approach (symmetric sequences) with
    orbit-aware moves.
    """
    
    def __init__(self, mode='general', seed=42):
        self.rng = np.random.RandomState(seed)
        self.mode = mode
        
        # Build negation orbits for symmetric sequences
        self.orbits = [frozenset([0])]
        for j in range(1, (P+1)//2):
            self.orbits.append(frozenset([j, P - j]))
        
        # Precompute QR structure
        self.chi = np.array([legendre_symbol(j) for j in range(P)], dtype=np.float64)
        self.chi[0] = 0
    
    def _make_symmetric(self, seq):
        """Make a sequence symmetric: s[j] = s[p-j]."""
        s = seq.copy()
        for j in range(1, (P+1)//2):
            s[P-j] = s[j]
        return s
    
    def _flip_orbit(self, seq, orb):
        """Flip all entries in an orbit."""
        s = seq.copy()
        for j in orb:
            s[j] = -s[j]
        return s
    
    def search_williamson(self, n_iterations=5_000_000, seed=42):
        """
        Search for Williamson-type matrices: 4 symmetric ±1 circulant matrices.
        A^2 + B^2 + C^2 + D^2 = 4pI (since symmetric circulants are normal).
        """
        rng = np.random.RandomState(seed)
        n_orbits = len(self.orbits)
        
        # Initialize 4 random symmetric sequences
        seqs = []
        for _ in range(4):
            s = np.ones(P, dtype=np.float64)
            for orb in self.orbits:
                val = rng.choice([-1.0, 1.0])
                for j in orb:
                    s[j] = val
            seqs.append(s)
        
        # Compute initial PSDs
        psd = compute_psd(seqs)
        cost = np.sum((psd - N)**2)
        best_cost = cost
        best_seqs = [s.copy() for s in seqs]
        
        T = 200.0
        T_min = 0.01
        cooling = (T_min / T) ** (1.0 / n_iterations)
        
        accepted = 0
        
        t0 = time.time()
        
        for it in range(1, n_iterations + 1):
            # Pick random sequence and random orbit
            seq_idx = rng.randint(4)
            orb_idx = rng.randint(n_orbits)
            orb = self.orbits[orb_idx]
            
            # Flip the orbit
            new_seq = self._flip_orbit(seqs[seq_idx], orb)
            
            # Recompute cost (incremental would be better but orbit flips are complex)
            new_seqs = seqs.copy()
            new_seqs[seq_idx] = new_seq
            new_psd = compute_psd(new_seqs)
            new_cost = np.sum((new_psd - N)**2)
            
            delta = new_cost - cost
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                seqs = new_seqs
                psd = new_psd
                cost = new_cost
                accepted += 1
                
                if cost < best_cost:
                    best_cost = cost
                    best_seqs = [s.copy() for s in seqs]
            
            T *= cooling
            
            if cost < 0.5:
                print(f"\n*** WILLIAMSON SOLUTION FOUND at iteration {it}! ***")
                return best_seqs, best_cost
            
            if it % 500_000 == 0:
                elapsed = time.time() - t0
                max_dev = np.max(np.abs(compute_psd(best_seqs) - N))
                print(f"  Williamson iter {it:>10,} | {elapsed:>7.1f}s | "
                      f"cost={best_cost:>10.1f} | Linf={max_dev:>6.2f} | "
                      f"T={T:.4f} | accept_rate={accepted/it:.3f}")
        
        return best_seqs, best_cost

    def search_general_multistart(self, n_restarts=20, iters_per_restart=2_000_000, seed=42):
        """Multi-start SA with general (non-symmetric) sequences."""
        overall_best_cost = float('inf')
        overall_best_seqs = None
        
        for restart in range(n_restarts):
            rng = np.random.RandomState(seed + restart * 1000)
            
            # Initialize with Legendre-based + perturbation
            chi_base = np.array([legendre_symbol(j) for j in range(P)], dtype=np.float64)
            chi_base[0] = 1
            
            seqs = []
            for i in range(4):
                s = chi_base.copy()
                if i < 3:
                    # Perturb ~10% of entries
                    flip_mask = rng.random(P) < 0.10
                    s[flip_mask] = -s[flip_mask]
                else:
                    # Last one: try all-ones or random
                    if restart % 3 == 0:
                        s = np.ones(P, dtype=np.float64)
                    elif restart % 3 == 1:
                        s = -chi_base.copy()
                        mask = rng.random(P) < 0.15
                        s[mask] = -s[mask]
                    # else: perturbed chi
                seqs.append(s)
            
            # Precompute FFTs
            seq_ffts = [fft(s) for s in seqs]
            psd = sum(np.abs(sf)**2 for sf in seq_ffts)
            cost = np.sum((psd - N)**2)
            
            best_cost = cost
            best_seqs = [s.copy() for s in seqs]
            
            T = 100.0
            T_min = 0.001
            cooling = (T_min / T) ** (1.0 / iters_per_restart)
            
            for it in range(1, iters_per_restart + 1):
                # Single flip
                seq_idx = rng.randint(4)
                pos = rng.randint(P)
                
                new_psd, new_fft = incremental_psd_update(psd, seq_ffts, seq_idx, pos, seqs)
                new_cost = np.sum((new_psd - N)**2)
                
                delta = new_cost - cost
                if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                    seqs[seq_idx][pos] *= -1
                    seq_ffts[seq_idx] = new_fft
                    psd = new_psd
                    cost = new_cost
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_seqs = [s.copy() for s in seqs]
                
                T *= cooling
                
                if cost < 0.5:
                    print(f"\n*** SOLUTION FOUND at restart {restart}, iteration {it}! ***")
                    return best_seqs, best_cost
            
            if best_cost < overall_best_cost:
                overall_best_cost = best_cost
                overall_best_seqs = best_seqs
            
            psd_best = compute_psd(best_seqs)
            max_dev = np.max(np.abs(psd_best - N))
            print(f"  Restart {restart:>3}: best_cost={best_cost:>10.1f} | "
                  f"Linf={max_dev:>6.2f} | overall_best={overall_best_cost:>10.1f}")
        
        return overall_best_seqs, overall_best_cost


def main():
    print("="*70)
    print("HADAMARD MATRIX H(668) SEARCH ENGINE")
    print("="*70)
    print(f"Target: 4 sequences of length {P} with PSD = {N} at all frequencies")
    print()
    
    # Strategy 1: Parallel Tempering
    print("Strategy 1: Parallel Tempering")
    print("-"*70)
    pt = ParallelTemperingSearch(n_replicas=12, seed=42)
    best_seqs, best_cost, log_data = pt.run(
        n_iterations=2_000_000,
        log_interval=200_000,
        exchange_interval=50
    )
    
    if best_cost < 0.5:
        print("SUCCESS! Building Hadamard matrix...")
        from hadamard_core import goethals_seidel_array, verify_hadamard, export_csv
        seqs_int = [s.astype(np.int8) for s in best_seqs]
        H = goethals_seidel_array(seqs_int[0], seqs_int[1], seqs_int[2], seqs_int[3])
        valid, msg = verify_hadamard(H)
        print(f"Verification: {msg}")
        if valid:
            export_csv(H, "hadamard_668.csv")
            print("SAVED to hadamard_668.csv")
            return True
    
    # Strategy 2: Williamson (symmetric) search
    print("\nStrategy 2: Williamson Symmetric Search")
    print("-"*70)
    ss = SmartSearch(mode='williamson', seed=42)
    best_seqs_w, best_cost_w = ss.search_williamson(n_iterations=3_000_000, seed=42)
    
    if best_cost_w < 0.5:
        print("SUCCESS via Williamson!")
        from hadamard_core import goethals_seidel_array, verify_hadamard, export_csv
        seqs_int = [s.astype(np.int8) for s in best_seqs_w]
        H = goethals_seidel_array(seqs_int[0], seqs_int[1], seqs_int[2], seqs_int[3])
        valid, msg = verify_hadamard(H)
        print(f"Verification: {msg}")
        if valid:
            export_csv(H, "hadamard_668.csv")
            return True
    
    # Strategy 3: Multi-start general SA
    print("\nStrategy 3: Multi-start General SA")
    print("-"*70)
    best_seqs_g, best_cost_g = ss.search_general_multistart(
        n_restarts=10, iters_per_restart=1_000_000, seed=42
    )
    
    if best_cost_g < 0.5:
        print("SUCCESS via General SA!")
        from hadamard_core import goethals_seidel_array, verify_hadamard, export_csv
        seqs_int = [s.astype(np.int8) for s in best_seqs_g]
        H = goethals_seidel_array(seqs_int[0], seqs_int[1], seqs_int[2], seqs_int[3])
        valid, msg = verify_hadamard(H)
        if valid:
            export_csv(H, "hadamard_668.csv")
            return True
    
    # Report best results
    all_costs = [best_cost, best_cost_w, best_cost_g]
    print(f"\nBest costs: PT={best_cost:.1f}, Williamson={best_cost_w:.1f}, General={best_cost_g:.1f}")
    print("No exact solution found.")
    
    # Save log
    with open("results/experiments/search_log.json", "w") as f:
        json.dump({
            "pt_final_cost": float(best_cost),
            "williamson_final_cost": float(best_cost_w),
            "general_final_cost": float(best_cost_g),
            "pt_log": log_data
        }, f, indent=2)
    
    return False

if __name__ == "__main__":
    main()
