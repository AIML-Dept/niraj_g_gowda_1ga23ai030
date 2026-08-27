"""
Simon's Algorithm — Tutorial 6 (AML23703, Quantum Computing)
================================================================

Covers all hands-on exercises from the tutorial handbook:
  1. Easy      - Build Simon's oracle for secret string '110' and verify
                 it against a truth table.
  2. Medium    - Run the full Simon's algorithm circuit for a 3-bit
                 secret and collect n-1 independent equations.
  3. Hard      - Extend to a 4-bit secret string and solve the resulting
                 linear system over GF(2) programmatically.
  4. Real-world- Compare oracle-query counts: Simon's algorithm vs.
                 classical brute force, for varying string lengths, and
                 plot the speedup.
  5. Challenge - A generalised simon_solve(oracle, n) function that works
                 for any secret bitstring, including the all-zero case.

Requirements:
    pip install qiskit qiskit-aer matplotlib numpy

Run:
    python simons_algorithm.py
"""

import itertools
import random
from typing import Callable, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile


# ---------------------------------------------------------------------------
# 1. EASY — Oracle construction for a fixed secret string
# ---------------------------------------------------------------------------

def build_simon_oracle(secret: str) -> QuantumCircuit:
    """
    Build a quantum oracle circuit implementing f(x) such that
    f(x) = f(y)  <=>  y = x XOR secret.

    Uses n input qubits + n output qubits (2n total).
    Standard construction:
        1. CNOT each input qubit i onto output qubit i (copies x onto output).
        2. Find the first '1' bit in `secret`, say at position j.
        3. For every other bit i where secret[i] == '1', add a CNOT from
           input qubit j onto output qubit i.
           (This entangles the outputs of x and x XOR secret identically.)
        4. If secret is all zeros, the function is 1-to-1 (skip step 3),
           which Simon's algorithm/post-processing must handle separately.
    """
    n = len(secret)
    qc = QuantumCircuit(2 * n, name=f"Oracle[{secret}]")

    # Step 1: copy inputs to outputs
    for i in range(n):
        qc.cx(i, n + i)

    # Step 2 & 3: only apply the "collision" pattern if secret != 0...0
    if "1" in secret:
        # bits are read left-to-right in the string; qubit index i <-> bit i
        j = secret.find("1")  # first qubit that has a '1' in the secret
        for i, bit in enumerate(secret):
            if bit == "1" and i != j:
                qc.cx(j, n + i)

    return qc


def verify_oracle_truth_table(secret: str) -> None:
    """
    Verify build_simon_oracle(secret) satisfies f(x) = f(x XOR secret)
    for every input x, by simulating the oracle classically (statevector
    simulation over basis states) and printing a truth table.
    """
    n = len(secret)
    oracle = build_simon_oracle(secret)
    sim = AerSimulator(method="statevector")

    print(f"\nTruth table for secret = '{secret}' (n = {n})")
    print(f"{'x':^{n}} | {'f(x)':^{n}}")
    print("-" * (2 * n + 3))

    outputs = {}
    for x_int in range(2 ** n):
        x_bits = format(x_int, f"0{n}b")
        qc = QuantumCircuit(2 * n, n)
        # Prepare |x> on the input register (qubits 0..n-1)
        for i, bit in enumerate(reversed(x_bits)):
            if bit == "1":
                qc.x(i)
        qc.compose(oracle, inplace=True)
        qc.measure(range(n, 2 * n), range(n))

        tqc = transpile(qc, sim)
        result = sim.run(tqc, shots=1).result()
        f_x = list(result.get_counts().keys())[0]
        outputs[x_bits] = f_x
        print(f"{x_bits:^{n}} | {f_x:^{n}}")

    # Correctness check: f(x) == f(x XOR secret) for all x
    ok = True
    for x_bits, f_x in outputs.items():
        x_int = int(x_bits, 2)
        s_int = int(secret, 2)
        y_bits = format(x_int ^ s_int, f"0{n}b")
        if outputs[y_bits] != f_x:
            ok = False
            print(f"MISMATCH: f({x_bits}) != f({y_bits})")
    print("Oracle verified correct." if ok else "Oracle FAILED verification.")


# ---------------------------------------------------------------------------
# Core circuit: Simon's algorithm quantum part (used by exercises 2-5)
# ---------------------------------------------------------------------------

def simon_circuit(oracle: QuantumCircuit, n: int) -> QuantumCircuit:
    """Build the full Simon's algorithm circuit: H^n -> Oracle -> H^n -> measure."""
    qc = QuantumCircuit(2 * n, n)
    qc.h(range(n))
    qc.compose(oracle, inplace=True)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


def run_simon_circuit(oracle: QuantumCircuit, n: int, shots: int = 1) -> str:
    """Run the Simon circuit once and return the measured bitstring."""
    sim = AerSimulator()
    qc = simon_circuit(oracle, n)
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return list(counts.keys())[0]


# ---------------------------------------------------------------------------
# GF(2) linear algebra helpers (used by exercises 3 & 5)
# ---------------------------------------------------------------------------

def gf2_gaussian_elimination(rows: List[List[int]]) -> List[List[int]]:
    """Row-reduce a matrix over GF(2) (in place on a copy). Returns echelon form."""
    mat = [row[:] for row in rows]
    n_cols = len(mat[0]) if mat else 0
    pivot_row = 0

    for col in range(n_cols):
        # find a pivot in this column at/after pivot_row
        pivot = None
        for r in range(pivot_row, len(mat)):
            if mat[r][col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        mat[pivot_row], mat[pivot] = mat[pivot], mat[pivot_row]
        for r in range(len(mat)):
            if r != pivot_row and mat[r][col] == 1:
                mat[r] = [(a ^ b) for a, b in zip(mat[r], mat[pivot_row])]
        pivot_row += 1
        if pivot_row == len(mat):
            break
    return mat


def solve_secret_from_equations(equations: List[str], n: int) -> Optional[str]:
    """
    Given a list of bitstrings y (each satisfying y . s = 0 mod 2),
    find the non-trivial secret s over GF(2) by brute-forcing candidates
    that satisfy every equation. Robust and simple — fine for tutorial-scale n.
    """
    # de-duplicate and drop the trivial all-zero equation (y=0 gives no info)
    eqs = sorted(set(y for y in equations if y != "0" * n))
    if not eqs:
        return "0" * n  # no useful info collected

    candidates = []
    for s_int in range(1, 2 ** n):  # skip s = 0 (found separately if needed)
        s_bits = format(s_int, f"0{n}b")
        s_vec = [int(b) for b in s_bits]
        if all(sum(int(y[i]) * s_vec[i] for i in range(n)) % 2 == 0 for y in eqs):
            candidates.append(s_bits)

    if len(candidates) == 1:
        return candidates[0]
    return candidates if candidates else None  # ambiguous (need more equations) or none


# ---------------------------------------------------------------------------
# 2. MEDIUM — Full 3-bit run, collecting n-1 independent equations
# ---------------------------------------------------------------------------

def run_medium_exercise(secret: str = "101") -> None:
    n = len(secret)
    oracle = build_simon_oracle(secret)
    equations = []

    print(f"\n--- Medium exercise: secret = '{secret}' ---")
    while len(equations) < n - 1:
        y = run_simon_circuit(oracle, n)
        if y != "0" * n and y not in equations:
            # check independence against existing set using rank test
            trial = equations + [y]
            rows = [[int(b) for b in row] for row in trial]
            reduced = gf2_gaussian_elimination(rows)
            rank = sum(1 for row in reduced if any(row))
            if rank == len(trial):
                equations.append(y)
                print(f"  Collected independent equation: {y}")

    recovered = solve_secret_from_equations(equations, n)
    print(f"Equations collected: {equations}")
    print(f"Recovered secret: {recovered}  (actual: {secret})")


# ---------------------------------------------------------------------------
# 3. HARD — 4-bit secret, solved programmatically over GF(2)
# ---------------------------------------------------------------------------

def run_hard_exercise(secret: str = "1011") -> None:
    n = len(secret)
    oracle = build_simon_oracle(secret)
    equations = []

    print(f"\n--- Hard exercise: secret = '{secret}' ---")
    attempts = 0
    while len(equations) < n - 1 and attempts < 200:
        attempts += 1
        y = run_simon_circuit(oracle, n)
        if y == "0" * n:
            continue
        trial = equations + [y]
        rows = [[int(b) for b in row] for row in trial]
        reduced = gf2_gaussian_elimination(rows)
        rank = sum(1 for row in reduced if any(row))
        if rank == len(trial):
            equations.append(y)

    recovered = solve_secret_from_equations(equations, n)
    print(f"Equations collected ({len(equations)}): {equations}")
    print(f"Recovered secret: {recovered}  (actual: {secret})")


# ---------------------------------------------------------------------------
# 5. CHALLENGE — Generalised solver for any secret, incl. all-zero case
# ---------------------------------------------------------------------------

def simon_solve(secret: str, max_attempts: int = 300) -> str:
    """
    Fully generalised Simon's algorithm: given a hidden `secret` (used only
    to build the oracle, simulating a black box), determine it via quantum
    queries + classical post-processing. Handles the all-zero edge case by
    falling back to classical verification once no non-trivial secret is
    consistent with the collected equations.
    """
    n = len(secret)
    oracle = build_simon_oracle(secret)
    equations: List[str] = []
    attempts = 0

    while len(equations) < n - 1 and attempts < max_attempts:
        attempts += 1
        y = run_simon_circuit(oracle, n)
        if y == "0" * n:
            continue
        trial = equations + [y]
        rows = [[int(b) for b in row] for row in trial]
        reduced = gf2_gaussian_elimination(rows)
        rank = sum(1 for row in reduced if any(row))
        if rank == len(trial):
            equations.append(y)

    candidate = solve_secret_from_equations(equations, n)

    # Disambiguate / handle s = 0: classically test f(0...0) vs f(e_i)
    def oracle_output(x_bits: str) -> str:
        sim = AerSimulator(method="statevector")
        qc = QuantumCircuit(2 * n, n)
        for i, bit in enumerate(reversed(x_bits)):
            if bit == "1":
                qc.x(i)
        qc.compose(oracle, inplace=True)
        qc.measure(range(n, 2 * n), range(n))
        tqc = transpile(qc, sim)
        result = sim.run(tqc, shots=1).result()
        return list(result.get_counts().keys())[0]

    if candidate is None or isinstance(candidate, list):
        # Not enough independent equations, or ambiguous — test s=0 directly:
        # if f is 1-to-1 (all f(e_i) distinct and != f(0)), secret is 0...0.
        zero_out = oracle_output("0" * n)
        basis_outs = [oracle_output(format(1 << i, f"0{n}b")) for i in range(n)]
        if len(set([zero_out] + basis_outs)) == n + 1:
            return "0" * n
        # otherwise fall back to whichever single candidate remains, if any
        if isinstance(candidate, list) and len(candidate) >= 1:
            for cand in candidate:
                if oracle_output("0" * n) == oracle_output(cand):
                    return cand
        return candidate if isinstance(candidate, str) else "0" * n

    return candidate


# ---------------------------------------------------------------------------
# 4. REAL-WORLD — Query complexity comparison: Simon vs classical brute force
# ---------------------------------------------------------------------------

def classical_query_count_worst_case(n: int) -> int:
    """
    Classical worst case to detect a 2-to-1 collision (find secret s):
    by pigeonhole, may need up to 2^(n-1) + 1 queries to guarantee a
    collision is found (birthday-paradox lower bound is often quoted,
    but the deterministic worst case is exponential: O(2^(n/2)) expected
    with random sampling, O(2^(n-1)+1) worst-case deterministic).
    We report the standard deterministic worst-case bound used in courses.
    """
    return 2 ** (n - 1) + 1


def simon_query_count(n: int) -> int:
    """Simon's algorithm needs O(n) oracle queries (n-1 independent + a few retries)."""
    return n - 1  # ideal/expected order; used for the asymptotic comparison plot


def compare_and_plot(max_n: int = 10) -> None:
    ns = list(range(2, max_n + 1))
    classical_counts = [classical_query_count_worst_case(n) for n in ns]
    simon_counts = [simon_query_count(n) for n in ns]

    print("\n--- Real-world exercise: query complexity comparison ---")
    print(f"{'n':>3} | {'Classical (worst-case)':>22} | {'Simon (O(n))':>12}")
    for n, c, s in zip(ns, classical_counts, simon_counts):
        print(f"{n:>3} | {c:>22} | {s:>12}")

    plt.figure(figsize=(7, 5))
    plt.plot(ns, classical_counts, marker="o", label="Classical (worst-case, exponential)")
    plt.plot(ns, simon_counts, marker="s", label="Simon's Algorithm (O(n))")
    plt.yscale("log")
    plt.xlabel("Bitstring length n")
    plt.ylabel("Oracle queries required (log scale)")
    plt.title("Simon's Algorithm vs. Classical Brute Force: Query Complexity")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("simon_vs_classical_speedup.png", dpi=150)
    print("Plot saved to simon_vs_classical_speedup.png")


# ---------------------------------------------------------------------------
# MAIN — run all exercises in sequence
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Exercise 1 (Easy)
    verify_oracle_truth_table("110")

    # Exercise 2 (Medium)
    run_medium_exercise("101")

    # Exercise 3 (Hard)
    run_hard_exercise("1011")

    # Exercise 4 (Real-world)
    compare_and_plot(max_n=10)

    # Exercise 5 (Challenge) — test on a few secrets, including all-zero
    print("\n--- Challenge exercise: generalised simon_solve() ---")
    for test_secret in ["101", "1010", "000", "0000"]:
        found = simon_solve(test_secret)
        status = "OK" if found == test_secret else "MISMATCH"
        print(f"secret={test_secret:>5}  ->  found={found:>5}   [{status}]")
