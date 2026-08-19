import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "RotOnlyMatrix"
ATOL = 1e-10

def pair_block_cycles(num_pairs, cycle_length):
    for subset in itertools.combinations(range(num_pairs), cycle_length):
        first = subset[0]
        for tail in itertools.permutations(subset[1:]):
            yield (first, *tail)

def inverse_cycle(cycle):
    return (cycle[0], *reversed(cycle[1:]))

def create_pair_block_permutation_matrix(cycle, num_qubit):
    if num_qubit % 2:
        raise ValueError("num_qubit must be even.")
    num_pairs = num_qubit // 2
    if any(pair < 0 or pair >= num_pairs for pair in cycle):
        raise ValueError(f"Invalid pair-block cycle {cycle} for {num_pairs} pairs.")

    wire_destination = list(range(num_qubit))
    for source, destination in zip(cycle, cycle[1:] + cycle[:1]):
        wire_destination[2 * source] = 2 * destination
        wire_destination[2 * source + 1] = 2 * destination + 1

    dim = 2**num_qubit
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    for column in range(dim):
        bits = [(column >> wire) & 1 for wire in range(num_qubit)]
        row = sum(bits[wire] << wire_destination[wire] for wire in range(num_qubit))
        matrix[row, column] = 1.0
    return matrix

def _cycle_key(cycle):
    return tuple(cycle)

def _save_generators_for_k(num_qubit, cycle_length, output_dir):
    cycles = list(pair_block_cycles(num_qubit // 2, cycle_length))
    expected = math.comb(num_qubit // 2, cycle_length) * math.factorial(cycle_length - 1)
    if len(cycles) != expected:
        raise AssertionError(f"Expected {expected} cycles, found {len(cycles)}.")

    processed = set()
    entries = []
    for cycle in cycles:
        key = _cycle_key(cycle)
        if key in processed:
            continue

        raw = create_pair_block_permutation_matrix(cycle, num_qubit)
        inverse = inverse_cycle(cycle)
        inverse_key = _cycle_key(inverse)
        if inverse_key not in {_cycle_key(item) for item in cycles}:
            raise AssertionError(f"Inverse cycle {inverse} is missing.")

        if np.allclose(raw, raw.conj().T, atol=ATOL):
            components = [("herm", raw, [list(cycle)])]
            processed.add(key)
        else:
            components = [
                ("re", (raw + raw.conj().T) / 2, [list(cycle), list(inverse)]),
                ("im", (raw - raw.conj().T) / (2j), [list(cycle), list(inverse)]),
            ]
            processed.update((key, inverse_key))

        for component, matrix, source_cycles in components:
            if not np.allclose(matrix, matrix.conj().T, atol=ATOL):
                raise AssertionError(f"Non-Hermitian {component} component for cycle {cycle}.")
            index = len(entries)
            filename = (
                f"perm_matrix_rot_only_block_{num_qubit}_{cycle_length}_"
                f"gen_{index:03d}_{component}.npy"
            )
            np.save(output_dir / filename, np.asarray(matrix, dtype=np.complex128))
            entries.append(
                {
                    "index": index,
                    "filename": filename,
                    "component": component,
                    "source_cycles": source_cycles,
                }
            )

    if len(entries) != expected:
        raise AssertionError(
            f"Hermitian decomposition must preserve the real generator count: "
            f"expected {expected}, found {len(entries)}."
        )
    return entries

def rot_only_generators_per_block(num_qubit):
    num_pairs = num_qubit // 2
    return sum(
        math.comb(num_pairs, cycle_length) * math.factorial(cycle_length - 1)
        for cycle_length in range(2, num_pairs + 1)
    )

def generate_rot_only_matrices(num_qubit, output_dir=OUT_DIR):
    if num_qubit % 2:
        raise ValueError("num_qubit must be even.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 2,
        "generator_family": "pair_block_cycles_without_pair_twirling",
        "num_qubit": num_qubit,
        "num_pairs": num_qubit // 2,
        "sectors": None,
        "generators": {},
    }
    for cycle_length in range(2, num_qubit // 2 + 1):
        entries = _save_generators_for_k(num_qubit, cycle_length, output_dir)
        manifest["generators"][str(cycle_length)] = entries
        print(f"k={cycle_length}: {len(entries)} Hermitian pair-block generators")

    total = sum(len(entries) for entries in manifest["generators"].values())
    expected_total = rot_only_generators_per_block(num_qubit)
    if total != expected_total:
        raise AssertionError(f"Expected {expected_total} generators, found {total}.")
    manifest["generators_per_block"] = total

    metadata_path = output_dir / f"perm_matrix_rot_only_block_{num_qubit}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(f"Saved {metadata_path} ({total} generators per block)")
    return metadata_path

def _matrix_exponential_from_hermitian(theta, matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return (eigenvectors * np.exp(1j * theta * eigenvalues)) @ eigenvectors.conj().T

def _global_su2(num_qubit, rng):
    a, b = rng.normal(size=2) + 1j * rng.normal(size=2)
    scale = np.sqrt(abs(a) ** 2 + abs(b) ** 2)
    single = np.array([[a, b], [-b.conjugate(), a.conjugate()]]) / scale
    result = np.array([[1.0 + 0.0j]])
    for _ in range(num_qubit):
        result = np.kron(result, single)
    return result

def verify_symmetries(num_qubit, output_dir=OUT_DIR, seed=7, tolerance=1e-9):
    if num_qubit < 6:
        raise ValueError("Use at least 6 qubits so a nontrivial pair permutation can be tested.")
    output_dir = Path(output_dir)
    metadata_path = output_dir / f"perm_matrix_rot_only_block_{num_qubit}_metadata.json"
    if not metadata_path.exists():
        generate_rot_only_matrices(num_qubit, output_dir)

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    rng = np.random.default_rng(seed)
    dim = 2**num_qubit
    layer = np.eye(dim, dtype=np.complex128)
    for entries in metadata["generators"].values():
        for entry in entries:
            generator = np.load(output_dir / entry["filename"])
            if not np.allclose(generator, generator.conj().T, atol=tolerance):
                raise AssertionError(f"{entry['filename']} is not Hermitian.")
            gate = _matrix_exponential_from_hermitian(rng.normal(), generator)
            if not np.allclose(gate.conj().T @ gate, np.eye(dim), atol=tolerance):
                raise AssertionError(f"{entry['filename']} did not produce a unitary gate.")
            layer = gate @ layer

    state = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    state /= np.linalg.norm(state)
    rotation = _global_su2(num_qubit, rng)
    pair_swap = create_pair_block_permutation_matrix((0, 1), num_qubit)

    rotation_error = np.linalg.norm(layer @ (rotation @ state) - rotation @ (layer @ state))
    pair_difference = np.linalg.norm(layer @ (pair_swap @ state) - pair_swap @ (layer @ state))

    print(f"global-SU(2) equivariance error: {rotation_error:.3e}")
    print(f"pair-permutation equivariance violation: {pair_difference:.3e}")
    if rotation_error >= tolerance:
        raise AssertionError("Rotation equivariance check failed.")
    if pair_difference <= 100 * tolerance:
        raise AssertionError("Generic parameters unexpectedly preserved pair-permutation equivariance.")
    return rotation_error, pair_difference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_qubit", type=int, required=True)
    args = parser.parse_args()

    generate_rot_only_matrices(args.num_qubit)

if __name__ == "__main__":
    main()
