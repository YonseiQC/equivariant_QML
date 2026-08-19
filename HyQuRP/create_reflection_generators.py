import argparse
from collections import Counter
from fractions import Fraction
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path

import numpy as np

def integer_partitions(n, max_part=None):
    if n == 0:
        yield ()
        return
    if max_part is None or max_part > n:
        max_part = n
    for first in range(max_part, 0, -1):
        for rest in integer_partitions(n - first, first):
            yield (first,) + rest

def diagram_cells(partition):
    return {
        (row, col)
        for row, width in enumerate(partition)
        for col in range(width)
    }

def contained_partitions(partition, target_size):
    for candidate in integer_partitions(target_size):
        if len(candidate) <= len(partition) and all(
            candidate[i] <= partition[i] for i in range(len(candidate))
        ):
            yield candidate

def border_strip_height(partition, subpartition):
    """Return the border-strip height, or None if the skew shape is invalid."""
    strip = diagram_cells(partition) - diagram_cells(subpartition)
    if not strip:
        return None

    seen = {next(iter(strip))}
    stack = list(seen)
    while stack:
        row, col = stack.pop()
        for neighbor in (
            (row + 1, col),
            (row - 1, col),
            (row, col + 1),
            (row, col - 1),
        ):
            if neighbor in strip and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    if seen != strip:
        return None

    for row, col in strip:
        square = {
            (row, col),
            (row + 1, col),
            (row, col + 1),
            (row + 1, col + 1),
        }
        if square <= strip:
            return None
    return len({row for row, _ in strip})

@lru_cache(maxsize=None)
def symmetric_group_character(partition, cycle_type):
    """Irreducible S_n character via the Murnaghan-Nakayama rule."""
    if not cycle_type:
        return int(len(partition) == 0)

    rim_hook_size = cycle_type[0]
    target_size = sum(partition) - rim_hook_size
    value = 0
    for subpartition in contained_partitions(partition, target_size):
        height = border_strip_height(partition, subpartition)
        if height is not None:
            value += (-1) ** (height - 1) * symmetric_group_character(
                subpartition, cycle_type[1:]
            )
    return value

def centralizer_size(cycle_type):
    value = 1
    for length, multiplicity in Counter(cycle_type).items():
        value *= (length**multiplicity) * math.factorial(multiplicity)
    return value

def expected_dimensions(num_pairs):
    """Return full, adjoint-even, and adjoint-odd dimensions.

    In each multiplicity block M_m(C), Hermitian matrices decompose into
    m(m+1)/2 real-symmetric and m(m-1)/2 imaginary-skew directions.
    """
    classes = list(integer_partitions(num_pairs))
    irreps = classes
    full_dimension = 0
    even_dimension = 0
    odd_dimension = 0
    multiplicities = {}

    for spin_index in range(num_pairs + 1):
        two_row = (
            (2 * num_pairs - spin_index, spin_index)
            if spin_index
            else (2 * num_pairs,)
        )
        restricted_character = {
            cycle: symmetric_group_character(
                two_row, tuple(sorted(cycle + cycle, reverse=True))
            )
            for cycle in classes
        }

        sector_multiplicities = {}
        for irrep in irreps:
            inner_product = sum(
                Fraction(
                    restricted_character[cycle]
                    * symmetric_group_character(irrep, cycle),
                    centralizer_size(cycle),
                )
                for cycle in classes
            )
            if inner_product.denominator != 1 or inner_product < 0:
                raise RuntimeError(
                    "Character decomposition produced an invalid multiplicity."
                )
            multiplicity = int(inner_product)
            if multiplicity:
                sector_multiplicities[str(irrep)] = multiplicity
                full_dimension += multiplicity**2
                even_dimension += multiplicity * (multiplicity + 1) // 2
                odd_dimension += multiplicity * (multiplicity - 1) // 2
        multiplicities[str(two_row)] = sector_multiplicities

    if even_dimension + odd_dimension != full_dimension:
        raise RuntimeError("Hermitian dimension decomposition is inconsistent.")
    return full_dimension, even_dimension, odd_dimension, multiplicities

def inverse_permutation(permutation):
    inverse = [0] * len(permutation)
    for source, target in enumerate(permutation):
        inverse[target] = source
    return tuple(inverse)

def conjugate_permutation(conjugator, permutation):
    """Return conjugator o permutation o conjugator^{-1}."""
    conjugator_inverse = inverse_permutation(conjugator)
    return tuple(
        conjugator[permutation[conjugator_inverse[index]]]
        for index in range(len(permutation))
    )

def make_pair_permutation_group(num_pairs):
    group = []
    for block_permutation in itertools.permutations(range(num_pairs)):
        wire_permutation = [0] * (2 * num_pairs)
        for block, target_block in enumerate(block_permutation):
            wire_permutation[2 * block] = 2 * target_block
            wire_permutation[2 * block + 1] = 2 * target_block + 1
        group.append(tuple(wire_permutation))
    return group

def twirl_orbits(permutation, pair_group):
    """Return an S_pair orbit, its inverse orbit, and their common line key."""
    orbit = {
        conjugate_permutation(group_element, permutation)
        for group_element in pair_group
    }
    inverse_orbit = {inverse_permutation(item) for item in orbit}

    canonical_key = min(orbit | inverse_orbit)
    return tuple(sorted(orbit)), tuple(sorted(inverse_orbit)), canonical_key

class PermutationRows:
    def __init__(self, num_qubit):
        self.num_qubit = num_qubit
        self.dimension = 2**num_qubit
        basis_indices = np.arange(self.dimension, dtype=np.uint64)
        self.bits = (
            (basis_indices[:, None] >> np.arange(num_qubit, dtype=np.uint64)) & 1
        ).astype(np.uint64)
        self._cache = {}

    def __call__(self, wire_permutation):
        rows = self._cache.get(wire_permutation)
        if rows is None:
            powers = np.left_shift(
                np.uint64(1), np.asarray(wire_permutation, dtype=np.uint64)
            )
            rows = np.asarray(self.bits @ powers, dtype=np.int64)
            self._cache[wire_permutation] = rows
        return rows

def component_action(
    orbit,
    inverse_orbit,
    component,
    probes,
    permutation_rows,
):
    """Apply a twirled component to probes using real arithmetic.

    For the adjoint-odd component, this returns the real skew matrix
    A-A^T acting on the probes.  Multiplication by 1/(2i) is unnecessary for
    rank selection and is restored when the dense Hermitian matrix is built.
    """
    if component == "adjoint_even":
        terms = tuple(sorted(set(orbit) | set(inverse_orbit)))
        output = np.zeros_like(probes)
        for term in terms:
            output[permutation_rows(term)] += probes
        return output / len(terms)

    if orbit == inverse_orbit:

        return np.zeros_like(probes)

    output = np.zeros_like(probes)
    for term in orbit:
        output[permutation_rows(term)] += probes
    for term in inverse_orbit:
        output[permutation_rows(term)] -= probes
    return output / len(orbit)

def add_if_independent(vector, orthonormal_basis, relative_tolerance):
    residual = np.asarray(vector, dtype=np.float64).copy()
    original_norm = np.linalg.norm(residual)
    if original_norm == 0:
        return False

    for _ in range(2):
        for basis_vector in orthonormal_basis:
            residual -= np.dot(basis_vector, residual) * basis_vector

    residual_norm = np.linalg.norm(residual)
    if residual_norm <= relative_tolerance * original_norm:
        return False
    orthonormal_basis.append(residual / residual_norm)
    return True

def select_component_representatives(
    *,
    component,
    num_qubit,
    target_rank,
    pair_group,
    permutation_rows,
    rng,
    probes,
    rank_tolerance,
    max_attempts,
    include_identity,
):
    orthonormal_sketches = []
    selected_representatives = []
    seen_orbit_pairs = set()

    def try_candidate(permutation):
        orbit, inverse_orbit, canonical_key = twirl_orbits(
            permutation, pair_group
        )
        if canonical_key in seen_orbit_pairs:
            return False
        seen_orbit_pairs.add(canonical_key)

        sketch = component_action(
            orbit,
            inverse_orbit,
            component,
            probes,
            permutation_rows,
        ).reshape(-1)
        if add_if_independent(
            sketch, orthonormal_sketches, rank_tolerance
        ):
            selected_representatives.append(canonical_key)
            print(
                f"Selected {component:13s} generator "
                f"{len(selected_representatives):3d}/{target_rank} "
                f"from orbit pair {len(seen_orbit_pairs):4d}"
            )
            return True
        return False

    if component == "adjoint_even" and include_identity:
        try_candidate(tuple(range(num_qubit)))

    attempts = 0
    while len(selected_representatives) < target_rank and attempts < max_attempts:
        attempts += 1
        random_permutation = tuple(rng.permutation(num_qubit).tolist())
        try_candidate(random_permutation)

    if len(selected_representatives) != target_rank:
        raise RuntimeError(
            f"Found {component} rank {len(selected_representatives)}, "
            f"expected {target_rank}, after {attempts} random attempts. "
            "Increase --max_attempts or --sketch_probes."
        )
    return selected_representatives

def verify_component_rank(
    representatives,
    component,
    pair_group,
    permutation_rows,
    probes,
    tolerance,
):
    sketches = []
    for representative in representatives:
        orbit, inverse_orbit, _ = twirl_orbits(representative, pair_group)
        sketches.append(
            component_action(
                orbit,
                inverse_orbit,
                component,
                probes,
                permutation_rows,
            ).reshape(-1)
        )
    sketch_matrix = np.stack(sketches, axis=0)
    singular_values = np.linalg.svd(sketch_matrix, compute_uv=False)
    numerical_rank = int(
        np.sum(singular_values > tolerance * singular_values[0])
    )
    if numerical_rank != len(representatives):
        raise RuntimeError(
            f"Independent verification found {component} rank "
            f"{numerical_rank}, but {len(representatives)} generators were selected."
        )
    return float(singular_values[-1] / singular_values[0])

def dense_component_matrix(
    orbit,
    inverse_orbit,
    component,
    permutation_rows,
    real_dtype,
):
    dimension = permutation_rows.dimension
    columns = np.arange(dimension, dtype=np.int64)

    if component == "adjoint_even":
        terms = tuple(sorted(set(orbit) | set(inverse_orbit)))
        matrix = np.zeros((dimension, dimension), dtype=real_dtype)
        coefficient = 1.0 / len(terms)
        for term in terms:
            matrix[permutation_rows(term), columns] += coefficient
    else:
        complex_dtype = np.complex64 if real_dtype == np.float32 else np.complex128
        matrix = np.zeros((dimension, dimension), dtype=complex_dtype)

        coefficient = -0.5j / len(orbit)
        for term in orbit:
            matrix[permutation_rows(term), columns] += coefficient
        for term in inverse_orbit:
            matrix[permutation_rows(term), columns] -= coefficient

    matrix_for_norm = matrix.astype(np.complex128, copy=False)
    rms_eigenvalue = math.sqrt(
        float(np.sum(np.abs(matrix_for_norm) ** 2)) / dimension
    )
    if rms_eigenvalue == 0:
        raise RuntimeError("Constructed a zero generator unexpectedly.")
    matrix /= rms_eigenvalue
    return matrix, rms_eigenvalue

def conjugate_dense_by_wire_permutation(
    matrix, wire_permutation, permutation_rows
):
    rows = permutation_rows(wire_permutation)
    conjugated = np.empty_like(matrix)
    conjugated[np.ix_(rows, rows)] = matrix
    return conjugated

def generate(args):
    if args.num_qubit <= 0 or args.num_qubit % 2:
        raise ValueError("--num_qubit must be a positive even integer.")

    num_pairs = args.num_qubit // 2
    if num_pairs > 5 and not args.allow_larger:
        raise ValueError(
            "This dense implementation is intended for at most five pairs. "
            "Pass --allow_larger only if you understand the factorial and 2^n costs."
        )

    (
        full_dimension,
        even_dimension,
        odd_dimension,
        multiplicities,
    ) = expected_dimensions(num_pairs)
    print(f"Full Hermitian operator dimension : {full_dimension}")
    print(f"(A + A^dagger)/2 dimension       : {even_dimension}")
    print(f"(A - A^dagger)/(2i) dimension    : {odd_dimension}")
    if num_pairs == 5 and (
        full_dimension,
        even_dimension,
        odd_dimension,
    ) != (206, 134, 72):
        raise RuntimeError(
            "The N=5 dimension check failed; expected (206, 134, 72)."
        )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent / "ReflectionMatrix"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_group = make_pair_permutation_group(num_pairs)
    permutation_rows = PermutationRows(args.num_qubit)

    include_identity = True
    saved_even_rank = even_dimension if include_identity else even_dimension - 1
    saved_target_rank = saved_even_rank + odd_dimension
    print(
        f"Matrices to save                  : {saved_target_rank} "
        f"({'including' if include_identity else 'excluding'} identity/global phase)"
    )

    rng = np.random.default_rng(args.seed)
    selection_probes = rng.standard_normal(
        (permutation_rows.dimension, args.sketch_probes)
    )
    even_representatives = select_component_representatives(
        component="adjoint_even",
        num_qubit=args.num_qubit,
        target_rank=saved_even_rank,
        pair_group=pair_group,
        permutation_rows=permutation_rows,
        rng=rng,
        probes=selection_probes,
        rank_tolerance=args.rank_tolerance,
        max_attempts=args.max_attempts,
        include_identity=include_identity,
    )
    odd_representatives = select_component_representatives(
        component="adjoint_odd",
        num_qubit=args.num_qubit,
        target_rank=odd_dimension,
        pair_group=pair_group,
        permutation_rows=permutation_rows,
        rng=rng,
        probes=selection_probes,
        rank_tolerance=args.rank_tolerance,
        max_attempts=args.max_attempts,
        include_identity=False,
    )

    verification_rng = np.random.default_rng(args.seed + 1)
    verification_probes = verification_rng.standard_normal(
        (permutation_rows.dimension, args.verify_probes)
    )
    even_verification_ratio = verify_component_rank(
        even_representatives,
        "adjoint_even",
        pair_group,
        permutation_rows,
        verification_probes,
        args.rank_tolerance,
    )
    odd_verification_ratio = verify_component_rank(
        odd_representatives,
        "adjoint_odd",
        pair_group,
        permutation_rows,
        verification_probes,
        args.rank_tolerance,
    )
    print(
        "Independent sketch ranks verified; "
        f"even sigma_min/sigma_max={even_verification_ratio:.3e}, "
        f"odd sigma_min/sigma_max={odd_verification_ratio:.3e}"
    )

    real_dtype = np.float32 if args.dtype == "float32" else np.float64
    global_internal_swap = tuple(
        index + 1 if index % 2 == 0 else index - 1
        for index in range(args.num_qubit)
    )
    first_nontrivial_pair_permutation = (
        pair_group[1] if len(pair_group) > 1 else pair_group[0]
    )

    identity_permutation = tuple(range(args.num_qubit))
    nonidentity_even = [
        representative
        for representative in even_representatives
        if representative != identity_permutation
    ]
    selected = []
    for component_index in range(
        max(len(nonidentity_even), len(odd_representatives))
    ):
        if component_index < len(nonidentity_even):
            selected.append(
                ("adjoint_even", nonidentity_even[component_index])
            )
        if component_index < len(odd_representatives):
            selected.append(
                ("adjoint_odd", odd_representatives[component_index])
            )
    if include_identity:
        selected.append(("adjoint_even", identity_permutation))

    if len(selected) != saved_target_rank:
        raise RuntimeError("Internal ordering lost a selected generator.")

    entries = []
    w_breaking_count = 0
    for index, (component, representative) in enumerate(selected):
        orbit, inverse_orbit, _ = twirl_orbits(representative, pair_group)
        matrix, pre_normalization_rms = dense_component_matrix(
            orbit,
            inverse_orbit,
            component,
            permutation_rows,
            real_dtype,
        )

        hermitian_error = float(
            np.max(np.abs(matrix - matrix.conj().T))
        )
        if hermitian_error > 5e-6:
            raise RuntimeError(
                f"Generator {index} is not Hermitian: error={hermitian_error}"
            )

        w_conjugated = conjugate_dense_by_wire_permutation(
            matrix, global_internal_swap, permutation_rows
        )
        w_commutator_relative = float(
            np.linalg.norm(
                (w_conjugated - matrix).astype(np.complex128)
            )
            / np.linalg.norm(matrix.astype(np.complex128))
        )
        if w_commutator_relative > 1e-7:
            w_breaking_count += 1

        pair_conjugated = conjugate_dense_by_wire_permutation(
            matrix,
            first_nontrivial_pair_permutation,
            permutation_rows,
        )
        pair_invariance_error = float(
            np.linalg.norm(
                (pair_conjugated - matrix).astype(np.complex128)
            )
            / np.linalg.norm(matrix.astype(np.complex128))
        )
        if pair_invariance_error > 5e-6:
            raise RuntimeError(
                f"Generator {index} failed pair-permutation invariance: "
                f"error={pair_invariance_error}"
            )

        filename = (
            f"reflection_generator_{args.num_qubit}_{index:03d}.npy"
        )
        np.save(output_dir / filename, matrix)
        entries.append(
            {
                "index": index,
                "filename": filename,
                "component": component,
                "formula": (
                    "(A+A^dagger)/2"
                    if component == "adjoint_even"
                    else "(A-A^dagger)/(2i)"
                ),
                "is_identity": representative == identity_permutation,
                "representative": list(representative),
                "twirl_orbit_size": len(orbit),
                "inverse_orbit_is_same": set(orbit) == set(inverse_orbit),
                "stored_dtype": str(matrix.dtype),
                "pre_normalization_rms_eigenvalue": pre_normalization_rms,
                "hermitian_max_error": hermitian_error,
                "pair_invariance_relative_error": pair_invariance_error,
                "w_commutator_relative_norm": w_commutator_relative,
            }
        )
        print(
            f"Saved {index + 1:3d}/{saved_target_rank}: {filename} "
            f"({component}, orbit={len(orbit):3d}, "
            f"W-break={w_commutator_relative > 1e-7})"
        )

    if w_breaking_count == 0:
        raise RuntimeError(
            "The selected basis contains no W-breaking direction, so it cannot "
            "distinguish reflections through this symmetry mechanism."
        )

    manifest = {
        "construction": [
            "H_plus=(A+A^dagger)/2",
            "H_minus=(A-A^dagger)/(2i)",
            "A=T_Spair[Pi(pi)]",
        ],
        "num_qubit": args.num_qubit,
        "num_pairs": num_pairs,
        "matrix_dimension": permutation_rows.dimension,
        "requested_real_precision": args.dtype,
        "stored_dtypes": {
            "adjoint_even": args.dtype,
            "adjoint_odd": (
                "complex64" if args.dtype == "float32" else "complex128"
            ),
        },
        "normalization": "sqrt(Tr(H^2)/matrix_dimension)=1",
        "full_hermitian_dimension": full_dimension,
        "adjoint_even_dimension": even_dimension,
        "adjoint_odd_dimension": odd_dimension,
        "identity_global_phase_removed": not include_identity,
        "num_generators": len(entries),
        "w_breaking_generators": w_breaking_count,
        "selection_seed": args.seed,
        "rank_verification_sigma_ratios": {
            "adjoint_even": even_verification_ratio,
            "adjoint_odd": odd_verification_ratio,
        },
        "restriction_multiplicities": multiplicities,
        "generators": entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Saved manifest: {manifest_path}")
    print(
        f"Done: {len(even_representatives)} adjoint-even + "
        f"{len(odd_representatives)} adjoint-odd = {len(entries)} "
        f"Hermitian generators; {w_breaking_count} do not commute with W."
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_qubit", type=int, required=True)
    args = parser.parse_args()

    if args.num_qubit != 10:
        raise ValueError("The reflection experiment requires --num_qubit 10.")

    args.output_dir = None
    args.seed = 260717
    args.dtype = "float32"
    args.sketch_probes = 3
    args.verify_probes = 5
    args.rank_tolerance = 1e-10
    args.max_attempts = 10000
    args.allow_larger = False

    generate(args)

if __name__ == "__main__":
    main()
