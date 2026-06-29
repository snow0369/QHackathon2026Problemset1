from __future__ import annotations

import numpy as np


def single_one_evolution_meas_Z0(
    time: float,
    *,
    num_qubits: int = 12,
    j_x: float = 1.0,
    j_y: float = 1.0,
    j_z: float = 1.2,
    h_x: float = 0.0,
    initial_excitation: int = 0,
    periodic: bool = True,
) -> float:
    r"""Compute the exact value of <Z_0> in the one-excitation subspace.

    The Hamiltonian convention is

        H = sum_{(i,j)} (
                j_x X_i X_j
              + j_y Y_i Y_j
              + j_z Z_i Z_j
            )
            + h_x sum_i X_i.

    This utility avoids constructing the full 2**num_qubits statevector.
    It works only in the one-excitation subspace, whose dimension is
    num_qubits.

    Args:
        time: Evolution time.
        num_qubits: Number of qubits.
        j_x: XX coupling coefficient.
        j_y: YY coupling coefficient.
        j_z: ZZ coupling coefficient.
        h_x: Transverse-field coefficient. Must be zero.
        initial_excitation: Initially excited qubit index.
            For |1000...000>, use initial_excitation=0.
        periodic: Whether to include the edge (num_qubits - 1, 0).

    Returns:
        Exact expectation value <Z_0>.
    """
    if num_qubits < 2:
        raise ValueError("num_qubits must be at least 2.")

    if periodic and num_qubits < 3:
        raise ValueError("A periodic chain requires num_qubits >= 3.")

    if not 0 <= initial_excitation < num_qubits:
        raise ValueError(
            f"initial_excitation must be in [0, {num_qubits - 1}]."
        )

    if not np.isclose(j_x, j_y):
        raise ValueError(
            "This utility assumes j_x == j_y, so that excitation number "
            "is conserved."
        )

    if not np.isclose(h_x, 0.0):
        raise ValueError(
            "This utility assumes h_x == 0, so that excitation number "
            "is conserved."
        )

    if periodic:
        edges = [(i, (i + 1) % num_qubits) for i in range(num_qubits)]
    else:
        edges = [(i, i + 1) for i in range(num_qubits - 1)]

    # Reduced Hamiltonian in the one-excitation basis.
    #
    # basis[q] means:
    #   qubit q is in |1>, and all other qubits are in |0>.
    h_eff = np.zeros((num_qubits, num_qubits), dtype=complex)

    for i, j in edges:
        # XX + YY moves the single excitation between neighboring sites:
        #
        #   (j_x X_i X_j + j_y Y_i Y_j) |...10...>
        #       = (j_x + j_y) |...01...>
        #
        hopping = j_x + j_y
        h_eff[i, j] += hopping
        h_eff[j, i] += hopping

        # ZZ is diagonal in the computational basis.
        # In the one-excitation basis, Z_i Z_j gives:
        #   -1 if the excitation is on i or j,
        #   +1 otherwise.
        h_eff[np.diag_indices(num_qubits)] += j_z
        h_eff[i, i] -= 2.0 * j_z
        h_eff[j, j] -= 2.0 * j_z

    psi0 = np.zeros(num_qubits, dtype=complex)
    psi0[initial_excitation] = 1.0

    eigvals, eigvecs = np.linalg.eigh(h_eff)

    amplitudes = eigvecs @ (
        np.exp(-1j * time * eigvals)
        * (eigvecs.conj().T @ psi0)
    )

    prob_qubit_0_is_one = np.abs(amplitudes[0]) ** 2

    # Z_0 gives -1 if qubit 0 is |1>, and +1 otherwise.
    z0 = 1.0 - 2.0 * prob_qubit_0_is_one

    return float(np.real_if_close(z0))


if __name__ == "__main__": # Usage example
    z0_exact = single_one_evolution_meas_Z0(
        1.0,
        num_qubits=12,
        j_x=1.0,
        j_y=1.0,
        j_z=1.2,
        h_x=0.0,
        initial_excitation=0,
        periodic=True,
    )

    print(f"Exact <Z_0> = {z0_exact:.8f}")