"""https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series.html"""

import functools
import json
import math
import pandas as pd
import pennylane as qml
import pennylane.numpy as np
import scipy


def fourier_squared_distance(
    list_of_coeffs: list[float], param_list: list[float]
) -> float:
    """
    Returns the squared l2-distance in Fourier space between a function
    characterized by its Fourier coefficients and the output of the
    quantum model

    Args:
        list_of coeffs (list(float)): A list of seven coefficients
                                      corresponding to the Fourier
                                      coefficients of the function we
                                      want to approximate
        param_list (list(float)): A list of six parameters characterizing
                                  the angles in the trainable circuit.

    Returns: (float): Squared l2-distance between the given function
                      and the output of the quantum model
    """

    dev = qml.device("default.qubit", wires=3)

    def S(x):
        """Data-encoding circuit block."""
        for w in range(dev.num_wires):
            qml.RX(x, wires=w)

    def W(theta):
        """Trainable circuit block."""
        qml.BasicEntanglerLayers(theta, wires=dev.wires)

    @qml.qnode(dev)
    def quantum_model(param_list, x):
        """This circuit returns the PauliZ expectation of
        the quantum model in the statement"""
        n_wires = dev.num_wires
        weights_1 = [param_list[:n_wires]]
        weights_2 = [param_list[-n_wires:]]

        W(weights_1)
        S(x)
        W(weights_2)
        return qml.expval(qml.PauliZ(wires=0))

    def fourier_coefficients(f: callable, K: int):
        """Computes the first 2*K+1 Fourier coefficients of a 2*pi periodic function."""
        n_coeffs = 2 * K + 1
        t = np.linspace(0, 2 * np.pi, n_coeffs, endpoint=False)
        y = np.fft.rfft(f(t)) / t.size
        return y

    def f(x):
        return np.array([quantum_model(param_list, x_) for x_ in x])

    coeffs_sample = fourier_coefficients(f, K=3)

    list_freqs = [0, 1, 2, 3, -3, -2, -1]
    distance = np.linalg.norm(
        [c - coeffs_sample[i] for i, c in zip(list_freqs, list_of_coeffs)]
    )

    return distance**2


# These functions are responsible for testing the solution.


def run(test_case_input: str) -> str:

    ins = json.loads(test_case_input)
    output = fourier_squared_distance(*ins)

    return str(output)


def check(solution_output: str, expected_output: str) -> None:
    """
    Compare solution with expected.

    Args:
            solution_output: The output from an evaluated solution. Will be
            the same type as returned.
            expected_output: The correct result for the test case.

    Raises:
            ``AssertionError`` if the solution output is incorrect in any way.
    """

    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-2
    ), f"Your calculated squared distance isn't quite right. Expected: {expected_output:.2f}, got {solution_output:.2f}"


test_cases = [
    [
        "[[-1.12422548e-01,0.0,9.47909940e-02,0.0,0.0,9.47909940e-02,0.0],[2,2,2,3,4,5]]",
        "0.0036766085933034303",
    ],
    [
        "[[-2.51161988e-01,0.0,1.22546112e-01,0.0, 0.0,1.22546112e-01, 0.0],[1.1,0.3,0.4,0.6,0.8,0.9]]",
        "0.6538589174369286",
    ],
]

for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")
