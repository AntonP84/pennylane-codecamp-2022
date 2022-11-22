"""https://pennylane.ai/qml/glossary/parameter_shift.html"""

import functools
import json
import math
import pandas as pd
import pennylane as qml
import pennylane.numpy as np
import scipy

dev = qml.device("default.qubit", wires=3)


@qml.qnode(dev)
def circuit(params: list[float]):
    """The quantum circuit that you will differentiate!

    Args:
        params (list(float)): The parameters for gates in the circuit

    Returns:
        (numpy.array): An expectation value.
    """
    qml.broadcast(qml.Hadamard, wires=range(3), pattern="single")
    qml.CRX(params[0], [1, 2])
    qml.CRY(params[1], [0, 1])
    qml.CRZ(params[2], [2, 0])
    return qml.expval(qml.PauliZ(0) + qml.PauliZ(1) + qml.PauliX(2))


def shifts_and_coeffs():
    """A function that defines the shift amounts and coefficients needed for
    defining a parameter-shift rule for CRX, CRY, and CRZ gates.

    Note:
    - the multiplier c and the shift s are determined completely by the type of transformation
    -  is finite rather than infinitesimal
    - formulas from https://docs.pennylane.ai/en/stable/code/api/pennylane.CRX.html

    Returns:
        shifts (list(float)): A list of shift amounts. Order them however you want!
        coeffs (list(float)): A list of coefficients. Order them however you want!
    """
    s1, s2 = math.pi / 2, 3 * math.pi / 2
    c1, c2 = [(math.sqrt(2) + v) / 4 / math.sqrt(2) for v in [+1, -1]]

    shifts = s1, s2
    coeffs = c1, c2
    return shifts, coeffs


def my_parameter_shift_grad(params: list[float]):
    """Your homemade parameter-shift rule function!
    NOTE: you cannot use qml.grad within this function

    Args:
        params (list(float)): The parameters for gates in the circuit

    Returns:
        gradient (numpy.array): The gradient of the circuit with respect to the given parameters.
    """

    def _shift_params(params: list[float], i, shift):
        params = list(params)

        ret = params[:i]
        ret.append(params[i] + shift)
        ret.extend(params[(i + 1) :])
        return ret

    gradient = np.zeros_like(params)

    shifts, coeffs = shifts_and_coeffs()
    s1, s2 = shifts
    c1, c2 = coeffs

    for i in range(len(params)):
        partial = c1 * (
            circuit(_shift_params(params, i, s1))
            - circuit(_shift_params(params, i, -s1))
        ) - c2 * (
            circuit(_shift_params(params, i, s2))
            - circuit(_shift_params(params, i, -s2))
        )
        gradient[i] = partial

    return np.round_(gradient, decimals=5).tolist()


# These functions are responsible for testing the solution.


def run(test_case_input: str) -> str:
    params = json.loads(test_case_input)
    gradient = my_parameter_shift_grad(params)
    return str(gradient)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-4
    ), "Your gradient isn't quite right!"


test_cases = [["[1.23, 0.6, 4.56]", "[0.08144, -0.33706, -0.37944]"]]

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
