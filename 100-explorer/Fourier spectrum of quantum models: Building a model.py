import functools
import json
import math
import pandas as pd
import pennylane as qml
import pennylane.numpy as np
import scipy


def quantum_model(
    n: int, train_params: list[float], x: float, model_type: str
) -> float:
    """
    Builds an in-series or parallel quantum model according to the specifications in the problem statement, returning
    the expectation value on the first wire.

    Args:
        n (int): The total number of encoding gates. Assumed to be smaller than,
                or equal to, the number of wires (3).
        train_params (list(float)): A list of length n + 1, which indicates the rotation parameter characterizing
                    the trainable circuit. If the circuit is parallel, only the first two parameters are used.
        x (float): A real number representing the input data point.
        model_type (str): A string that is "series" or "parallel", depending on the type of model.

    Returns:
        (float): The expectation value of PauliZ measurements on the first wire.
    """

    num_wires = 3

    dev = qml.device("default.qubit", wires=num_wires)

    def trainable_circuit(theta):
        qml.Hadamard(wires=0)
        qml.CRX(theta, wires=[0, 1])
        qml.Hadamard(wires=1)
        qml.CRX(theta, wires=[1, 2])
        qml.Hadamard(wires=2)
        qml.CRX(theta, wires=[2, 0])

    def encoding_circuit_serial(x):
        qml.RX(x, wires=0)

    def encoding_circuit_parallel(x):
        for i in range(num_wires):
            qml.RX(x, wires=i)

    @qml.qnode(dev)
    def circuit(n, train_params, x, model_type):
        trainable_circuit(train_params[0])
        if model_type == "parallel":
            encoding_circuit_parallel(x)
            trainable_circuit(train_params[1])
        elif model_type == "series":
            for theta in train_params[1:]:
                encoding_circuit_serial(x)
                trainable_circuit(theta)
        return qml.expval(qml.PauliZ(0))

    return circuit(n, train_params, x, model_type)


# These functions are responsible for testing the solution.


def run(test_case_input: str) -> str:

    ins = json.loads(test_case_input)
    output = quantum_model(*ins)

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
    ), "Your circuit doesn't look quite right."


test_cases = [
    ['[2,[0.4,0.8,1.0],1.2,"parallel"]', "0.977856732420062"],
    ['[3,[0.6,0.7,0.8,1.2],1.3,"series"]', "0.37048658401504975"],
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
