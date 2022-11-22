import functools
import json
import math
import pandas as pd
import pennylane as qml
import pennylane.numpy as np
import scipy


def circuit():
    """Succession of gates that will generate the requested matrix."""

    def H_via_U3(wire):
        qml.U3(theta=np.pi / 2, phi=0, delta=np.pi, wires=wire)

    def Z_via_U3(wire):
        qml.U3(theta=0, phi=0, delta=np.pi, wires=wire)

    H_via_U3(wire=1)
    qml.CNOT(wires=[0, 1])
    H_via_U3(wire=1)
    Z_via_U3(wire=0)
    H_via_U3(wire=2)

    # implemented as:
    # qml.Hadamard(1)
    # qml.CNOT(wires=[0,1])
    # qml.Hadamard(1)
    # qml.PauliZ(0)
    # qml.Hadamard(2)

    # another option:
    # qml.CZ(wires=[0,1])
    # qml.PauliZ(1)
    # qml.Hadamard(2)


# These functions are responsible for testing the solution.


def run(input: str) -> str:
    matrix = qml.matrix(circuit)().real

    with qml.tape.QuantumTape() as tape:
        circuit()

    names = [op.name for op in tape.operations]
    return json.dumps({"matrix": matrix.tolist(), "gates": names})


def check(user_output: str, expected_output: str) -> str:
    parsed_output = json.loads(user_output)
    matrix_user = np.array(parsed_output["matrix"])
    gates = parsed_output["gates"]

    solution = (
        1
        / np.sqrt(2)
        * np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, -1, 0, 0, 0, 0, 0, 0],
                [0, 0, -1, -1, 0, 0, 0, 0],
                [0, 0, -1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, -1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, -1],
            ]
        )
    )

    assert np.allclose(
        matrix_user, solution
    ), f"Expected:\n{solution.round(2)}\n\nGot:\n{matrix_user.round(2)}"
    assert len(set(gates)) == 2 and "U3" in gates and "CNOT" in gates


test_cases = [["No input", "No output"]]

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
