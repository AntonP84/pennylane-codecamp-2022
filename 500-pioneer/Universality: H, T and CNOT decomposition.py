import functools
import json
import math
import pandas as pd
import pennylane as qml
import pennylane.numpy as np
import scipy


def U():
    """
    This quantum function will simply contain H, T and CNOT gates.
    It will not return any value.
    """

    # # sqrt(CNOT_{12})
    # qml.Hadamard(wires=1)
    # qml.ctrl(qml.T, control=0)(wires=1)
    # qml.ctrl(qml.T, control=0)(wires=1)
    # qml.Hadamard(wires=1)

    # sqrt(CNOT_{12})
    # https://quantumcomputing.stackexchange.com/questions/8274/how-to-implement-the-controlled-square-root-of-not-gate-on-the-ibm-q-composer
    qml.Hadamard(wires=1)
    qml.T(wires=0)
    qml.T(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.T(wires=1)
    qml.T(wires=1)
    qml.T(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=1)

    qml.CNOT(wires=[1, 0])


dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev)
def circuit():
    """Main circuit given in the statement.

    Here the operators you have defined in U will be embedded.
    """

    qml.CNOT(wires=[0, 1])

    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)

    U()

    qml.CNOT(wires=[1, 0])

    U()

    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)

    return qml.state()


# These functions are responsible for testing the solution.


def run(input: str) -> str:
    matrix = qml.matrix(circuit)().real

    with qml.tape.QuantumTape() as tape:
        U()

    names = [op.name for op in tape.operations]
    return json.dumps({"matrix": matrix.tolist(), "gates": names})


def check(user_output: str, expected_output: str) -> str:

    parsed_output = json.loads(user_output)
    matrix_user = np.array(parsed_output["matrix"])
    gates = parsed_output["gates"]

    solution = qml.matrix(qml.SWAP)(wires=[0, 1])

    assert np.allclose(
        matrix_user, solution
    ), f"Expected:\n{solution.round(2)}\n\nGot:\n{matrix_user.round(2)}"
    assert (
        len(set(gates)) == 3
        and "Hadamard" in gates
        and "CNOT" in gates
        and "T" in gates
    ), f"Gates found: {gates}"


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
