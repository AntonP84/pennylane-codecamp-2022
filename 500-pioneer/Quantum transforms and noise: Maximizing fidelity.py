import functools
import json
import math
import pandas as pd
import pennylane as qml
import pennylane.numpy as np
import scipy


@qml.qfunc_transform
def rotate_rots(tape, params):
    for op in tape.operations + tape.measurements:
        if op.name == "RX":
            if list(op.wires) == [0]:
                qml.RX(op.parameters[0] + params[0], wires=op.wires)
            else:
                qml.RX(op.parameters[0] + params[1], wires=op.wires)
        elif op.name == "RY":
            if list(op.wires) == [0]:
                qml.RY(op.parameters[0] + params[2], wires=op.wires)
            else:
                qml.RY(op.parameters[0] + params[3], wires=op.wires)
        elif op.name == "RZ":
            if list(op.wires) == [0]:
                qml.RZ(op.parameters[0] + params[4], wires=op.wires)
            else:
                qml.RZ(op.parameters[0] + params[5], wires=op.wires)
        else:
            qml.apply(op)


def circuit():
    """Available circuit which params you can tune"""
    for wire in [0, 1]:
        qml.RX(np.pi / 2, wires=wire)
        qml.RY(np.pi / 2, wires=wire)
        qml.RZ(np.pi / 2, wires=wire)


def optimal_fidelity(target_params: list[float], pauli_word: str) -> float:
    """This function returns the maximum fidelity between
    the final state that we obtain with only Pauli rotations
    with respect to the state we obtain with the target circuit.

    Args:
        - target_params (list(float)): List of the two parameters in the target circuit.
            The first is the parameter of the Pauli Rotation,
            the second is the parameter of the CRX gate.
        - pauli_word: A string that is either 'X', 'Y', or 'Z',
            depending on the Pauli rotation implemented by the target circuit.
    Returns:
        - (float): Maximum fidelity between the states produced by both circuits.
    """

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def target_circuit(target_params: list[float], pauli_word: str):
        """This QNode is target circuit whose effect we want to emulate"""
        alpha, beta = target_params
        word_to_op = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}

        word_to_op[pauli_word](alpha, wires=0)
        qml.CRX(beta, wires=[0, 1])
        qml.T(wires=0)
        qml.S(wires=1)

        return qml.state()

    @qml.qnode(dev)
    def rotated_circuit(rot_params: list[float]):
        """This QNode is the available circuit, with rotated parameters

        Args:
            - rot_params (list(float)): A list containing the values of the independent
                rotation parameters for each gate in the available circuit. The order
                will not matter, since you are optimizing for these and will return
                the minimal value of a cost function (related to the fidelity).
        """
        rotate_rots(rot_params)(circuit)()
        return qml.state()

    def calc_fidelity(rot_params):
        return qml.math.fidelity(
            rotated_circuit(rot_params), target_circuit(target_params, pauli_word)
        )

    steps = 1000
    eps = 1e-5
    np.random.seed(42)

    opt = qml.AdamOptimizer()
    params = np.random.random(6, requires_grad=True)

    cost_prev = 0.0
    for step in range(steps):
        params, cost = opt.step_and_cost(calc_fidelity, params)
        if abs(cost - cost_prev) < eps:
            break
        cost_prev = cost

    return cost


# These functions are responsible for testing the solution.


def run(test_case_input: str) -> str:

    ins = json.loads(test_case_input)
    output = optimal_fidelity(*ins)

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
    ), f"""Your calculated optimal fidelity isn't quite right.
    Expected:\n\t{expected_output:.3f}
    Got:\n\t{solution_output:.3f}"""


test_cases = [['[[1.6,0.9],"X"]', "0.9502"], ['[[0.4,0.5],"Y"]', "0.9977"]]

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
