import pennylane as qml

def ops_to_mat(ops, wire_order):
    return qml.matrix(qml.tape.QuantumScript(ops), wire_order=wires)
