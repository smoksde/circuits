from flask import Flask, render_template, jsonify, request
import json

import circuits.circuit as circuit
import graph
from test import run_tests

app = Flask(__name__)


@app.route("/run-tests")
def run_tests_route():
    result = run_tests()
    return jsonify(json.loads(result))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get-circuits", methods=["GET"])
def get_circuits():
    return jsonify({"circuits": list(circuit.CIRCUIT_FUNCTIONS.keys())})


@app.route("/get-circuit", methods=["POST"])
def get_circuit():
    data = request.get_json()
    circuit_name = data.get("circuit")
    if circuit_name not in circuit.CIRCUIT_FUNCTIONS:
        return jsonify({"error": "Invalid circuit name"}), 400
    cg = graph.CircuitGraph()
    bit_len = 4
    circuit_func = circuit.CIRCUIT_FUNCTIONS.get(circuit_name)
    if circuit_func:
        circuit_func(cg, bit_len)
        try:
            data = cg.to_json()
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": f"Error serializing circuit: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid circuit type"}), 400


if __name__ == "__main__":
    app.run(debug=True)
