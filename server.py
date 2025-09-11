from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import importlib, io
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from map import *

from core.graph import CircuitGraph
from core.interface import *
import measurement
import circuits.circuit as circuit

app = FastAPI()


@app.get("/funcs")
def funcs():
    return JSONResponse(
        [
            name
            for name in dir(circuit)
            if name.startswith("setup_") and callable(getattr(circuit, name))
        ]
    )


@app.get("/metrics")
def get_metrics():
    return JSONResponse(measurement.metrics)


@app.get("/plot_types")
def get_plot_types():
    return JSONResponse(measurement.plot_types)


@app.get("/interfaces")
def get_interfaces():
    return JSONResponse(measurement.interfaces)


@app.post("/treemap")
async def generate_treemap(req: Request):
    try:
        j = await req.json()
        fn_name = j.get("fn")
        bit_width = j.get("bit_width", 8)

        if not fn_name:
            raise HTTPException(400, "Function name is required")

        setup_fn = getattr(circuit, fn_name, None)
        if not callable(setup_fn):
            raise HTTPException(400, f"No such function: {fn_name}")

        circuit_graph = CircuitGraph()
        interface = GraphInterface(circuit_graph)

        try:
            setup_fn(interface, bit_len=bit_width)
        except TypeError:
            setup_fn(circuit_graph, bit_len=bit_width)

        fig = build_area_treemap(circuit_graph.groups, circuit_graph.nodes)
        fig_json = fig.to_json()

        return JSONResponse(json.loads(fig_json))

    except Exception as e:
        raise HTTPException(500, f"Error generating treemap: {str(e)}")


def render_plot(specs, metric="depth", interface="graph", title="Circuit Metrics"):
    plt.figure(figsize=(6, 3.5))
    for s in specs:
        fn_name = s["fn"]
        bits = s.get("bits", [4, 8, 16])
        setup_fn = getattr(circuit, fn_name, None)
        if not callable(setup_fn):
            raise HTTPException(400, f"no function {fn_name}")
        vals = [
            measurement.analyze_circuit_function(
                fn_name,
                setup_fn,
                b,
                interface_name=interface,
                metric=metric,
                use_cache=True,
                fill_cache=True,
            )[metric]
            for b in bits
        ]

        label = (
            fn_name[6:] if fn_name.startswith("setup_") else fn_name
        )  # or s.get("name", fn_name)
        plt.plot(bits, vals, marker="o", linestyle="--", label=label)
    plt.title(title)
    plt.xlabel("bits")
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend(fontsize=8)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=120)
    plt.close()
    buf.seek(0)
    return buf


def render_function_approximation_plot(
    specs,
    metric="depth",
    interface="graph",
    title="Measurements for different bit widths",
):
    plt.figure(figsize=(6, 3.5))
    for s in specs:
        fn_name = s["fn"]
        bits = s.get("bits", [4, 8, 16])
        setup_fn = getattr(circuit, fn_name, None)
        if not callable(setup_fn):
            raise HTTPException(400, f"no function {fn_name}")
        vals = [
            measurement.analyze_circuit_function(
                fn_name,
                setup_fn,
                b,
                interface_name=interface,
                use_cache=True,
                fill_cache=True,
            )[metric]
            for b in bits
        ]

        label = (
            fn_name[6:] if fn_name.startswith("setup_") else fn_name
        )  # or s.get("name", fn_name)
        plt.plot(bits, vals, marker="o", linestyle="--", label="measured")

    plt.title(label)
    plt.xlabel("bits")
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=120)
    plt.close()
    buf.seek(0)
    return buf


@app.post("/plot.png")
async def plot_png(req: Request):
    j = await req.json()
    specs = j.get("experiments") or j.get("specs")
    plot_type = j.get("plot_type", "general")
    metric = j.get("metric", "depth")
    interface = j.get("interface", "graph")
    # print("metric: ", metric)
    if not specs:
        raise HTTPException(400, "No experiments provided")

    # Map function names to actual Python functions
    for s in specs:
        fn_name = s["fn"]
        setup_fn = getattr(circuit, fn_name, None)
        if not callable(setup_fn):
            raise HTTPException(400, f"No such function: {fn_name}")
        s["setup_fn"] = setup_fn
        s.setdefault("name", fn_name)

    if plot_type == "general":
        buf = render_plot(specs, metric=metric, interface=interface)
    elif plot_type == "function_approximation":
        buf = render_function_approximation_plot(
            specs, metric=metric, interface=interface
        )
    else:
        buf = render_plot(specs, metric=metric, interface=interface)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(open("index.html", "r", encoding="utf-8").read())
