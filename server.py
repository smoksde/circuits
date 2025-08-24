from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import importlib, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, deque

from graph import CircuitGraph
import metrics
import circuits.circuit as circuit

app = FastAPI()

@app.get("/funcs")
def funcs():
    return JSONResponse([name for name in dir(circuit) if name.startswith("setup_") and callable(getattr(circuit, name))])

@app.get("/metrics")
def get_metrics():
    return JSONResponse(metrics.metrics)

def render_plot(specs, metric="depth", title="Metric vs bits"):
    plt.figure(figsize=(6,3.5))
    for s in specs:
        fn_name = s["fn"]
        bits = s.get("bits", [4,8,16])
        setup_fn = getattr(circuit, fn_name, None)
        if not callable(setup_fn): raise HTTPException(400, f"no function {fn_name}")
        vals = [ metrics.analyze_circuit_function(fn_name, setup_fn, b)[metric] for b in bits ]
        label = (fn_name[6:] if fn_name.startswith("setup_") else fn_name) # or s.get("name", fn_name)
        plt.plot(bits, vals, marker="o", label=label)
    plt.title(title); plt.xlabel("bits"); plt.ylabel(metric); plt.grid(True); plt.legend()
    buf = io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format="png", dpi=120); plt.close(); buf.seek(0)
    return buf

@app.post("/plot.png")
async def plot_png(req: Request):
    j = await req.json()
    specs = j.get("experiments") or j.get("specs")
    metric = j.get("metric", "depth")
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

    buf = render_plot(specs, metric=metric, title=j.get("title", "Circuit Metrics"))
    return StreamingResponse(buf, media_type="image/png")

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(open("index.html","r").read())