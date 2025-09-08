import json

with open("measurements_cache.json", "r") as f:
    data = json.load(f)

for circuit_name, circuits in data.items():
    print(f"\n=== {circuit_name} ===")
    for bit_width in sorted(circuits, key=lambda x: int(x)):
        c = circuits[bit_width]
        print(f"bit_len={c['bit_len']:>5} | depth={c['depth']:>5} | num_nodes={c['num_nodes']}")
