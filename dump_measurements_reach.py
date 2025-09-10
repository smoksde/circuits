from pathlib import Path
import cache

BASE_DIR = Path(__file__).resolve().parent
CACHE_FILE = BASE_DIR / "measurements_cache.json"

data = cache.load(CACHE_FILE)

for circuit, configs in data.items():
    summary = []
    for bitwidth in sorted(configs.keys(), key=lambda x: int(x)):
        stats = configs[bitwidth]
        if set(stats.keys()) == {"depth"}:
            summary.append(f"{bitwidth}(depth only)")
        else:
            summary.append(f"{bitwidth}(full)")
    print(f"{circuit}: {', '.join(summary)}")
