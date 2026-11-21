"""Extract sim inputs from the kit: env config (from the recorded hello) and a map crop.

Usage: python sim/extract.py [minX minY maxX maxY]   (default crop: 200x200 m around the zone start center)
Writes sim/env.json and sim/map_crop.json. Stdlib only.
The live site (play.joinbuilders.org/arena/v2/*) serves byte-identical map files to kit/arena/v2 (checked 2026-11-21).
"""
import gzip, json, struct, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT, OUT = ROOT / "kit", ROOT / "sim"


def load_config():
    with gzip.open(KIT / "check" / "stream.jsonl.gz", "rt") as f:
        hello = json.loads(f.readline())
    assert hello["type"] == "hello", "first stream line must be hello"
    return hello["config"]


def zone_timeline(z):
    # Absolute times per phase; next centers are random per match, so only radii/times are fixed.
    t, r, out = 0.0, z["start_radius"], []
    for p in z["phases"]:
        out.append({"hold_from": t, "shrink_start": t + p["hold_s"], "shrink_end": t + p["hold_s"] + p["shrink_s"],
                    "radius_from": r, "radius_to": p["radius"], "dps": p["dps"]})
        t, r = out[-1]["shrink_end"], p["radius"]
    return out


def crop_map(bbox):
    minX, minY, maxX, maxY = bbox
    world = json.loads((KIT / "arena" / "v2" / "world.json").read_text())
    T = world["terrain"]
    raw = (KIT / "arena" / "v2" / T["file"]).read_bytes()
    assert len(raw) == T["nx"] * T["ny"] * 2, "terrain size mismatch"
    i0, i1 = max(0, int(minX - T["minX"])), min(T["nx"] - 1, int(maxX - T["minX"]))
    j0, j1 = max(0, int(minY - T["minY"])), min(T["ny"] - 1, int(maxY - T["minY"]))
    rows = []
    for j in range(j0, j1 + 1):
        off = (j * T["nx"] + i0) * 2
        rows.append([round(h * T["scale"], 2) for h in struct.unpack_from(f"<{i1 - i0 + 1}h", raw, off)])

    def overlaps(ring):
        xs, ys = ring[0::2], ring[1::2]
        return min(xs) <= maxX and max(xs) >= minX and min(ys) <= maxY and max(ys) >= minY

    tr = world["trees"]
    nf, fx, fy = len(tr["fields"]), tr["fields"].index("x"), tr["fields"].index("y")
    trees = [tr["data"][k:k + nf] for k in range(0, len(tr["data"]), nf)
             if minX <= tr["data"][k + fx] <= maxX and minY <= tr["data"][k + fy] <= maxY]
    return {
        "frame": world["frame"], "bbox": [minX, minY, maxX, maxY], "rules": world["rules"],
        "terrain": {"minX": T["minX"] + i0 * T["step"], "minY": T["minY"] + j0 * T["step"], "step": T["step"],
                    "nx": i1 - i0 + 1, "ny": j1 - j0 + 1, "heights_m": rows, "layout": "heights_m[j][i], y = minY + j*step"},
        "prisms": [p for p in world["prisms"] if overlaps(p["ring"])],
        "trees": {"fields": tr["fields"], "rows": trees},
    }


if __name__ == "__main__":
    cfg = load_config()
    cx, cy = cfg["zone"]["start_center"]
    bbox = [float(v) for v in sys.argv[1:5]] if len(sys.argv) >= 5 else [cx - 100, cy - 100, cx + 100, cy + 100]
    env = {"source": "kit/check/stream.jsonl.gz hello.config", "config": cfg,
           "derived": {"zone_timeline": zone_timeline(cfg["zone"]),
                       "dt_tick": 1 / cfg["match"]["tick_hz"],
                       "dt_decision": cfg["match"]["decide_every"] / cfg["match"]["tick_hz"]}}
    tl = env["derived"]["zone_timeline"]
    assert tl[-1]["shrink_end"] == 415 and tl[-1]["radius_to"] == 0, tl[-1]  # protocol.md §9: closes ~6:55
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    m = crop_map(bbox)
    (OUT / "map_crop.json").write_text(json.dumps(m))
    print(f"env.json written; crop {m['terrain']['nx']}x{m['terrain']['ny']} m, "
          f"{len(m['prisms'])} prisms, {len(m['trees']['rows'])} trees")
