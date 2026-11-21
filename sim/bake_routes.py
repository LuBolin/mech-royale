"""Bake walkable terrain, rooftops and bridge decks from the official kit arena."""
import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
from arena_routes import ATLAS, DIRECTIONS, STEP
from world import get_world


def bake(world):
    ground = world.ground
    xmin, xmax, ymin, ymax = world.bounds
    y, x = np.ogrid[:world.ny, :world.nx]
    inside = ((world.x0+x >= xmin+3) & (world.x0+x <= xmax-3) &
              (world.y0+y >= ymin+3) & (world.y0+y <= ymax-3))
    floors = (world.top, world.slab_hi)
    walk = []
    # The body's centre needs support. Its sides need clearance, so a narrow
    # bridge can support a mech even when the shoulders overhang the edges.
    for layer, floor in enumerate(floors):
        free = inside & ((~world.steep | (floor > ground + 0.3)) if layer == 0 else (floor > world.top))
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if dx*dx + dy*dy > 9:
                    continue
                def neighbor(grid):
                    return np.roll(grid, (dy, dx), axis=(0, 1))
                # Sloping terrain supports the body at its centre. Side checks
                # reject vertical walls and decks intersecting the occupied height.
                solid = neighbor(world.top)
                wall = (solid > neighbor(ground) + 0.3) & (solid > floor + 1.2)
                deck = (neighbor(world.slab_lo) < floor + 9) & (neighbor(world.slab_hi) > floor + 1.2)
                trunk = neighbor(world.tmask) & (neighbor(world.trunk) > floor)
                free &= ~(wall | deck | trunk)
        walk.append(free)
    rows, cols = np.mgrid[0:world.ny:STEP, 0:world.nx:STEP]
    shape = (2, *rows.shape)
    links = np.zeros(shape, np.uint8)
    upper = np.zeros_like(links)  # destination layer for each directional link
    cover = np.zeros_like(links)
    heights = np.stack([floor[rows, cols] for floor in floors])
    chest = heights + 7.5
    for bit, (dx, dy) in enumerate(DIRECTIONS):
        reachable = np.stack([mask[rows, cols] for mask in walk])
        prev = heights.copy()
        on_deck = np.zeros(shape, bool)
        for d in range(1, STEP + 1):
            i, j = cols + d*dx, rows + d*dy
            reachable &= (i >= 0) & (i < world.nx) & (j >= 0) & (j < world.ny)
            i, j = i.clip(0, world.nx-1), j.clip(0, world.ny-1)
            low, high = [floor[j, i] for floor in floors]
            onto_low = walk[0][j, i] & (np.abs(low - prev) <= 1.2)
            onto_high = walk[1][j, i] & (np.abs(high - prev) <= 1.2)
            on_deck = onto_high & (~onto_low | (np.abs(high-prev) < np.abs(low-prev)))
            reachable &= onto_low | onto_high
            prev = np.where(on_deck, high, low)
        links[reachable] |= 1 << bit
        upper[reachable & on_deck] |= 1 << bit
        for d in range(2, int(32 / math.hypot(dx, dy)) + 1):
            i = (cols + d*dx).clip(0, world.nx-1)
            j = (rows + d*dy).clip(0, world.ny-1)
            blocked = (world.top[j,i] >= chest) | ((world.slab_lo[j,i] <= chest) & (world.slab_hi[j,i] >= chest))
            cover[blocked] |= 1 << bit
    # A short skybridge may have one valid landing pad at 4 m resolution
    # and no walking edge. Preserve its real height for airborne access.
    pads = links != 0
    pads[1] |= walk[1][rows, cols]
    cover[~pads] = 0
    heights[~pads] = 0
    heights_cm = np.rint(100 * heights).astype(np.int16)
    return links, upper, cover, heights_cm, pads


def flight_candidates(world, pads, heights_cm, stats):
    """Approaches from a lower surface to a nearby roof/deck. The flight model
    validates each direction independently; a valid descent needn't work uphill.
    Keep a few alternatives per side of each ledge, denser on narrow bridge decks.
    """
    _, rows, cols = pads.shape
    layer_size = rows * cols
    heights = heights_cm.astype(float) / 100
    terrain = world.ground[::STEP, ::STEP]
    overhead = world.slab_lo[::STEP, ::STEP]
    groups = {}
    for high_layer in (0, 1):
        elevated = pads[high_layer] & (heights[high_layer] > terrain + stats["step_up"])
        hr, hc = np.nonzero(elevated)
        high_z = heights[high_layer, hr, hc]
        for direction, (dx, dy) in enumerate(DIRECTIONS):
            for stride in (2, 3, 4, 5, 6):
                dist = math.hypot(dx, dy) * STEP * stride
                if dist > 26:
                    continue
                lr, lc = hr + stride*dy, hc + stride*dx
                bounded = (lr >= 0) & (lr < rows) & (lc >= 0) & (lc < cols)
                lr = lr.clip(0, rows-1)
                lc = lc.clip(0, cols-1)
                for low_layer in (0, 1):
                    low_z = heights[low_layer, lr, lc]
                    dz = high_z - low_z
                    viable = bounded & pads[low_layer, lr, lc] & (dz > stats["step_up"] + 0.25)
                    if low_layer == 0:
                        # A takeoff beneath the destination bridge cannot climb through it.
                        viable &= overhead[lr, lc] > high_z + stats["height"]
                    # Allow sparse routes between distinct roofs too. The narrow deck
                    # cells retain independent approaches even when their heights match.
                    tile = 3 if high_layer else 8
                    for i in np.flatnonzero(viable):
                        hi = int(high_layer*layer_size + hr[i]*cols + hc[i])
                        lo = int(low_layer*layer_size + lr[i]*cols + lc[i])
                        key = (high_layer, int(hr[i]//tile), int(hc[i]//tile),
                               int(round(high_z[i])), low_layer, int(round(low_z[i]/8)),
                               direction if high_layer else direction//2)
                        preferred = 12.0
                        rank = (abs(dist-preferred), dist, hi, lo)
                        pool = groups.setdefault(key, [])
                        if len(pool) < 3 or rank < pool[-1][0]:
                            pool.append((rank, lo, hi))
                            pool.sort()
                            del pool[3:]
    return groups


def bake_flights(world, pads, heights_cm, stats):
    from flight import plan_flight
    layer_size, cols = pads[0].size, pads.shape[2]
    height = heights_cm.ravel()
    def point(n):
        row, col = divmod(n % layer_size, cols)
        return world.x0 + STEP*col, world.y0 + STEP*row, float(height[n]) / 100

    accepted = {}
    groups = flight_candidates(world, pads, heights_cm, stats)
    for pool in groups.values():
        for reverse in (False, True):
            for _, lo, hi in pool:
                edge = (hi, lo) if reverse else (lo, hi)
                if edge in accepted:
                    break
                flight = plan_flight(world, point(edge[0]), point(edge[1]), stats, stats["energy"])
                if flight is not None:
                    accepted[edge] = (flight.energy_cost, flight.airtime)
                    break
    edges = sorted(accepted)
    costs = np.array([accepted[e] for e in edges], dtype=np.float32).reshape(-1, 2)
    return np.array(edges, dtype=np.int32).reshape(-1, 2), costs[:,0], costs[:,1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena", type=Path, default=ROOT / "kit/arena")
    args = parser.parse_args()
    world = get_world(args.arena)
    links, upper, cover, heights_cm, pads = bake(world)
    with gzip.open(args.arena.parent / "check/stream.jsonl.gz", "rt") as f:
        stats = json.loads(next(f))["config"]["mech"]
    air_edges, air_energy, air_time = bake_flights(world, pads, heights_cm, stats)
    np.savez_compressed(ATLAS, links=links, upper=upper, cover=cover, heights_cm=heights_cm,
                        air_edges=air_edges, air_energy=air_energy, air_time=air_time,
                        nominal_jet_drain=stats["jet_drain"], map_digest=world.map_digest)
    print(f"{np.count_nonzero(links)} linked nodes, {len(air_edges)} directed flights, "
          f"{np.count_nonzero((links[0] != 0) & (links[1] != 0))} overlapping cells, {ATLAS.stat().st_size} bytes")
