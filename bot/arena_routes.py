"""Surface routes on the fixed arena. Vertically stacked nodes have independent walk links and cover."""
import heapq
import math
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

STEP = 4
DIRECTIONS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
ATLAS = Path(__file__).with_name("arena_routes.npz")


class Waypoint(NamedTuple):
    x: float
    y: float
    z: float
    air: bool = False  # fly here from the preceding waypoint


@lru_cache(maxsize=2)
def load_routes(world):
    if not ATLAS.is_file():
        return None
    with np.load(ATLAS, allow_pickle=False) as data:
        if str(data["map_digest"]) != world.map_digest:
            return None
        return Routes(world, data["links"], data["upper"], data["cover"], data["heights_cm"],
                      data["air_edges"], data["air_time"], data["air_energy"], float(data["nominal_jet_drain"]))


class Routes:
    def __init__(self, world, links, upper, cover, heights_cm, air_edges, air_time, air_energy, nominal_jet_drain):
        self.world = world
        self.width = links.shape[2]
        self.layer_size = links[0].size
        self.links = links.tobytes()
        self.upper = upper.tobytes()
        self.cover = cover.tobytes()
        self.heights = (heights_cm.ravel().astype(float) * 0.01).tolist()
        self.edges = [(dx + dy * self.width, STEP * math.hypot(dx, dy)) for dx, dy in DIRECTIONS]
        self.flights = {}
        self.landing_nodes = set()
        for (source, dest), seconds, energy in zip(air_edges.tolist(), air_time.tolist(), air_energy.tolist()):
            # Burn is quantized at the simulator's 20 Hz physics rate. Keep duration
            # so upgraded jets are priced at the drain in the current observation.
            burn_ticks = round(energy / (0.05 * nominal_jet_drain))
            self.flights.setdefault(source, []).append((dest, seconds, burn_ticks))
            self.landing_nodes.add(dest)

    def point(self, node):
        row, col = divmod(node % self.layer_size, self.width)
        return self.world.x0 + STEP * col, self.world.y0 + STEP * row

    def plan(self, pos, goal, purpose, threats, circle, stats, energy=0, allow_air=True):
        """Bounded A* for travel, or a local walking-distance search for shelter and firing angles."""
        x, y, z = pos
        w = self.world
        col, row = round((x - w.x0) / STEP), round((y - w.y0) / STEP)
        anchors = []
        for dx, dy in ((0, 0), *DIRECTIONS):
            i, j = col + dx, row + dy
            n = j * self.width + i
            if 0 <= i < self.width and 0 <= n < self.layer_size:
                for node in (n, n+self.layer_size):
                    if self.links[node] or node in self.flights or node in self.landing_nodes:
                        px, py = self.point(node)
                        distance = math.hypot(px-x, py-y)
                        if abs(z-self.heights[node]) <= distance + stats["step_up"]:
                            anchors.append((distance, node, px, py))
        for distance, start, px, py in sorted(anchors):
            if distance < 0.1:
                if abs(z-self.heights[start]) <= stats["step_up"]:
                    break
                continue
            ux, uy = (px-x)/distance, (py-y)/distance
            # Check continuous support from these exact feet. A clear ray alone
            # would let a bot under a deck attach its path to the floor above it.
            foot = z
            samples = math.ceil(distance)
            for k in range(1, samples+1):
                frac = k / samples
                support = w.floor(x+(px-x)*frac, y+(py-y)*frac, foot, stats["step_up"])
                if abs(support-foot) > stats["step_up"] + 0.05:
                    break
                foot = support
            else:
                if abs(foot-self.heights[start]) <= 0.1 and all(
                    w.clear(x-uy*r, y+ux*r, z, ux, uy, distance, stats["step_up"], stats["height"])
                    for r in (0, -stats["radius"], stats["radius"])
                ):
                    break
        else:
            return []

        travel = purpose == "travel"
        def remaining(n):
            horizontal = math.dist(self.point(n), goal[:2])
            return math.hypot(horizontal, self.heights[n]-goal[2]) if len(goal) == 3 else horizontal

        # Two paths to the same roof may leave very different amounts of fuel for the
        # next climb. Round costs up and omit future regeneration until observed.
        tick_energy = stats["jet_drain"] * 0.05
        initial = (start, int(max(0, energy) / tick_energy + 1e-8) if allow_air else 0)
        parent = {initial: None}
        costs = {initial: 0.0}
        # Skip a costlier route to a node if another leaves at least as much fuel.
        # This keeps long ground paths from filling the search with nearly identical jet detours.
        budgets = {start: {initial[1]: 0.0}}
        queue = [(0.0, 0.0, initial)]
        best = initial
        best_range = remaining(start) if travel else math.inf
        spots = []
        cx, cy, radius = circle
        start_ring = math.hypot(x-cx, y-cy)
        while queue and len(costs) < (1600 if travel else 430):
            _, cost, state = heapq.heappop(queue)
            n, fuel = state
            if cost != costs[state]:
                continue
            px, py = self.point(n)
            if travel:
                d = remaining(n)
                if d < best_range:
                    best, best_range = state, d
                if d < STEP and (len(goal) == 2 or abs(self.heights[n]-goal[2]) < stats["step_up"]):
                    best = state
                    break
            elif cost <= 92 and math.hypot(px-cx, py-cy) <= max(radius, start_ring):
                mask = self.cover[n]
                if purpose == "shelter":
                    protected = sum(bool(mask & (1 << (round(math.atan2(ty-py, tx-px)/(math.pi/4)) % 8)))
                                    for tx, ty, _ in threats)
                    if protected:
                        spots.append((cost - 24 * protected / len(threats), state))
                elif cost >= 12:
                    tx, ty, _ = threats[0]
                    angle = math.atan2(py-ty, px-tx) - math.atan2(y-ty, x-tx)
                    lateral = abs(math.sin(angle)) * math.hypot(x-tx, y-ty)
                    if lateral > 8 and math.hypot(px-tx, py-ty) >= min(18, math.hypot(x-tx, y-ty)):
                        elevation = max(-4, min(6, self.heights[n]-z))
                        spots.append((cost - min(lateral, 35) * 0.8 - min(mask.bit_count(), 3) - elevation, state))

            neighbors = []
            for bit, (offset, length) in enumerate(self.edges):
                if not self.links[n] & (1 << bit):
                    continue
                dest = n % self.layer_size + offset
                if self.upper[n] & (1 << bit):
                    dest += self.layer_size
                neighbors.append((dest, fuel, math.hypot(length, self.heights[dest]-self.heights[n]), False))
            if allow_air:
                for dest, seconds, ticks in self.flights.get(n, ()):
                    if ticks <= fuel:
                        # Include time to line up at the ledge; prefer walking when comparable.
                        neighbors.append((dest, fuel-ticks, (seconds+0.4)*stats["walk_speed"] + ticks*tick_energy*0.12, True))
            for dest, fuel_left, edge_cost, airborne in neighbors:
                child = (dest, fuel_left)
                new = cost + edge_cost
                if new >= costs.get(child, math.inf) or (not travel and new > 92):
                    continue
                dx, dy = self.point(dest)
                if math.hypot(dx-cx, dy-cy) > max(radius, start_ring) + 1:
                    continue
                alternatives = budgets.get(dest)
                if alternatives is None:
                    budgets[dest] = {fuel_left: new}
                else:
                    if any(f >= fuel_left and c <= new for f, c in alternatives.items()):
                        continue
                    alternatives[fuel_left] = new
                costs[child], parent[child] = new, (state, airborne)
                heapq.heappush(queue, (new + (remaining(dest) if travel else 0), new, child))

        if not travel:
            for _, state in heapq.nsmallest(20, spots):
                n = state[0]
                px, py = self.point(n)
                height = self.heights[n] + (stats["height"] * 0.85 if purpose == "shelter" else stats["muzzle_height"])
                visible = [w.los(px, py, height, tx, ty, tz, 1.5) for tx, ty, tz in threats]
                if (purpose == "shelter" and not any(visible)) or (purpose == "flank" and visible[0]):
                    best = state
                    break
            else:
                return []
        elif best == initial:
            return []
        path = []
        while True:
            n = best[0]
            edge = parent[best]
            path.append(Waypoint(*self.point(n), self.heights[n], edge[1] if edge else False))
            if edge is None:
                break
            best = edge[0]
        return path[::-1]
