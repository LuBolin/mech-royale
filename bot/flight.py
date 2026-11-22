"""Jump and jet transfers sampled at the arena's 20 Hz physics rate."""
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

DT = 0.05
CONTROL = 2  # physics steps per action


@dataclass(frozen=True)
class Flight:
    airtime: float
    peak_height: float
    energy_cost: float
    commands: tuple
    positions: tuple


@lru_cache(maxsize=512)
def _vertical(jump, gravity, jet, max_rise, burn_steps):
    """Vertical position after each physics step, starting with a grounded jump."""
    vz, z = jump, 0.0
    heights = [z]
    for tick in range(burn_steps):
        vz = min(vz + jet * DT, max_rise) - gravity * DT
        z += vz * DT
        heights.append(z)
    # Below launch height allow up to 150 m of descent, including all arena roofs.
    coast = math.ceil((vz + math.sqrt(vz*vz + 2*gravity*(max(0, z)+150))) / (gravity*DT)) + 2
    t = np.arange(1, coast + 1)
    return np.concatenate((heights, z + t * vz * DT - 0.5 * gravity * DT**2 * t * (t+1)))


def _vertical_args(stats, energy):
    drain = stats["jet_drain"] * DT * CONTROL
    burns = int(min(stats["energy"], max(0.0, energy)) / drain)
    return (stats["jump_speed"], stats["gravity"], stats["jet_accel"], stats["max_rise_speed"]), burns


def flight_envelope(stats, energy, elevation=0.0):
    """Height gain and descent time for the longest fully funded, continuous jet pulse."""
    args, burns = _vertical_args(stats, energy)
    profile = _vertical(*args, burns * CONTROL)
    peak = int(profile.argmax())
    below = np.flatnonzero(profile[peak:] < elevation)
    airtime = (peak + int(below[0])) * DT if profile[peak] >= elevation and len(below) else 0.0
    return float(profile[peak]), airtime


@lru_cache(maxsize=384)
def _horizontal(distance, vx, vy, speed, ground_accel, air_accel, delay, steps):
    """Trace in coordinates facing the landing point, including takeoff momentum."""
    px = py = 0.0
    points = [(px, py)]
    moves = []
    for tick in range(steps):
        if tick % CONTROL == 0:
            dx, dy = distance - px, -py
            remaining = math.hypot(dx, dy)
            scale = min(1.0 / max(remaining, 1e-6), 2.5 / speed,
                        math.sqrt(2 * air_accel / max(remaining, 1e-6)) / speed)
            mx, my = (dx*scale, dy*scale) if tick >= delay else (0.0, 0.0)
            moves.append((mx, my))
        dvx, dvy = mx*speed-vx, my*speed-vy
        accel = (air_accel if tick else ground_accel) * DT
        scale = min(1.0, accel / max(1e-9, math.hypot(dvx, dvy)))
        vx += dvx * scale
        vy += dvy * scale
        px += vx * DT
        py += vy * DT
        points.append((px, py))
    return np.array(points), moves


_ANGLES = np.arange(8) * (math.pi / 4)
_FOOTPRINT = np.vstack((np.zeros(2), np.column_stack((np.cos(_ANGLES), np.sin(_ANGLES)))))


def _clear(world, xs, ys, zs, stats, goal_z):
    """Reject intersecting arcs; the landing step may touch its intended surface."""
    radius, height = stats["radius"], stats["height"]
    x0, x1, y0, y1 = world.bounds
    if xs.min()-radius < x0 or xs.max()+radius > x1 or ys.min()-radius < y0 or ys.max()+radius > y1:
        return False
    i = np.rint(xs[:, None] + radius*_FOOTPRINT[:, 0] - world.x0).astype(int).clip(0, world.nx-1)
    j = np.rint(ys[:, None] + radius*_FOOTPRINT[:, 1] - world.y0).astype(int).clip(0, world.ny-1)
    feet = zs[:, None]
    # Terrain supports the centre of the cylinder; structures can contact its rim.
    terrain, top = world.ground[j, i], world.top[j, i]
    wall = (top > terrain + 0.15) & (top > feet + 0.12)
    overhead = (world.slab_lo[j, i] < feet + height) & (world.slab_hi[j, i] > feet + 0.12)
    trunk = world.trunk[j, i] > feet
    if wall.any() or overhead.any() or trunk.any() or (terrain[:, 0] > zs + 0.20).any():
        return False
    support = world.floor(float(xs[-1]), float(ys[-1]), goal_z, stats["step_up"])
    return abs(support - goal_z) < 0.4


def plan_flight(world, start, goal, stats, energy, velocity=(0, 0, 0), charging=False):
    """Plan a complete grounded transfer, or return None if no sampled arc lands safely."""
    x, y, z = start
    gx, gy, gz = goal
    dx, dy, dz = gx-x, gy-y, gz-z
    distance = math.hypot(dx, dy)
    if distance < 1e-4:
        return None
    ux, uy = dx/distance, dy/distance
    vx, vy = velocity[:2]
    speed = stats["walk_speed"] * (stats["charge_move_mult"] if charging else 1)
    args, burns = _vertical_args(stats, energy)
    full = _vertical(*args, burns * CONTROL)
    peak = int(full.argmax())
    if dz > full[peak]:
        return None
    landing = np.flatnonzero(full[peak:] < dz)
    if not len(landing):
        return None
    max_steps = peak + int(landing[0])
    # An immediate approach clears low or distant ledges cheaply. A vertical
    # start makes a near wall reachable without assuming the bot can fly through it.
    delays = [0]
    clear_wall = np.flatnonzero(full[:peak+1] > dz + 0.3)
    if dz > stats["step_up"] and len(clear_wall):
        delays.append(CONTROL * math.ceil(int(clear_wall[0])/CONTROL))
    horizontal = [_horizontal(distance, vx*ux+vy*uy, vy*ux-vx*uy, speed,
                              stats["ground_accel"], stats["air_accel"], d, max_steps)
                  for d in delays]
    # Select the least-fuel feasible pulse. Both ignition and cutoff coincide
    # with decision boundaries, so the controller can actually fly the returned arc.
    for burn in range(burns+1):
        heights = _vertical(*args, burn * CONTROL)
        apex = int(heights.argmax())
        if heights[apex] < dz:
            continue
        falling = np.flatnonzero(heights[apex:] <= dz)
        if not len(falling):
            continue
        end = apex + int(falling[0])
        if not 0 < end <= max_steps:
            continue
        for points, moves in horizontal:
            if math.hypot(points[end, 0]-distance, points[end, 1]) > 1.2:
                continue
            # The centre must enter the landing platform. Clamp only the
            # descending contact step, as the simulator does on touchdown.
            a, b = points[1:end+1].T
            xs, ys = x+ux*a-uy*b, y+uy*a+ux*b
            zs = z + heights[1:end+1]
            zs[-1] = gz
            if not _clear(world, xs, ys, zs, stats, gz):
                continue
            commands = tuple(((ux*ax-uy*ay, uy*ax+ux*ay), i < burn, i == 0)
                             for i, (ax, ay) in enumerate(moves[:math.ceil(end/CONTROL)]))
            pos = np.column_stack((xs, ys, zs))
            positions = tuple(tuple(p) for p in pos[np.minimum(np.arange(len(commands))*CONTROL + 1, end-1)])
            cost = min(burn*CONTROL, end) * DT * stats["jet_drain"]
            return Flight(airtime=end*DT, peak_height=float(heights[apex]), energy_cost=cost,
                          commands=commands, positions=positions)
    return None
