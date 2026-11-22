"""Local combat footwork using shot lines at the actual height of each mech."""
import math

import numpy as np


def _move_to(me, point):
    dx, dy = point[0] - me["pos"][0], point[1] - me["pos"][1]
    distance = math.hypot(dx, dy)
    if distance < 0.35:
        return (0.0, 0.0)
    # Stop accelerating before overshooting a narrow peek. The motor may boost.
    scale = min(1.0, distance / 3.0) / distance
    return dx * scale, dy * scale


def choose_position(bot, obs, target, enemies, preferred, fallback):
    """Return a world-space move command, or None to keep the existing footwork."""
    me, now = obs["self"], obs["t"]
    if not target or not me["grounded"]:
        return None
    x, y, z = me["pos"]
    st, world = me["stats"], bot.world
    tx, ty = target["pos"][:2]
    distance = max(0.1, math.hypot(tx-x, ty-y))
    rx, ry = (tx-x)/distance, (ty-y)/distance
    # Evaluate the selected opponent and up to three other plausible shooters.
    others = sorted((e for e in enemies if e["id"] != target["id"] and
                     math.hypot(e["pos"][0]-x, e["pos"][1]-y) <= 245),
                    key=lambda e: math.hypot(e["pos"][0]-x, e["pos"][1]-y))[:3]
    foes = [target, *others]
    ours = {w["kind"]: w["stats"] for w in me["weapons"] if w}
    coil = ours.get("lightning")
    enemy_coil = "lightning" in target["weapons"]
    if coil:
        preferred = min(preferred, coil["range"] * 0.72)
    elif enemy_coil:
        preferred = max(preferred, bot.cfg["weapons"]["lightning"]["range"] + 18)

    attention = []
    for e in foes:
        ex, ey = e["pos"][:2]
        dx, dy = x-ex, y-ey
        yaw = e["aim"][0]
        facing = (dx*math.cos(yaw) + dy*math.sin(yaw)) / max(0.1, math.hypot(dx, dy))
        # A mech facing someone else can still turn back. Keep a smaller crossing cost.
        attention.append(1.15 if e.get("charging") and facing > 0.95 else 1.0 if facing > 0.85 else 0.55)
    key = tuple((e["id"], att) for e, att in zip(foes, attention)), tuple(ours), tuple(target["weapons"])
    cached = getattr(bot, "_combat_step", None)
    if cached and key == cached[0] and now < cached[1]:
        return _move_to(me, cached[2]) if cached[2] is not None else None

    # Sample the near side, edge, and far side of a corner along each heading.
    # Keep path samples grouped so we can charge for crossing a second shooter's lane.
    points = [(x, y, z)]
    paths = [[0]]
    fallback_length = max(0.001, math.hypot(*fallback))
    fx, fy = fallback[0]/fallback_length, fallback[1]/fallback_length
    zone = obs["zone"]
    for k in range(12):
        angle = math.tau * k / 12
        c, s = math.cos(angle), math.sin(angle)
        ux, uy = rx*c-ry*s, ry*c+rx*s
        path = []
        for step in (3.0, 7.5, 15.0):
            px, py = x+step*ux, y+step*uy
            if math.dist((px,py), zone["center"]) + st["radius"] > zone["radius"]:
                break
            if not all(world.clear(x-uy*r, y+ux*r, z, ux, uy, step, st["step_up"], st["height"])
                       for r in (0, -0.85*st["radius"], 0.85*st["radius"])):
                break
            floor = world.floor(px, py, z, st["step_up"])
            if floor < z - st["step_up"] - 1.0:
                break
            path.append(len(points))
            points.append((px,py,floor))
            paths.append(path.copy())

    feet = np.asarray(points, dtype=float)
    body = feet + [0, 0, st["height"] * 0.5]
    muzzle = feet + [0, 0, st["muzzle_height"]]
    foe_feet = np.asarray([e["pos"] for e in foes], dtype=float)
    sizes = 1 + 0.05 * np.array([e.get("cubes", 0) for e in foes])
    foe_centers = foe_feet.copy()
    foe_centers[:,2] += 4.5 * sizes
    foe_guns = foe_feet.copy()
    foe_guns[:,2] += 7.0 * sizes
    outgoing = world.los_many(np.tile(muzzle, (len(foes), 1)),
                              np.repeat(foe_centers, len(body), axis=0)).reshape(len(foes), len(body))
    incoming = world.los_many(np.repeat(foe_guns, len(body), axis=0),
                              np.tile(body,(len(foes),1))).reshape(len(foes), len(body))
    ranges = np.linalg.norm(body[None,:,:] - foe_centers[:,None,:], axis=2)
    our_range = max((w.get("range", 200) for w in ours.values()), default=0)
    shots = outgoing & (ranges < our_range)
    can_shoot = shots.any(axis=0)
    pressure = np.zeros(len(body))
    strongest = np.zeros(len(body))
    for i, foe in enumerate(foes):
        weapons = [kind for kind in foe["weapons"] if kind]
        reach = max((bot.cfg["weapons"][kind].get("range", 200) for kind in weapons), default=0)
        danger = incoming[i] * (ranges[i] < reach) * attention[i]
        pressure += danger
        strongest = np.maximum(strongest, danger)
    crossfire = pressure - strongest

    # Reward actual outgoing shots, discount exposed paths and strongly prefer only
    # one opponent with a firing solution. Height alone earns no points.
    score = 5.6 * can_shoot + 0.2 * shots[0] - 1.6 * pressure - 3.2 * crossfire
    distance_error = np.abs(ranges[0] - preferred)
    score += 0.043 * (distance_error[0] - distance_error)
    if coil:
        score += 2.8 * (outgoing[0] & (ranges[0] <= coil["range"] * 0.94))
        if distance > coil["range"]:
            score += 0.14 * (ranges[0,0]-ranges[0])
    elif enemy_coil:
        leash = bot.cfg["weapons"]["lightning"]["range"] + 6
        score -= 0.18 * np.maximum(0.0, leash-ranges[0])

    choice = None
    geometry_matters = (can_shoot.any() != can_shoot.all() or
                        np.ptp(pressure) > 0.2 or (can_shoot[0] and not incoming[0,0]))
    if geometry_matters or coil or enemy_coil:
        # A small movement bias preserves strafing in open ground. Near cover an equally
        # good firing lane a single step away should beat a broad, exposed swing.
        displacement = feet[:,:2] - (x,y)
        length = np.linalg.norm(displacement, axis=1)
        speed_bias = displacement @ np.array([fx,fy]) / np.maximum(1, length)
        score += 0.3 * speed_bias - (0.045 if geometry_matters else 0.012) * length
        for i,path in enumerate(paths):
            score[i] -= 1.7 * (max(crossfire[k] for k in path) - crossfire[i])
        best = int(score.argmax())
        choice = points[best][:2]
    bot._corner_shot = bool(choice is not None and geometry_matters and can_shoot[best])
    bot._combat_step = key, now + 0.3, choice
    return _move_to(me, choice) if choice is not None else None
