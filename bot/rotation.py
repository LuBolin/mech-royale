"""Timing a rotation against the observed moving ring."""
import math


def departure_slack(world, me, zone, now, target, margin):
    """Seconds left to fight before leaving for the nearest part of the next safe circle."""
    x, y, z = me["pos"]
    nx, ny = zone["next_center"]
    radius = zone["next_radius"]
    distance = math.hypot(nx-x, ny-y)
    remaining = distance - max(0.0, radius-margin)
    if not me["in_zone"]:
        return -math.inf
    if remaining <= 1e-6:
        return math.inf

    cx, cy = zone["center"]
    hold = max(0.0, zone["shrink_start"]-now)
    shrink = max(0.0, zone["shrink_end"]-now-hold)
    # Find the instant the moving boundary will cross our current position.
    crossing = math.inf
    if distance > radius:
        lo, hi = 0.0, 1.0
        for _ in range(8):
            f = (lo+hi)/2
            ring = zone["radius"] + f*(radius-zone["radius"])
            if math.hypot(x-cx-f*(nx-cx), y-cy-f*(ny-cy)) <= ring:
                lo = f
            else:
                hi = f
        crossing = hold + shrink*lo

    st = me["stats"]
    ux, uy = (nx-x)/distance, (ny-y)/distance
    clear = world.clear(x, y, z, ux, uy, min(remaining, 32), st["step_up"], st["height"])
    travel = remaining/st["walk_speed"] * (1.1 if clear else 1.45)
    # Footing, acceleration and a local detour can consume part of the nominal ETA.
    buffer = 1.5 if clear else 3.5
    if target is not None:
        tx, ty = target["pos"][:2]
        vx, vy = target["vel"][:2]
        # Budget for the separation a short pursuit would add. An enemy running
        # along our rotation can still be engaged by the independent torso.
        away = max(0.0, -(vx*ux+vy*uy))
        if (tx-x)*ux + (ty-y)*uy < 0 or away > 1:
            buffer += min(3.0, 0.75+2*away/st["walk_speed"])
    return min(hold+shrink-travel, crossing) - buffer
