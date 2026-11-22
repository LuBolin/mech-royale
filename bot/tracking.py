"""Pre-aim a reachable sightline using the enemy's current memory observation."""
import math


def exit_aim(world, origin, target, stats):
    x, y, muzzle = origin
    tx, ty, tz = target["pos"]
    vx, vy, vz = target["vel"]
    half_height = 4.5 * (1 + 0.05 * target.get("cubes", 0))
    speed = math.hypot(vx, vy)
    if speed > 1.5:
        # Reread velocity every decision. An observed reversal flips the search immediately.
        directions = [(vx/speed, vy/speed)]
        reach = min(36, max(6, 1.5 * speed))
    else:
        # A stationary mech gives no reliable exit direction. Hold the closest
        # reachable edge on either side instead of projecting stale velocity.
        distance = max(0.1, math.hypot(tx-x, ty-y))
        side = (-(ty-y)/distance, (tx-x)/distance)
        directions = [side, (-side[0], -side[1])]
        reach = 16
    points = []
    for dx, dy in directions:
        px, py, feet = tx, ty, tz
        for step in range(2, int(reach)+1, 2):
            if not world.clear(px, py, feet, dx, dy, 2.0, stats["step_up"], half_height*2):
                break
            px, py = tx + dx*step, ty + dy*step
            floor = world.floor(px, py, feet, stats["step_up"])
            elapsed = step / max(speed, stats["walk_speed"] * 0.5)
            feet = max(floor, tz + elapsed*vz - 0.5*stats["gravity"]*elapsed**2)
            if world.los(x, y, muzzle, px, py, feet + half_height, 1.0):
                points.append((step, px, py, feet + half_height))
                break
    if points:
        _, tx, ty, center = min(points, key=lambda p: p[0])
    else:
        center = tz + half_height
    return math.atan2(ty-y, tx-x), math.atan2(center-muzzle, max(0.1, math.hypot(tx-x, ty-y)))


def clear_mortar_arc(world, origin, yaw, pitch, speed, gravity, distance):
    """Check the curved route to a remembered target before shooting over a wall."""
    x, y, z = origin
    horizontal = speed * math.cos(pitch)
    vx, vy = horizontal * math.cos(yaw), horizontal * math.sin(yaw)
    vz = speed * math.sin(pitch)
    duration = distance / max(horizontal, 0.01)
    pieces = max(1, math.ceil(duration / 0.2))
    start = origin
    for k in range(1, pieces+1):
        t = duration * k / pieces
        end = (x + vx*t, y + vy*t, z + vz*t - 0.5*gravity*t*t)
        if world.solid(*end) or not world.los(*start, *end, 1.0):
            return False
        start = end
    return True
