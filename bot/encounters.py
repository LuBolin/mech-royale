"""Decisions that depend on who is exchanging fire, and who can interrupt it."""
import math


def facing(shooter, pos, cone):
    x, y, _ = shooter["pos"]
    yaw = shooter["aim"][0]
    dx, dy = pos[0]-x, pos[1]-y
    return dx*math.cos(yaw) + dy*math.sin(yaw) >= math.cos(cone)*math.hypot(dx, dy)


def shot_lane(world, shooter, victim):
    x, y, z = shooter["pos"]
    tx, ty, tz = victim["pos"]
    return world.los(x, y, z+7*(1+0.05*shooter.get("cubes", 0)),
                     tx, ty, tz+4.5*(1+0.05*victim.get("cubes", 0)))


class Encounter:
    def __init__(self):
        self.hp = {}
        self.duel = None
        self.duel_until = -1
        self.recover_until = -1

    def observe(self, obs, enemies, world):
        """Confirm an exchange from a beam, or damage plus a matching third-party gun report."""
        t, me = obs["t"], obs["self"]
        known = {m["id"]: m for m in enemies}
        for beam in obs["beams"]:
            shooter = beam["by"]
            if shooter == me["id"] or shooter not in known:
                continue
            for end in beam["path"][1:]:
                for m in enemies:
                    if m["id"] == shooter:
                        continue
                    x, y, z = m["pos"]
                    if math.hypot(end[0]-x, end[1]-y) < 4 and z-1 <= end[2] <= z+10:
                        self.duel = (shooter, m["id"])
                        self.duel_until = t + 2.0 - beam.get("age", 0)
        our_hits = {e.get("target") for e in obs["events"] if e["type"] == "hit"}
        for m in enemies:
            old = self.hp.get(m["id"])
            if not old or t-old[0] > 0.25 or m["id"] in our_hits or m["hp"] >= old[1]-5:
                continue
            for s in obs["sounds"]:
                if s["kind"] != "gun" or s["age"] > 0.35:
                    continue
                for other in enemies:
                    if other is m or not facing(other, m["pos"], 0.19) or facing(other, me["pos"], 0.24):
                        continue
                    dx, dy = other["pos"][0]-me["pos"][0], other["pos"][1]-me["pos"][1]
                    if (dx*math.cos(s["bearing"])+dy*math.sin(s["bearing"]) >
                            0.95*math.hypot(dx, dy) and shot_lane(world, other, m)):
                        self.duel = (other["id"], m["id"])
                        self.duel_until = t + 1.5
                        break
        self.hp = {m["id"]: (t, m["hp"]) for m in enemies}
        if self.duel and t < self.duel_until and all(mid in known for mid in self.duel):
            return self.duel
        return ()

    def choose(self, obs, world, target, enemies, duel, attackers):
        t, me = obs["t"], obs["self"]
        hp = me["hp"]/me["max_hp"]
        facing_us = []
        for enemy in enemies:
            if not (facing(enemy, me["pos"], 0.24) or t-attackers.get(enemy["id"], (-99,))[0] < 1):
                continue
            if math.dist(enemy["pos"], me["pos"]) < 240 and shot_lane(world, enemy, me):
                facing_us.append(enemy)
        # Estimate damage we can actually deliver during a short peek, including
        # cooldown, reload, aim slew, range and gun travel time.
        finish = False
        if target:
            x,y,z = me["pos"]
            tx,ty,tz = target["pos"]
            distance = math.hypot(tx-x, ty-y)
            error = abs(math.remainder(math.atan2(ty-y,tx-x)-me["aim"][0], math.tau))
            window = 0.9 - error/max(0.1,me["stats"]["turn_rate"])
            burst = 0
            for weapon in me["weapons"]:
                if not weapon:
                    continue
                s = weapon["stats"]
                if distance >= s.get("range", 200):
                    continue
                available = window-max(weapon["cooldown"],weapon.get("equip",0))
                if weapon["kind"] == "gun":
                    available -= distance/s["speed"]
                    if available >= 0 and weapon["reload"] <= 0:
                        burst += min(weapon["ammo"],1+int(available/s["cooldown"]))*s["damage"]
                elif weapon["kind"] == "lightning" and available >= 0:
                    burst += min(int(me["energy"]/s["energy"]),1+int(available/s["cooldown"]))*s["damage"]
                elif weapon["kind"] == "laser" and available >= 0 and weapon.get("charge",0) >= s["min_charge"]:
                    burst += s["damage"]*(s["min_frac"] + (1-s["min_frac"])*min(1,weapon["charge"]/s["charge"]))
            finish = (target["hp"] <= burst*(1+0.04*min(me["cubes"],10))*0.8
                      and shot_lane(world, me, target))

        if finish or hp >= 0.8:
            self.recover_until = -1
        if len(facing_us) >= 2 and hp < 0.68 and not finish:
            self.recover_until = t + me["stats"]["regen_delay"] + 4
        if t < self.recover_until:
            return "retreat" if facing_us else "recover"
        if duel and target and target["id"] in duel and not facing_us:
            # If the other duelist also has a return shot, wait behind the wall or rotate.
            other = next((e for e in enemies if e["id"] in duel and e["id"] != target["id"]), None)
            if other and not finish and shot_lane(world, other, me):
                return "screen"
            if shot_lane(world, me, target):
                return "engage"
            return "angle"
        return "engage" if finish else None
