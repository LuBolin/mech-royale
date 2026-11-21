"""Parameterised battle-royale bot. Every behaviour knob comes from params.py (the genome)."""
import math
import random

import numpy as np

import params as PR
from world import get_world

TAU = 2 * math.pi
LOUD = ("gun", "mortar", "explosion", "laser", "lightning", "laser_charge")
HEAL_HP = 0.85  # after contact breaks, hide until regen brings hp back to this fraction


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def ballistic(v, g, R, dz, high):
    """Launch pitch to hit a point R metres away and dz higher, or None if out of reach."""
    v2 = v * v
    disc = v2 * v2 - g * (g * R * R + 2 * dz * v2)
    if disc < 0 or R < 1:
        return None
    r = math.sqrt(disc)
    return math.atan((v2 + r) / (g * R)) if high else math.atan((v2 - r) / (g * R))


class Bot:
    name = "ponybot"

    def __init__(self, hello, params=None, aim_noise=0.0):
        cfg = hello["config"]
        self.cfg = cfg
        self.world = get_world(hello["map"]["dir"])
        p = PR.defaults()
        p.update(PR.load() if params is None else params)
        self.p = p
        self.rng = random.Random(hello.get("seed", 0))
        self.aim_noise = aim_noise
        self.reach = cfg["loot"]["pickup_radius"]
        self.wpref = {"gun": -1.0, "mortar": p["w_mortar"], "laser": p["w_laser"], "lightning": p["w_lightning"]}
        self.threat = None  # (t, bearing, attacker id)
        self.sound = None  # (t, bearing)
        self.hist = []
        self.unstick = -1.0
        self.detour = 1
        self.strafe = 1
        self.strafe_next = 0.0
        self.scan = 0.0
        self.goal_hint = None  # (x, y) set by subclasses (house bots hunting)
        self.mode = "roam"
        # Survival logic (threat list, unseen-fire response, hide-and-heal). House bots switch it off so the
        # simulator's opponents keep the logic that was calibrated against real practice scores.
        self.survival = p.get("survival", 1.0) >= 0.5
        self.attackers = {}  # attacker id -> (t, bearing) of its latest hit on us
        self.last_hit = -1e9
        self.flee_cache = (-1.0, None)  # (valid until t, (ux, uy))

    # ------------------------------------------------------------------ main
    def act(self, obs):
        p, W, rng = self.p, self.world, self.rng
        t = obs["t"]
        me = obs["self"]
        x, y, z = me["pos"]
        st = me["stats"]
        hpf = me["hp"] / max(1.0, me["max_hp"])
        energy = me["energy"]
        weapons = me["weapons"]
        kinds = [w["kind"] if w else None for w in weapons]
        a = {"tick": obs["tick"]}

        for e in obs["events"]:
            if e["type"] == "damaged" and e.get("bearing") is not None:
                self.threat = (t, e["bearing"], e.get("by"))
                if e.get("by") is not None:
                    self.attackers[e["by"]] = (t, e["bearing"])
                    self.last_hit = t
        for s in obs["sounds"]:
            if s["kind"] in LOUD and s["dist"] != "far":
                self.sound = (t - s["age"], s["bearing"])
        if obs["offer"]:
            ch = obs["offer"]["choices"]
            a["pick"] = max(range(len(ch)), key=lambda i: p.get("up_" + ch[i]["id"], 0.0))

        mechs = [m for m in obs["mechs"] if m["kind"] == "mech"]
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
        hit_recent = self.threat is not None and t - self.threat[0] < 2.5

        # ---- target
        target, best = None, 1e18
        for m in mechs:
            d = math.hypot(m["pos"][0] - x, m["pos"][1] - y)
            if d > p["engage_range"]:
                continue
            s = d / 100.0 + p["tgt_w_hp"] * m.get("hp", 1000) / max(1.0, m.get("max_hp", 1400))
            if hit_recent and self.threat[2] == m["id"]:
                s -= p["tgt_w_threat"]
            if s < best:
                best, target = s, m

        # ---- zone geometry
        zone = obs["zone"]
        nc, nr = zone["next_center"], zone["next_radius"]
        cc, cr = zone["center"], zone["radius"]
        dn = math.hypot(x - nc[0], y - nc[1])
        dc = math.hypot(x - cc[0], y - cc[1])
        margin = min(p["zone_margin"], 0.5 * nr)
        tts = zone["shrink_start"] - t
        walk = st["walk_speed"]
        need = max(0.0, dn - (nr - margin)) / walk
        zone_urgent = (not me["in_zone"]) or dc > cr - 4 or (dn > nr - margin and tts < need + p["zone_lead"])
        ux, uy = ((x - nc[0]) / dn, (y - nc[1]) / dn) if dn > 1e-6 else (1.0, 0.0)
        hr = max(0.0, nr - margin) * p["center_bias"]
        home = (nc[0] + ux * min(dn, hr), nc[1] + uy * min(dn, hr))

        # ---- threats: visible mechs in engage range plus anyone who hit us in the last 5 s, seen or not.
        # An unseen attacker is placed 60 m out along the bearing it hit us from.
        threats = []
        if self.survival:
            seen = set()
            for m in mechs:
                hit = self.attackers.get(m["id"])
                if math.hypot(m["pos"][0] - x, m["pos"][1] - y) < p["engage_range"] or (hit and t - hit[0] < 5):
                    threats.append((m["pos"][0], m["pos"][1], m["pos"][2], m.get("hp", 1400)))
                    seen.add(m["id"])
            for aid, (ta, b) in list(self.attackers.items()):
                if t - ta >= 5:
                    del self.attackers[aid]
                elif aid not in seen:
                    threats.append((x + math.cos(b) * 60, y + math.sin(b) * 60, z, 1400))

        # ---- mode
        mode = "roam"
        strength = me["hp"] * (1 + 0.04 * min(me["cubes"], 10)) * (1.3 if kinds[1] else 1.0)
        aggr = p["aggr"] * (p["late_aggr"] if obs["alive"] <= p["late_alive"] else 1.0)
        if self.survival:
            if threats:
                # judge the fight against everyone involved, not one target; fight back 1-on-1 only
                n = len(threats)
                fight = target is not None and hpf >= p["flee_hp"] and n <= p["max_enemies"] and (
                    strength * aggr >= sum(th[3] for th in threats)
                    or (hit_recent and self.threat[2] == target["id"]))
                mode = "fight" if fight else "flee"
            elif hpf < HEAL_HP and t - self.last_hit < 45:
                mode = "hide"  # contact broken: stand still (silent) until regen refills us
        elif target is not None:
            near = sum(1 for m in mechs if math.hypot(m["pos"][0] - x, m["pos"][1] - y) < p["engage_range"])
            fight = hpf >= p["flee_hp"] and strength * aggr >= target.get("hp", 1400) and near <= p["max_enemies"]
            mode = "fight" if fight or (hit_recent and self.threat[2] == target["id"] and hpf >= p["flee_hp"]) else "flee"
        # early zone damage is a fraction of one gun's dps: finish breaking contact before running for the zone
        flee_first = self.survival and mode == "flee" and zone["dps"] < 50 and me["hp"] > 10 * zone["dps"]
        if zone_urgent and not flee_first:
            mode = "zone"
        self.mode = mode

        mvx = mvy = 0.0
        boost = jump = False
        shoot_obj = None
        goal_z = None
        if mode == "zone":
            mvx, mvy = home[0] - x, home[1] - y
            boost = (not me["in_zone"] or tts < need) and energy > p["boost_energy"] * 0.5
        elif mode == "fight":
            tx, ty = target["pos"][0], target["pos"][1]
            d = max(1e-3, math.hypot(tx - x, ty - y))
            vx_, vy_ = (tx - x) / d, (ty - y) / d
            pref = p["pref_dist"]
            if "lightning" in kinds:
                pref = min(pref, p["lightning_dist"])
            radial = clamp((d - pref) / 15.0, -1.0, 1.0)
            if t >= self.strafe_next:
                if rng.random() < 0.7:
                    self.strafe = -self.strafe
                self.strafe_next = t + p["strafe_period"] * (0.5 + rng.random())
            s = p["strafe"] * self.strafe
            mvx, mvy = vx_ * radial - vy_ * s, vy_ * radial + vx_ * s
        elif mode == "flee":
            if self.survival:
                mvx, mvy = self.flee_cover(x, y, z, t, threats, nc, nr, margin, st)
            else:
                mvx, mvy = self.flee_dir(x, y, z, target, nc, nr, margin, st)
            boost = energy > p["boost_energy"]
        elif mode == "hide":
            pass  # stay put; regen needs 5 s without damage
        else:
            goal, gkind, gobj, gbest = None, None, None, 1e18
            safe_r = nr - margin * 0.5

            def safe(px, py, d):
                return math.hypot(px - nc[0], py - nc[1]) < safe_r or tts > p["zone_lead"] + d / walk + 5

            cur_pref = self.wpref[kinds[1]] + p["swap_margin"] if kinds[1] else 0.0
            for pk in obs["pickups"]:
                k = pk["kind"]
                if k in kinds:
                    continue
                gain = self.wpref[k] - cur_pref
                if gain <= 0:
                    continue
                px, py, pz = pk["pos"]
                d = math.hypot(px - x, py - y) + 2 * max(0.0, pz - z)
                if d < p["loot_dist"] and safe(px, py, d):
                    sc = d / (0.2 + gain)
                    if sc < gbest:
                        gbest, goal, gkind, gobj = sc, (px, py, pz), "loot", pk
            for dr in obs["drops"]:
                px, py, pz = dr["pos"]
                d = math.hypot(px - x, py - y)
                contested = any(math.hypot(m["pos"][0] - px, m["pos"][1] - py) < p["drop_contest"] for m in mechs)
                if d < p["drop_dist"] and dr["holder"] in (None, me["id"]) and dr["lands_in"] < d / walk + 8 \
                        and not contested:
                    sc = d * 0.7
                    if sc < gbest:
                        gbest, goal, gkind, gobj = sc, (px, py, pz), "drop", dr
            for c in obs["cubes"]:
                px, py, pz = c["pos"]
                d = math.hypot(px - x, py - y) + 2 * max(0.0, pz - z)
                if d < p["cube_dist"] and safe(px, py, d) and d * 0.8 < gbest:
                    gbest, goal, gkind, gobj = d * 0.8, (px, py, pz), "cube", c
            gun = weapons[0] if kinds[0] == "gun" else None
            if gun and (gun["ammo"] > 0 or gun["reload"] > 0):
                for c in obs["crates"]:
                    px, py, pz = c["pos"]
                    d = math.hypot(px - x, py - y)
                    if d < p["crate_dist"] and safe(px, py, d) and d + 15 < gbest:
                        gbest, goal, gkind, gobj = d + 15, (px, py, pz), "crate", c
            if hpf > p["npc_hp"]:
                for n in npcs:
                    px, py, pz = n["pos"]
                    d = math.hypot(px - x, py - y)
                    if d < p["npc_dist"] and safe(px, py, d) and d + 25 < gbest:
                        gbest, goal, gkind, gobj = d + 25, (px, py, pz), "npc", n
            if self.goal_hint is not None:
                goal, gkind = (self.goal_hint[0], self.goal_hint[1], z), "hint"
            if goal is None:
                goal, gkind = (home[0], home[1], z), "home"
            gx, gy, gz = goal
            d = math.hypot(gx - x, gy - y)
            if gkind in ("crate", "npc"):
                shoot_obj = gobj
                standoff = 18.0
                mvx, mvy = (gx - x, gy - y) if d > standoff else (0.0, 0.0)
            elif gkind == "home" and d < 6:
                if p["camp"] < 0.5:
                    ang = t * 0.3 + self.rng.random() * 0.1
                    mvx, mvy = math.cos(ang) * 0.35, math.sin(ang) * 0.35
            else:
                mvx, mvy = gx - x, gy - y
                if gkind in ("loot", "cube") and gz > z + st["step_up"] and d < 12:
                    goal_z = gz
            m_ = math.hypot(mvx, mvy)
            if m_ > 1e-6:
                scale = 1.0 if gkind in ("drop", "hint") else 1.0 - 0.6 * p["quiet"]
                if gkind == "home" and d < 6:
                    scale = min(scale, 0.35)
                mvx, mvy = mvx / m_ * scale, mvy / m_ * scale

        # ---- dodge incoming mortar shells (their path is visible and gravity is known)
        shell = self.shell_threat(obs["projectiles"], x, y, z)
        if shell is not None:
            lx, ly = shell
            d = math.hypot(x - lx, y - ly) or 1e-3
            mvx, mvy = (x - lx) / d, (y - ly) / d
            boost = energy > 10
            mode = self.mode = "dodge"

        # ---- navigation
        m_ = math.hypot(mvx, mvy)
        if m_ > 1.0:
            mvx, mvy, m_ = mvx / m_, mvy / m_, 1.0
        need_jet = False
        if m_ > 1e-3:
            mvx, mvy, need_jet = self.steer(x, y, z, mvx, mvy, m_, st)
        self.hist.append((t, x, y))
        if len(self.hist) > 12:
            self.hist.pop(0)
        if m_ > 0.5 and len(self.hist) >= 10 and t - self.hist[0][0] < 1.5:
            if math.hypot(x - self.hist[0][1], y - self.hist[0][2]) < 1.2:
                self.unstick = t + 0.8
                self.detour = -self.detour
                self.hist.clear()
                jump = me["grounded"]  # one hop, no jet burn
        if t < self.unstick and m_ > 1e-3:  # stuck: sidestep along the obstacle
            mvx, mvy = -mvy * self.detour, mvx * self.detour
        if need_jet or goal_z is not None:  # climbing is the only way forward
            if me["grounded"]:
                jump = True
            boost = energy > 5

        # ---- aim and fire
        yaw, pitch = me["aim"]
        want_yaw, want_pitch = None, 0.0
        fire0 = fire1 = False
        muz = z + st["muzzle_height"]
        eye = z + st["eye_height"]
        if target is not None:
            want_yaw, want_pitch, fire0, fire1 = self.shoot(me, target, kinds, weapons, x, y, z, muz, eye)
        elif shoot_obj is not None:
            px, py, pz = shoot_obj["pos"]
            d = math.hypot(px - x, py - y)
            want_yaw = math.atan2(py - y, px - x)
            want_pitch = math.atan2(pz + 1.2 - muz, max(1.0, d))
            tol = math.atan2(1.2 * p["aim_tol"], max(d, 1.0))
            fire0 = d < 120 and abs(wrap(yaw - want_yaw)) < tol and abs(pitch - want_pitch) < tol * 1.5
        elif self.threat is not None and t - self.threat[0] < p["face_threat_s"]:
            want_yaw = self.threat[1]
        elif self.sound is not None and t - self.sound[0] < 2.0 and p["face_sound"] >= 0.5:
            want_yaw = self.sound[1]
        elif math.hypot(mvx, mvy) > 0.5:
            want_yaw = math.atan2(mvy, mvx)
        else:
            self.scan += p["scan_speed"] * 0.1
            want_yaw = yaw + p["scan_speed"] * 0.25
        if want_yaw is None:
            want_yaw = yaw

        # dodge: a visible laser charging at us, or fresh damage
        if me["grounded"] and p["dodge"] > 0:
            danger = hit_recent and t - self.threat[0] < 0.3
            for m in mechs:
                if m.get("charging"):
                    b = math.atan2(y - m["pos"][1], x - m["pos"][0])
                    if abs(wrap(m["aim"][0] - b)) < 0.08:
                        danger = True
            if danger and rng.random() < p["dodge"]:
                self.strafe = -self.strafe
                if rng.random() < 0.5:
                    jump = True

        gun = weapons[0] if kinds[0] == "gun" else None
        if gun and target is None and gun["reload"] <= 0 and gun["ammo"] < gun["stats"]["mag"] * p["reload_frac"]:
            a["reload"] = True

        # weapon pickup within reach
        for pk in obs["pickups"]:
            k = pk["kind"]
            if k in kinds or math.hypot(pk["pos"][0] - x, pk["pos"][1] - y) > self.reach - 0.3:
                continue
            if abs(pk["pos"][2] - z) > 3.8:
                continue
            if kinds[1] is None and self.wpref[k] > 0:
                a["take"] = pk["id"]
                break
            if kinds[1] and self.wpref[k] > self.wpref[kinds[1]] + p["swap_margin"]:
                if not (kinds[1] == "laser" and weapons[1].get("charge", 0) > 0):
                    a["take"] = pk["id"]
                    a["drop"] = 1
                    break

        if self.aim_noise:
            want_yaw += rng.gauss(0, self.aim_noise)
            want_pitch += rng.gauss(0, self.aim_noise * 0.5)
        want_pitch = clamp(want_pitch, st["min_pitch"], st["max_pitch"])
        a["move"] = [round(mvx, 3), round(mvy, 3)]
        a["aim"] = [round(wrap(want_yaw), 4), round(want_pitch, 4)]
        a["boost"] = bool(boost)
        a["jump"] = bool(jump)
        a["fire"] = [bool(fire0), bool(fire1)]
        return a

    # ------------------------------------------------------------------ helpers
    def shoot(self, me, target, kinds, weapons, x, y, z, muz, eye):
        p, W = self.p, self.world
        yaw, pitch = me["aim"]
        tx, ty, tz = target["pos"]
        tvx, tvy, tvz = target["vel"]
        size = 1 + 0.05 * target.get("cubes", 0)
        tcz = tz + 4.5 * size
        dxy = max(1.0, math.hypot(tx - x, ty - y))
        vis = W.los(x, y, eye, tx, ty, tcz)
        tol = math.atan2(2.5 * size * p["aim_tol"], dxy)
        gun = weapons[0] if kinds[0] == "gun" else None
        gs = gun["stats"]["speed"] if gun else 252.0
        T = dxy / gs
        lx, ly, lz = tx + tvx * T * p["lead"], ty + tvy * T * p["lead"], tcz + tvz * T * p["lead"] * 0.5
        g_yaw = math.atan2(ly - y, lx - x)
        g_pitch = math.atan2(lz - muz, math.hypot(lx - x, ly - y))
        d_yaw = math.atan2(ty - y, tx - x)
        d_pitch = math.atan2(tcz - muz, dxy)
        want = (g_yaw, g_pitch)
        fire1 = False
        k1, w1 = kinds[1], weapons[1]
        use = "gun"
        if k1 == "mortar" and w1["cooldown"] < 0.3:
            ms = w1["stats"]
            tube = z + me["stats"]["height"] + 0.5
            th = ballistic(ms["speed"], ms["gravity"], dxy, tz + 1.0 - tube, not vis)
            if th is not None and 25 < dxy < 240:
                Tf = dxy / (ms["speed"] * math.cos(th))
                mx, my = tx + tvx * Tf * p["lead"], ty + tvy * Tf * p["lead"]
                dm = math.hypot(mx - x, my - y)
                th2 = ballistic(ms["speed"], ms["gravity"], dm, tz + 1.0 - tube, not vis)
                if th2 is not None:
                    want = (math.atan2(my - y, mx - x), th2)
                    use = "mortar"
                    fire1 = abs(wrap(yaw - want[0])) < 0.04 and abs(pitch - want[1]) < 0.03
        elif k1 == "laser" and dxy < w1["stats"]["range"] and vis:
            want = (d_yaw, d_pitch)
            use = "laser"
            full = w1["stats"]["charge"]
            if w1.get("charge", 0) <= 0:
                fire1 = w1["cooldown"] <= 0.05
            else:
                ready = w1["charge"] >= full * p["laser_release"] - 0.05
                aimed = abs(wrap(yaw - d_yaw)) < tol and abs(pitch - d_pitch) < tol
                fire1 = not (ready and aimed)
        elif k1 == "laser" and w1.get("charge", 0) > 0:
            fire1 = True  # keep the charge; it auto-fires after hold_s
        elif k1 == "lightning":
            ls = w1["stats"]
            fire1 = vis and dxy < ls["range"] * 0.95 and abs(wrap(yaw - d_yaw)) < math.radians(ls["cone_deg"]) * 0.4
        err_y = abs(wrap(yaw - want[0]))
        err_p = abs(pitch - want[1])
        fire0 = False
        if gun and vis and use != "mortar":
            fire0 = err_y < tol and err_p < tol * 1.5 and dxy < gun["stats"]["range"] * p["fire_range"]
        return want[0], want[1], fire0, fire1

    def shell_threat(self, projectiles, x, y, z):
        """Landing point of a visible mortar shell that will burst near us within dodge_shell_t, else None."""
        lim = self.p["dodge_shell_t"]
        if lim <= 0:
            return None
        ws = self.cfg["weapons"]["mortar"]
        g, reach = ws["gravity"], ws["splash_radius"] + 3.5
        for pr in projectiles:
            if pr["kind"] != "mortar":
                continue
            px, py, pz = pr["pos"]
            vx, vy, vz = pr["vel"]
            t = 0.0
            while t < lim:
                t += 0.1
                qx, qy, qz = px + vx * t, py + vy * t, pz + vz * t - 0.5 * g * t * t
                if qz <= self.world.h(qx, qy) or (abs(qx - x) < 3 and abs(qy - y) < 3 and z <= qz <= z + 10):
                    if math.hypot(qx - x, qy - y) < reach:
                        return qx, qy
                    break
        return None

    def steer(self, x, y, z, mvx, mvy, m, st):
        W = self.world
        ux, uy = mvx / m, mvy / m
        su = st["step_up"]
        if W.clear(x, y, z, ux, uy, 6.0, su):
            return mvx, mvy, False
        for k in (1, 2, 3, 4, 5):
            for sgn in (self.detour, -self.detour):
                ang = sgn * k * 0.5236
                c, s = math.cos(ang), math.sin(ang)
                rx, ry = ux * c - uy * s, ux * s + uy * c
                if W.clear(x, y, z, rx, ry, 6.0, su):
                    self.detour = sgn
                    return rx * m, ry * m, False
        return mvx, mvy, True

    def flee_cover(self, x, y, z, t, threats, nc, nr, margin, st):
        """Escape direction scored against every threat: away from them, toward points none of them can see.
        Kept for 0.6 s so the bot commits to one shelter instead of zig-zagging into fire."""
        p, W = self.p, self.world
        su = st["step_up"]
        until, cached = self.flee_cache
        if cached is not None and t < until and W.clear(x, y, z, cached[0], cached[1], 6.0, su):
            return cached
        ax = ay = 0.0
        for tx, ty, _, _ in threats:
            d2 = max(25.0, (x - tx) ** 2 + (y - ty) ** 2)
            ax, ay = ax + (x - tx) / d2, ay + (y - ty) / d2
        away = math.atan2(ay, ax)
        angs = [away + k * TAU / 16 for k in range(16)]
        pts = [(x + math.cos(a) * 25, y + math.sin(a) * 25) for a in angs]
        P = np.array([(px, py, W.h(px, py) + 4.5) for px, py in pts])
        E = np.array([(tx, ty, tz + 8.0) for tx, ty, tz, _ in threats])
        seen = W.sees_many(np.repeat(E, len(P), 0), np.tile(P, (len(E), 1)), 2.0).reshape(len(E), len(P))
        cover = 1.0 - seen.mean(0)
        best, bdir = -1e9, (math.cos(away), math.sin(away))
        for k, a in enumerate(angs):
            c, s = math.cos(a), math.sin(a)
            sc = math.cos(a - away) + p["cover"] * 1.5 * cover[k]
            if math.hypot(pts[k][0] - nc[0], pts[k][1] - nc[1]) > nr - margin:
                sc -= 1.0
            if not W.clear(x, y, z, c, s, 6.0, su):
                sc -= 2.0
            if sc > best:
                best, bdir = sc, (c, s)
        self.flee_cache = (t + 0.6, bdir)
        return bdir

    def flee_dir(self, x, y, z, tgt, nc, nr, margin, st):
        p, W = self.p, self.world
        tx, ty, tz = tgt["pos"]
        away = math.atan2(y - ty, x - tx)
        best, bdir = -1e9, (math.cos(away), math.sin(away))
        for k in range(8):
            ang = away + (k - 3.5) * 0.4
            c, s = math.cos(ang), math.sin(ang)
            px, py = x + c * 20, y + s * 20
            sc = math.cos(ang - away)
            if p["cover"] > 0 and not W.sees(tx, ty, tz + 8, px, py, W.h(px, py) + 4.5, 2.0):
                sc += p["cover"] * 1.5
            if math.hypot(px - nc[0], py - nc[1]) > nr - margin:
                sc -= 1.0
            if not W.clear(x, y, z, c, s, 6.0, st["step_up"]):
                sc -= 2.0
            if sc > best:
                best, bdir = sc, (c, s)
        return bdir
