"""Parameterised battle-royale bot. Every behaviour knob comes from params.py (the genome)."""
import math
import random

import numpy as np

import params as PR
from world import get_world

TAU = 2 * math.pi
LOUD = ("gun", "mortar", "explosion", "laser", "lightning", "laser_charge")
HEAL_HP = 0.85  # after contact breaks, hide until regen brings hp back to this fraction
GHOST_R = {"lightning": 35.0, "gun": 90.0, "laser": 150.0, "mortar": 140.0}  # typical range per weapon (Vivek)


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
        self.goal = None  # (x, y, z, kind) the bot is walking to this tick; read by sim/viewer.py only
        self.capture_sample = None  # (drop id, progress) from the previous observation
        # Survival logic (threat list, unseen-fire response, hide-and-heal). House bots switch it off so the
        # simulator's opponents keep the logic that was calibrated against real practice scores.
        self.survival = p.get("survival", 1.0) >= 0.5
        self.attackers = {}  # attacker id -> (t, bearing) of its latest hit on us
        self.last_hit = -1e9
        self.flee_cache = (-1.0, None)  # (valid until t, (ux, uy))
        self.on = {k: p[k] >= 0.5 for k in ("sidestep", "hide", "hostile_only", "mortar_laser", "ffa_detect",
                                             "fire_early", "laser_smart", "ghost_fire", "pincer", "berserk",
                                             "tgt_hpd", "fight_loot")}
        self.hostile = {}  # mech id -> last t it hit us, aimed at us or came close
        self.loot = {}  # pickup id -> (kind, pos, t last seen)
        self.cur_target = None
        self.laser_seen = -1e9  # last t a visible mech held a laser
        self.ffa = False  # latched once the lobby looks like a free-for-all
        self.alive_prev = None
        self.foreign_deaths = 0
        self.beam_fights = 0  # beams we saw land on a mech other than us
        self.busy = {}  # mech id -> last t it was seen fighting someone other than us
        self.ghost = None  # (t, x, y, z) estimated position of an unseen shooter

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

        mechs = [m for m in obs["mechs"] if m["kind"] == "mech"]
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
        vis_ids = {m["id"] for m in mechs}
        kills = 0
        for e in obs["events"]:
            if e["type"] == "kill":
                kills += 1
            if e["type"] == "damaged" and e.get("bearing") is not None:
                self.threat = (t, e["bearing"], e.get("by"))
                if e.get("by") is not None:
                    self.attackers[e["by"]] = (t, e["bearing"])
                    self.hostile[e["by"]] = t
                    self.last_hit = t
                    if self.on["ghost_fire"] and e["by"] not in vis_ids:
                        R = GHOST_R.get(e.get("weapon"), 60.0)
                        self.ghost = (t, x + math.cos(e["bearing"]) * R, y + math.sin(e["bearing"]) * R, z)
        # beams: one landing on us locates its shooter exactly; one landing on another mech is someone else's fight
        for b in obs["beams"]:
            ox, oy, oz = b["path"][0]
            ex, ey, _ = b["path"][-1]
            if math.hypot(ex - x, ey - y) < 6:
                self.hostile[b["by"]] = t
                if self.on["ghost_fire"] and b["by"] not in vis_ids:
                    self.ghost = (t, ox, oy, oz - st["muzzle_height"])
                continue
            for m in mechs:
                if m["id"] != b["by"] and math.hypot(m["pos"][0] - ex, m["pos"][1] - ey) < 5:
                    self.busy[m["id"]] = self.busy[b["by"]] = t
                    self.beam_fights += 1
                    break
        # free-for-all: practice house bots never fight each other, so deaths we didn't cause before the zone
        # can kill anyone, or beams between other mechs, mean a real lobby
        if self.alive_prev is not None:
            self.foreign_deaths += max(0, self.alive_prev - obs["alive"] - kills)
        self.alive_prev = obs["alive"]
        if self.on["ffa_detect"] and not self.ffa:
            # sim: >= 2 before 150 s latched in 0/64 practice and 4/4 ffa matches; a single early death misfired
            self.ffa = (self.foreign_deaths >= 2 and t < 150) or self.beam_fights >= 2
        for s in obs["sounds"]:
            if s["kind"] in LOUD and s["dist"] != "far":
                self.sound = (t - s["age"], s["bearing"])
        if obs["offer"]:
            ch = obs["offer"]["choices"]
            a["pick"] = max(range(len(ch)), key=lambda i: p.get("up_" + ch[i]["id"], 0.0))

        hit_recent = self.threat is not None and t - self.threat[0] < 2.5

        # ---- hostility: most practice house bots ignore us until hunting, so leave passive mechs alone
        for m in mechs:
            if "laser" in m.get("weapons", ()):
                self.laser_seen = t
            mx_, my_ = m["pos"][0] - x, m["pos"][1] - y
            d = math.hypot(mx_, my_)
            off = abs(wrap(m["aim"][0] - math.atan2(-my_, -mx_)))
            if d < 45 or (d < 220 and off < (0.25 if m.get("charging") else 0.12)):
                self.hostile[m["id"]] = t

        # lasers do +60% to mortar carriers and cook the mortar off: shun it while a laser is around
        wpref = dict(self.wpref)
        if self.on["mortar_laser"] and t - self.laser_seen < 20:
            wpref["mortar"] = -1.0
        # weapon memory: keep pickups after they leave view; forget one that should be in sight but is gone
        seen_ids = set()
        for pk in obs["pickups"]:
            self.loot[pk["id"]] = (pk["kind"], pk["pos"], t)
            seen_ids.add(pk["id"])
        for pid, (_, (px, py, _), ts) in list(self.loot.items()):
            if pid not in seen_ids and (t - ts > p["loot_mem_s"] or math.hypot(px - x, py - y) < 40):
                del self.loot[pid]

        def hostile(m):
            return not self.on["hostile_only"] or t - self.hostile.get(m["id"], -1e9) < 12 or (
                self.ffa and t - self.busy.get(m["id"], -1e9) < 2)

        # ---- target
        target, best = None, 1e18
        for m in mechs:
            d = math.hypot(m["pos"][0] - x, m["pos"][1] - y)
            if d > p["engage_range"] or not hostile(m):
                continue
            if self.on["tgt_hpd"]:
                s = m.get("hp", 1400) / 1400.0 * (1 + d / 120.0)
                mx, my, mz = m["pos"]
                center_z = mz + 4.5 * (1 + 0.05 * m.get("cubes", 0))
                if not W.los(x, y, z + st["eye_height"], mx, my, center_z):
                    s += 1e6
                elif m["id"] == self.cur_target:
                    s *= 1 - p["hpd_keep_margin"]
            else:
                s = d / 100.0 + p["tgt_w_hp"] * m.get("hp", 1000) / max(1.0, m.get("max_hp", 1400))
                if hit_recent and self.threat[2] == m["id"]:
                    s -= p["tgt_w_threat"]
                if m["id"] == self.cur_target:
                    s -= p["stick"]
            if self.ffa and t - self.busy.get(m["id"], -1e9) < 2:
                s -= p["third_party"]
            if s < best:
                best, target = s, m
        self.cur_target = target["id"] if target else None

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
        if nr <= p["final_r"]:
            hr = 0.0  # final circles: the exact centre is the last safe point
        home = (nc[0] + ux * min(dn, hr), nc[1] + uy * min(dn, hr))

        # ---- threats: visible mechs in engage range plus anyone who hit us in the last 5 s, seen or not.
        # An unseen attacker is placed 60 m out along the bearing it hit us from.
        threats = []
        if self.survival:
            seen = set()
            for m in mechs:
                hit = self.attackers.get(m["id"])
                if (math.hypot(m["pos"][0] - x, m["pos"][1] - y) < p["engage_range"] and hostile(m)) or (
                        hit and t - hit[0] < 5):
                    threats.append((m["pos"][0], m["pos"][1], m["pos"][2], m.get("hp", 1400)))
                    seen.add(m["id"])
            for aid, (ta, b) in list(self.attackers.items()):
                if t - ta >= 5:
                    del self.attackers[aid]
                elif aid not in seen:
                    g = self.ghost if self.on["ghost_fire"] and self.ghost and self.ghost[0] >= ta else None
                    threats.append((g[1], g[2], g[3], 1400) if g else (x + math.cos(b) * 60, y + math.sin(b) * 60, z, 1400))

        # ---- surrounded: threats spread over more than 150 degrees; leave through the widest gap
        gap_dir = None
        if self.on["pincer"] and len(threats) >= 2:
            angs = sorted(math.atan2(ty_ - y, tx_ - x) for tx_, ty_, _, _ in threats)
            gaps = [(b - a, a) for a, b in zip(angs, angs[1:])] + [(angs[0] + TAU - angs[-1], angs[-1])]
            g, a0 = max(gaps)
            if g < TAU - math.radians(150):
                gap_dir = (math.cos(a0 + g / 2), math.sin(a0 + g / 2))

        # ---- mode
        mode = "roam"
        strength = me["hp"] * (1 + 0.04 * min(me["cubes"], 10)) * (1.3 if kinds[1] else 1.0)
        aggr = p["aggr"] * (p["late_aggr"] if obs["alive"] <= p["late_alive"] else 1.0)
        if self.ffa:
            aggr *= p["ffa_aggr"]
        # Trading health can buy placement in practice. In a free-for-all other mechs can eliminate each other.
        berserk = self.on["berserk"] and (not self.ffa or obs["alive"] <= 2)
        finish = target is not None and target.get("hp", 1400) < p["finish_hp"] * target.get("max_hp", 1400)
        if self.survival:
            if threats:
                # judge the fight against everyone involved, not one target; fight back 1-on-1 only
                n = len(threats)
                fight = target is not None and hpf >= p["flee_hp"] and n <= p["max_enemies"] and (
                    strength * aggr >= sum(th[3] for th in threats)
                    or (hit_recent and self.threat[2] == target["id"]))
                # berserk: practice hunters share vision and house bots only die to us, so running never pays
                mode = "fight" if fight or (finish and gap_dir is None) or berserk else "flee"
            elif self.on["hide"] and hpf < HEAL_HP and t - self.last_hit < 45:
                mode = "hide"  # contact broken: stand still (silent) until regen refills us
        elif target is not None:
            near = sum(1 for m in mechs if math.hypot(m["pos"][0] - x, m["pos"][1] - y) < p["engage_range"])
            fight = hpf >= p["flee_hp"] and strength * aggr >= target.get("hp", 1400) and near <= p["max_enemies"]
            mode = "fight" if fight or (hit_recent and self.threat[2] == target["id"] and hpf >= p["flee_hp"]) else "flee"
        if gap_dir is not None and mode in ("fight", "flee") and not berserk:
            mode = "flee"
        # early zone damage is a fraction of one gun's dps: finish breaking contact before running for the zone
        flee_first = self.survival and mode == "flee" and zone["dps"] < 50 and me["hp"] > 10 * zone["dps"]
        if zone_urgent and not flee_first:
            mode = "zone"
        self.mode = mode

        mvx = mvy = 0.0
        boost = jump = False
        shoot_obj = None
        goal_z = None
        self.goal = None
        if mode == "zone":
            self.goal = (home[0], home[1], z, "zone")
            mvx, mvy = home[0] - x, home[1] - y
            boost = (not me["in_zone"] or tts < need) and energy > p["boost_energy"] * 0.5
        elif mode == "fight":
            # berserk with no visible target: strafe against the unseen attacker's estimated position
            tx, ty = target["pos"][:2] if target else threats[0][:2]
            d = max(1e-3, math.hypot(tx - x, ty - y))
            vx_, vy_ = (tx - x) / d, (ty - y) / d
            pref = p["pref_dist"]
            if "laser" in kinds and p["laser_dist"] > 0:
                pref = p["laser_dist"]
            if "lightning" in kinds:
                pref = min(pref, p["lightning_dist"])
            radial = clamp((d - pref) / 15.0, -1.0, 1.0)
            if t >= self.strafe_next:
                if rng.random() < 0.7:
                    self.strafe = -self.strafe
                self.strafe_next = t + p["strafe_period"] * (0.5 + rng.random())
            s = p["strafe"] * self.strafe
            mvx, mvy = vx_ * radial - vy_ * s, vy_ * radial + vx_ * s
            if p["score_moves"] >= 0.5 and target is not None and me["grounded"]:
                mvx, mvy = self.fight_direction(me, target, mechs, pref, zone, (mvx, mvy))
            if self.on["fight_loot"]:
                # keep shooting (aim is independent of movement) while walking over cubes or to a better weapon
                g = None
                for c in obs["cubes"]:
                    dl = math.hypot(c["pos"][0] - x, c["pos"][1] - y)
                    if dl < 30 and abs(c["pos"][2] - z) < 4 and (g is None or dl < g[0]):
                        g = (dl, c["pos"])
                if g is None:
                    cur = wpref[kinds[1]] + p["swap_margin"] if kinds[1] else 0.0
                    for k, ppos, _ in self.loot.values():
                        dl = math.hypot(ppos[0] - x, ppos[1] - y)
                        if k not in kinds and wpref[k] > cur and dl < p["fight_loot_dist"] and (
                                math.hypot(ppos[0] - nc[0], ppos[1] - nc[1]) < nr) and (g is None or dl < g[0]):
                            g = (dl, ppos)
                if g is not None:
                    mvx, mvy = g[1][0] - x, g[1][1] - y
        elif mode == "flee":
            if gap_dir is not None:
                mvx, mvy = gap_dir
            elif self.survival:
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

            cur_pref = wpref[kinds[1]] + p["swap_margin"] if kinds[1] else 0.0
            reach = max(p["loot_dist"], p["loot_dist_empty"]) if kinds[1] is None else p["loot_dist"]
            for pid, (k, ppos, _) in self.loot.items():
                if k in kinds:
                    continue
                gain = wpref[k] - cur_pref
                if gain <= 0:
                    continue
                px, py, pz = ppos
                pk = {"id": pid, "kind": k, "pos": ppos}
                d = math.hypot(px - x, py - y) + 2 * max(0.0, pz - z)
                if d < reach and safe(px, py, d):
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
            self.goal = (gx, gy, gz, gkind)
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
                if t < p["open_boost_s"]:
                    scale, boost = 1.0, energy > 5  # race to the first loot before the squad wakes up
                mvx, mvy = mvx / m_ * scale, mvy / m_ * scale

        if p["finish_drop"] >= 0.5:
            drop = self.finishing_drop(obs, mechs)
            if (drop is not None and mode in ("fight", "roam") and not zone_urgent
                    and not (hit_recent and hpf < 0.25)):
                px, py, pz = drop["pos"]
                mode = self.mode = "capture"
                self.goal = (px, py, pz, "drop")
                goal_z = None
                boost = False
                if math.hypot(px - x, py - y) > 0.6 * drop["radius"]:
                    mvx, mvy = px - x, py - y
                else:
                    mvx = mvy = 0.0

        # ---- dodge incoming mortar shells (their path is visible and gravity is known)
        shell = self.shell_threat(obs["projectiles"], x, y, z)
        if shell is not None:
            lx, ly = shell
            d = math.hypot(x - lx, y - ly) or 1e-3
            mvx, mvy = (x - lx) / d, (y - ly) / d
            boost = energy > 10
            mode = self.mode = "dodge"
        elif self.on["sidestep"]:
            side = self.bullet_threat(obs["projectiles"], x, y, z, st["height"])
            if side is not None:
                mvx, mvy = side
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
        elif self.ghost is not None and t - self.ghost[0] < 2.5 and kinds[0] == "gun":
            # unseen shooter (behind foliage, outside the wedge): return fire at its estimated position
            _, gx, gy, gz = self.ghost
            d = max(1.0, math.hypot(gx - x, gy - y))
            want_yaw = math.atan2(gy - y, gx - x)
            want_pitch = math.atan2(gz + 4.5 - muz, d)
            tol = math.atan2(2.5 * p["aim_tol"], d)
            fire0 = d < weapons[0]["stats"]["range"] * p["fire_range"] and abs(wrap(yaw - want_yaw)) < tol \
                and abs(pitch - want_pitch) < tol * 1.5
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

        if (p["fight_boost"] >= 0.5 and mode == "fight" and me["grounded"]
                and not jump and m_ > 1e-3 and energy > p["boost_energy"]):
            boost = True

        # weapon pickup within reach
        for pk in obs["pickups"]:
            k = pk["kind"]
            if k in kinds or math.hypot(pk["pos"][0] - x, pk["pos"][1] - y) > self.reach - 0.3:
                continue
            if abs(pk["pos"][2] - z) > 3.8:
                continue
            if kinds[1] is None and wpref[k] > 0:
                a["take"] = pk["id"]
                break
            if kinds[1] and wpref[k] > wpref[kinds[1]] + p["swap_margin"]:
                if not (kinds[1] == "laser" and weapons[1].get("charge", 0) > 0):
                    a["take"] = pk["id"]
                    a["drop"] = 1
                    break
        if "take" not in a and kinds[1] == "mortar" and wpref["mortar"] < 0 and t - self.laser_seen < 0.15:
            a["drop"] = 1

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
        share_aim = k1 == "laser" and p.get("gun_laser_cooldown", 0) >= 0.5
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
            ls = w1["stats"]
            full = ls["charge"]
            if w1.get("charge", 0) <= 0:
                # firing is held until the weapon is ready, so asking one tick early loses nothing
                fire1 = w1["cooldown"] <= (0.1 + 1e-6 if self.on["fire_early"] else 0.05)
                if share_aim and not fire1 and gun and gun["ammo"] > 0 and gun["reload"] <= 0:
                    want = (g_yaw, g_pitch)
                    use = "gun"
            else:
                ready = w1["charge"] >= full * p["laser_release"] - 0.05
                aimed = abs(wrap(yaw - d_yaw)) < tol and abs(pitch - d_pitch) < tol
                if (p.get("laser_lethal", 0) >= 0.5 and not ready and aimed
                        and w1["charge"] >= ls["min_charge"] and target.get("hp") is not None
                        and me["stats"]["min_pitch"] <= d_pitch <= me["stats"]["max_pitch"]
                        and math.hypot(dxy, tcz - muz) < ls["range"]):
                    cubes = self.cfg["cubes"]
                    frac = ls["min_frac"] + (1 - ls["min_frac"]) * min(1.0, w1["charge"] / full)
                    damage = ls["damage"] * frac * (1 + cubes["damage_per_cube"] * min(me["cubes"], cubes["stat_cap"]))
                    if "mortar" in target.get("weapons", ()):
                        damage *= 1 + ls.get("cookoff", 0.0)
                    # Enemy armor is hidden. Discount the current charge rather than counting a future tick.
                    if 0.8 * damage >= target["hp"] and W.los(x, y, muz, tx, ty, tcz):
                        ready = True
                if self.on["laser_smart"] and w1["charge"] >= ls["min_charge"] - 1e-6:
                    # the shot leaves next tick: judge aim after one tick of torso slew toward the target's next spot
                    turn = me["stats"].get("turn_rate", 4.0) * 0.1
                    nx, ny = tx + tvx * 0.1, ty + tvy * 0.1
                    n_yaw = math.atan2(ny - y, nx - x)
                    yaw_n = yaw + clamp(wrap(n_yaw - yaw), -turn, turn)
                    pr = me["stats"].get("pitch_rate", 3.0) * 0.1
                    pitch_n = pitch + clamp(d_pitch - pitch, -pr, pr)
                    aimed = abs(wrap(yaw_n - n_yaw)) < tol and abs(pitch_n - d_pitch) < tol
                    frac = ls["min_frac"] + (1 - ls["min_frac"]) * min(1.0, (w1["charge"] + 0.1) / full)
                    dmg = ls["damage"] * frac * (1 + 0.04 * min(me["cubes"], 10))
                    if "mortar" in target.get("weapons", ()):
                        dmg *= 1 + ls.get("cookoff", 0.0)
                    kill = dmg >= 1.05 * target.get("hp", 1e9)
                    fade = not W.los(x, y, eye, tx + tvx * 0.4, ty + tvy * 0.4, tcz + tvz * 0.4)
                    ready = ready or kill or fade
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
            slew_s = p.get("gun_slew_s", 0.0)
            if slew_s > 0 or share_aim:
                # Both triggers share one torso. Only credit motion toward the commanded aim,
                # and keep the gun quiet when a different weapon needs an incompatible line.
                st = me["stats"]
                slew_s = clamp(slew_s, 0.0, 0.05)
                yaw_step = st["turn_rate"] * slew_s
                pitch_step = st["pitch_rate"] * slew_s
                next_yaw = yaw + clamp(wrap(want[0] - yaw), -yaw_step, yaw_step)
                command_pitch = clamp(want[1], st["min_pitch"], st["max_pitch"])
                next_pitch = pitch + clamp(command_pitch - pitch, -pitch_step, pitch_step)
                fire0 = (abs(wrap(want[0] - g_yaw)) < tol and abs(want[1] - g_pitch) < tol * 1.5
                         and st["min_pitch"] <= g_pitch <= st["max_pitch"]
                         and abs(wrap(next_yaw - g_yaw)) < tol and abs(next_pitch - g_pitch) < tol * 1.5
                         and math.dist((x, y, muz), (lx, ly, lz)) < gun["stats"]["range"] * p["fire_range"]
                         and W.los(x, y, muz, lx, ly, lz))
        return want[0], want[1], fire0, fire1

    def finishing_drop(self, obs, mechs):
        """Return a nearly completed ring while its clock advances and the area stays safe."""
        me = obs["self"]
        previous = self.capture_sample
        self.capture_sample = None
        for drop in obs["drops"]:
            if drop["holder"] != me["id"] or drop["lands_in"] > 0:
                continue
            progress = drop["progress"]
            self.capture_sample = (drop["id"], progress)
            remaining = drop["capture_s"] - progress
            if (previous is None or previous[0] != drop["id"] or progress <= previous[1]
                    or not 0 < remaining <= 1.2 + 1e-6):
                continue
            px, py, _ = drop["pos"]
            x, y, _ = me["pos"]
            radius = drop["radius"]
            if math.hypot(x - px, y - py) > radius:
                continue
            zone = obs["zone"]
            horizon = remaining + 0.1
            if math.dist((px, py), zone["center"]) + radius > zone["radius"]:
                continue
            if (obs["t"] + horizon >= zone["shrink_start"] and
                    math.dist((px, py), zone["next_center"]) + radius > zone["next_radius"]):
                continue
            contest_radius = radius + self.cfg["mech"]["boost_speed"] * horizon
            if all(math.hypot(m["pos"][0] - px, m["pos"][1] - py) > contest_radius for m in mechs):
                return drop
        return None

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

    def bullet_threat(self, projectiles, x, y, z, height):
        """Unit move direction out of the path of a gun round passing within 3.5 m in the next 1.2 s, else None.
        jack (practice board): removing this dodge cost 0.5 points."""
        best = None
        for pr in projectiles:
            if pr["kind"] != "gun":
                continue
            px, py, pz = pr["pos"]
            vx, vy, vz = pr["vel"]
            v2 = vx * vx + vy * vy
            if v2 < 1.0:
                continue
            tc = ((x - px) * vx + (y - py) * vy) / v2  # time of closest approach
            if not 0.0 < tc < 1.2:
                continue
            cx, cy, cz = px + vx * tc, py + vy * tc, pz + vz * tc
            miss = math.hypot(x - cx, y - cy)
            if miss < 3.5 and z - 1.0 <= cz <= z + height + 1.0 and (best is None or tc < best[0]):
                if miss > 0.2:
                    nx, ny = (x - cx) / miss, (y - cy) / miss
                else:  # dead centre: step to our current strafe side
                    s = math.sqrt(v2)
                    nx, ny = -vy / s * self.strafe, vx / s * self.strafe
                best = (tc, (nx, ny))
        return best[1] if best else None

    def fight_direction(self, me, target, mechs, preferred, zone, fallback):
        """Score eight short ground moves around the selected mech."""
        world, st = self.world, me["stats"]
        x, y, z = me["pos"]
        tx, ty, tz = target["pos"]
        dist = math.hypot(tx - x, ty - y)
        if dist < 0.01:
            return fallback
        rx, ry = (tx - x) / dist, (ty - y) / dist
        tx_side, ty_side = -ry * self.strafe, rx * self.strafe
        crossfire = sorted((m for m in mechs if m["id"] != target["id"]),
                           key=lambda m: math.hypot(m["pos"][0] - x, m["pos"][1] - y))[:2]
        crossfire = [m["pos"] for m in crossfire if math.hypot(m["pos"][0] - x, m["pos"][1] - y) < 230]
        target_z = tz + 4.5 * (1 + 0.05 * target.get("cubes", 0))
        cx, cy = zone["center"]
        best, direction = -math.inf, fallback
        diagonal = math.sqrt(0.5)
        for radial, side in ((0, 1), (0, -1), (diagonal, diagonal), (diagonal, -diagonal),
                             (-diagonal, diagonal), (-diagonal, -diagonal), (1, 0), (-1, 0)):
            ux, uy = radial * rx + side * tx_side, radial * ry + side * ty_side
            reach = 9.0
            if not world.clear(x, y, z, ux, uy, reach, st["step_up"], st["height"]):
                reach = 6.0  # the downstream steering probe uses the same clearance distance
                if not world.clear(x, y, z, ux, uy, reach, st["step_up"], st["height"]):
                    continue
            px, py = x + reach * ux, y + reach * uy
            if math.hypot(px - cx, py - cy) > zone["radius"]:
                continue
            floor = world.floor(px, py, z, st["step_up"])
            if floor < z - 4.5:  # stay on the current fighting level instead of stepping off a roof
                continue
            # Each metre of range correction earns a point. Favor strafing when already in range.
            score = abs(dist - preferred) - abs(math.hypot(tx - px, ty - py) - preferred)
            score += 2.0 * abs(side) + 0.5 * side + 0.3 * reach
            if score <= best:  # visibility checks below can only subtract points
                continue
            if not world.los(px, py, floor + st["muzzle_height"], tx, ty, target_z, 3.0):
                score -= 12.0
            for ex, ey, ez in crossfire:
                if score <= best:
                    break
                if world.los(ex, ey, ez + st["muzzle_height"], px, py, floor + 0.5 * st["height"], 3.0):
                    score -= 6.0
            if score > best:
                best, direction = score, (ux, uy)
        return direction

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
