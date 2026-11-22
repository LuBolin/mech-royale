"""Parameterised battle-royale bot. Every behaviour knob comes from params.py (the genome)."""
import math
import random

import numpy as np

import params as PR
from arena_routes import load_routes
from flight import plan_flight
from encounters import Encounter
from positioning import choose_position
from rotation import departure_slack
from tracking import clear_mortar_arc, exit_aim
from world import get_world

TAU = 2 * math.pi
LOUD = ("gun", "mortar", "explosion", "laser", "lightning", "laser_charge")
HEAL_HP = 0.85  # after contact breaks, hide until regen brings hp back to this fraction
GHOST_R = {"lightning": 35.0, "gun": 90.0, "laser": 150.0, "mortar": 140.0}  # typical range per weapon (Vivek)
EV_DIRS = np.array([(0.0, 0.0)] + [(math.cos(i * math.pi / 8), math.sin(i * math.pi / 8)) for i in range(16)])
EV_DT, EV_K, EV_PROG = 0.05, 20, 9  # evade_move: tick, simulated ticks (1 s), progress/LOS tick (0.5 s)


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
        self.combat_pilot = p["combat_pilot"] >= 0.5
        self.encounter = (Encounter() if self.combat_pilot and p["encounter_tactics"] >= 0.5
                          and p.get("survival", 1) >= 0.5 else None)
        self.routes = load_routes(self.world) if self.combat_pilot else None
        self.route = []
        self.route_key = None
        self.route_until = -1
        self.flight = None
        self.flight_started = -1
        self.flight_goal = None
        self.air_retry = -1
        self.takeoff_approach = None
        self.last_contact = None
        self.heard_sector = None
        # The board replays fixed seeds, so the same code scores the same; salt draws a fresh sample of our own dice.
        self.rng = random.Random(hello.get("seed", 0) + int(p["salt"]))
        self.phase = (hello.get("slot", 0) * 2.39996) % TAU  # spreads explore and calm-ring choices across slots
        self.seen, self.seen_at = set(), -1.0  # explore: 28 m grid cells our view has swept
        self.explore_pt, self.explore_at = None, -1.0
        self.calm_last = None  # crowd_avoid: last ring point chosen
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
        self.lose_at = -1e9  # last t the lose rule judged the fight lost
        self.on = {k: p[k] >= 0.5 for k in ("sidestep", "hide", "hostile_only", "mortar_laser", "ffa_detect",
                                             "fire_early", "laser_smart", "ghost_fire", "pincer", "berserk",
                                             "tgt_hpd", "fight_loot")}
        self.econ = p.get("econ_v2", 0) >= 0.5
        self.loot3 = p.get("loot_v3", 0) >= 0.5
        self.npc_hit = {}  # drone id -> last t it damaged us (loot_v3)
        self.cube_mem = {}  # cube id -> (pos, t last seen) (loot_v3)
        self.hostile = {}  # mech id -> last t it hit us, aimed at us or came close
        self.loot = {}  # pickup id -> (kind, pos, t last seen)
        self.cur_target = None
        self.hunt = p.get("hunt_v2", 0) >= 0.5
        self.hit_by = {}  # mech id -> last t it damaged us (hunt_v2)
        self.stick = (None, -1e9)  # (id, last t) of the hunt_v2 target
        self.laser_seen = -1e9  # last t a visible mech held a laser
        self.ffa = False  # latched once the lobby looks like a free-for-all
        self.alive_prev = None
        self.foreign_deaths = 0
        self.beam_fights = 0  # beams we saw land on a mech other than us
        self.busy = {}  # mech id -> last t it was seen fighting someone other than us
        self.ghost = None  # (t, x, y, z) estimated position of an unseen shooter
        self.ffa_ret = False  # ffa_profile: retreating to cover until healed
        self.danger_t = -1e9  # ffa_profile: last t a threat was in sight or hit us

    # ------------------------------------------------------------------ main
    def act(self, obs):
        p, W, rng = self.p, self.world, self.rng
        t = obs["t"]
        opening = t < p["quiet_open"]  # top practice bots loot nearby and fight only close or answering fire
        me = obs["self"]
        x, y, z = me["pos"]
        st = me["stats"]
        hpf = me["hp"] / max(1.0, me["max_hp"])
        energy = me["energy"]
        weapons = me["weapons"]
        kinds = [w["kind"] if w else None for w in weapons]
        self.takeoff_approach = None
        a = {"tick": obs["tick"]}

        mechs = [m for m in obs["mechs"] if m["kind"] == "mech"]
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
        vis_ids = {m["id"] for m in mechs}
        kills = 0
        for e in obs["events"]:
            if e["type"] == "kill":
                kills += 1
            if self.loot3 and e["type"] == "damaged" and e.get("weapon") == "npc":
                self.npc_hit[e.get("by")] = t  # a drone is not a mech threat: shoot it, do not fight a ghost
                continue
            if e["type"] == "damaged" and e.get("bearing") is not None:
                self.threat = (t, e["bearing"], e.get("by"))
                if e.get("by") is not None:
                    self.attackers[e["by"]] = (t, e["bearing"])
                    self.hostile[e["by"]] = t
                    self.hit_by[e["by"]] = t
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
            if p["ffa_profile"] >= 0.5:
                # azing's 1 death before 35 s. 24 v2 practice seeds: first foreign death >= 147 s; beams between
                # others are rarely visible. ffa_hit_s > 0 also latches on any hit before that time: the sim's
                # practice hunters first hit at 21.7 s, but the real hunters' timing is unknown, so it is off by default.
                self.ffa = self.ffa or (self.foreign_deaths >= 1 and t < 35) or 0 < self.last_hit < p["ffa_hit_s"]
        for s in obs["sounds"]:
            if s["kind"] in LOUD and s["dist"] != "far":
                self.sound = (t - s["age"], s["bearing"])
                if self.routes and (self.heard_sector is None or t-s["age"] > self.heard_sector[0]):
                    # Anchor the noisy bearing to where it was heard, even as our mech moves.
                    distance = 35 if s["dist"] == "near" else 100
                    sector = []
                    for offset in (-0.18, 0.0, 0.18):
                        bearing = s["bearing"] + offset
                        sx, sy = x + distance*math.cos(bearing), y + distance*math.sin(bearing)
                        sector.append((sx, sy, W.h(sx, sy) + 7))
                    self.heard_sector = (t-s["age"], sector)
        duel = self.encounter.observe(obs, mechs, W) if self.encounter else ()
        if duel:
            for mid in duel:
                self.busy[mid] = t
        if obs["offer"]:
            ch = obs["offer"]["choices"]
            a["pick"] = max(range(len(ch)), key=lambda i: p.get("up_" + ch[i]["id"], 0.0))

        hit_recent = self.threat is not None and t - self.threat[0] < 2.5

        # ---- hostility: most practice house bots ignore us until hunting, so leave passive mechs alone
        closing, engaged = {}, set()
        late = obs["zone"]["next_radius"] <= 70 or obs["alive"] <= 3
        prof = self.ffa and p["ffa_profile"] >= 0.5
        for m in mechs:
            if "laser" in m.get("weapons", ()):
                self.laser_seen = t
            mx_, my_ = m["pos"][0] - x, m["pos"][1] - y
            d = math.hypot(mx_, my_)
            off = abs(wrap(m["aim"][0] - math.atan2(-my_, -mx_)))
            if d < 45 or (d < 220 and off < (0.25 if m.get("charging") else 0.12)):
                self.hostile[m["id"]] = t
            if self.hunt:
                # closing speed: relative velocity along the line from it to us
                closing[m["id"]] = -((m["vel"][0] - me["vel"][0]) * mx_ + (m["vel"][1] - me["vel"][1]) * my_) / max(d, 1e-3)
                if prof:
                    # giddy's ffa (Ghost) rule: being looked at is not being attacked
                    if (d < 60 or t - self.hit_by.get(m["id"], -1e9) < 4 or (m.get("charging") and off < 0.22 and d < 120)
                            or (m.get("hp", 1400) < 350 and d < 120) or obs["alive"] <= 2):
                        engaged.add(m["id"])
                elif (d < 60 or t - self.hit_by.get(m["id"], -1e9) < 6 or late or (closing[m["id"]] > 3 and d < 180)
                        or (m.get("hp", 1400) < 0.4 * m.get("max_hp", 1400) and d < 160) or (off < 0.22 and d < 190)) and (
                        not opening or d < 60 or t - self.hit_by.get(m["id"], -1e9) < 6):
                    engaged.add(m["id"])

        # lasers do +60% to mortar carriers and cook the mortar off: shun it while a laser is around
        wpref = dict(self.wpref)
        if self.on["mortar_laser"] and t - self.laser_seen < 20:
            wpref["mortar"] = -1.0
        # weapon memory: keep pickups after they leave view; forget one that should be in sight but is gone
        seen_ids = set()
        for pk in obs["pickups"]:
            self.loot[pk["id"]] = (pk["kind"], pk["pos"], t)
            seen_ids.add(pk["id"])
        for pid, (_, (px, py, pz), ts) in list(self.loot.items()):
            if pid not in seen_ids:
                nearby = math.hypot(px - x, py - y) < 40
                # The deck hides its pickups from the mech underneath. Keep their
                # last sighting while walking out to reach the launch side.
                missing = nearby and (not self.routes or W.los(x, y, z+st["eye_height"], px, py, pz+0.5))
                if t - ts > p["loot_mem_s"] or missing:
                    del self.loot[pid]

        # loot_v3: remember cubes for 20 s; a kill's pile (2+ cubes within 7 m) is worth a longer walk
        cubes, cube_far = obs["cubes"], set()
        if self.loot3:
            for c in obs["cubes"]:
                self.cube_mem[c["id"]] = (c["pos"], t)
            shown = {c["id"] for c in obs["cubes"]}
            for cid, (cp, ts) in list(self.cube_mem.items()):
                gone = cid not in shown and math.hypot(cp[0] - x, cp[1] - y) < 30 and                     W.los(x, y, z + st["eye_height"], cp[0], cp[1], cp[2] + 0.5)
                if t - ts > 20 or gone:
                    del self.cube_mem[cid]
            cubes = [{"id": cid, "pos": cp} for cid, (cp, _) in self.cube_mem.items()]
            cube_far = {c["id"] for c in cubes for o in cubes if o is not c and math.dist(c["pos"], o["pos"]) < 7}

        def hostile(m):
            if m["id"] in duel:
                return True
            if self.combat_pilot and obs["alive"] == 2:
                return True
            if self.hunt:
                return m["id"] in engaged
            return not self.on["hostile_only"] or t - self.hostile.get(m["id"], -1e9) < 12 or (
                self.ffa and t - self.busy.get(m["id"], -1e9) < 2)

        # ---- target
        target, best = None, 1e18
        for m in mechs:
            d = math.hypot(m["pos"][0] - x, m["pos"][1] - y)
            if d > (240 if self.hunt else p["engage_range"]) or not hostile(m):
                continue
            if self.hunt:
                hp = m.get("hp", 1400)
                s = d + p["hunt_hpw"] * hp - (200 if hp <= 0.3 * m.get("max_hp", 1400) else 0)
                if p["laser_pulses"] >= 0.5 and kinds[1] == "laser":
                    # a placement comes from a counter reaching zero: prefer the mech the fewest full beams finish
                    ls = weapons[1]["stats"]
                    beam = ls["damage"] * (1 + 0.04 * min(me["cubes"], 10)) * (
                        1 + ls.get("cookoff", 0.0) * ("mortar" in m.get("weapons", ())))
                    s += 300 * math.ceil(hp / max(1.0, beam))
                s -= (200 if m["id"] == self.stick[0] and t - self.stick[1] < 2 else 0)
                s -= (40 if t - self.hit_by.get(m["id"], -1e9) < 4 else 0) + (15 if closing[m["id"]] > 3 else 0)
                mx, my, mz = m["pos"]
                # the reference skips mechs out of sight; keep them only as a last resort so we still close in
                if not W.los(x, y, z + st["eye_height"], mx, my, mz + 4.5 * (1 + 0.05 * m.get("cubes", 0))):
                    s += 1e6
            elif self.on["tgt_hpd"]:
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
        if self.hunt and target and best < 1e5:
            self.stick = (target["id"], t)
        # loot_v3: with no mech in sight, shoot a drone that hit us lately or one close enough to open fire
        drone = None
        if self.loot3 and (target is None or best >= 1e5):
            dbest = 1e18
            for n in npcs:
                nx, ny, nz = n["pos"]
                d = math.hypot(nx - x, ny - y)
                hit = t - self.npc_hit.get(n["id"], -1e9) < 3
                if (hit and d < 150 or target is None and d < 50) and d - 100 * hit < dbest and                         W.los(x, y, z + st["muzzle_height"], nx, ny, nz + 1.75):
                    dbest, drone = d - 100 * hit, n
        # free fire: practice squad members that are not hunting do not shoot back, so any visible one is a
        # target we can hit without changing course (the top board bots fire at every mech in range).
        # Dealing damage pauses our own regen, so a hurt mech that nobody is hitting heals instead.
        free = None
        # Laser only: it is hitscan at any range, and the 6.29 reference never opens on a non-hunter without one.
        if (p["free_fire"] >= 0.5 and kinds[1] == "laser" and (target is None or best >= 1e5) and not self.ffa
                and (hpf > 0.6 or t - self.last_hit < 5)):
            laser = weapons[1]["stats"]
            shot = laser["damage"] * (1 + 0.04 * min(me["cubes"], 10))
            fbest = None
            for m in mechs:
                mx, my, mz = m["pos"]
                d = math.hypot(mx - x, my - y)
                if d < laser["range"] and W.los(
                        x, y, z + st["eye_height"], mx, my, mz + 4.5 * (1 + 0.05 * m.get("cubes", 0))):
                    key = (math.ceil(m.get("hp", 1400) / shot), d)  # fewest shots left to a kill, then nearest
                    if fbest is None or key < fbest:
                        fbest, free = key, m
        if self.routes and target:
            tx, ty, tz = target["pos"]
            self.last_contact = (t, (tx, ty, tz + 4.5 * (1 + 0.05 * target.get("cubes", 0))))

        if p["explore"] >= 0.5 and t < 125:
            self.mark_seen(t, x, y, me["aim"][0])

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
        if self.encounter:
            zone_urgent = departure_slack(W, me, zone, t, target, margin) <= 0

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
        retreat = False
        if berserk and p["lose_rule"] >= 0.5:
            # Top practice bot's rule: leave only while clearly losing, and only 1.5 s past the last losing tick.
            near = sum(1 for m in mechs if math.hypot(m["pos"][0] - x, m["pos"][1] - y) < 160)
            losing = (near >= 2 and hpf < 0.55) or (near >= 3 and hpf < 0.8)
            closing_kill = False
            if target is not None:
                tf = target.get("hp", 1400) / max(1.0, target.get("max_hp", 1400))
                losing = (losing or (hpf < 0.35 and tf > hpf + 0.1)) and not (tf < 0.25 and hpf > 0.2 and near <= 2)
                closing_kill = math.hypot(target["pos"][0] - x, target["pos"][1] - y) < 120 and (
                    target.get("hp", 1400) < 200 or target.get("hp", 1400) < 0.5 * me["hp"])
            late = nr <= 70 or obs["alive"] <= 3
            if losing:
                self.lose_at = t
            retreat = t - self.lose_at < 1.5 and not closing_kill and not late
        if self.survival:
            if threats:
                # judge the fight against everyone involved, not one target; fight back 1-on-1 only
                n = len(threats)
                fight = target is not None and hpf >= p["flee_hp"] and n <= p["max_enemies"] and (
                    strength * aggr >= sum(th[3] for th in threats)
                    or (hit_recent and self.threat[2] == target["id"]))
                # berserk: practice hunters share vision and house bots only die to us, so running never pays
                mode = "flee" if retreat else "fight" if fight or (finish and gap_dir is None) or berserk else "flee"
            elif (self.on["hide"] or prof) and hpf < HEAL_HP and t - self.last_hit < 45:
                mode = "hide"  # contact broken: stand still (silent) until regen refills us
        elif target is not None:
            near = sum(1 for m in mechs if math.hypot(m["pos"][0] - x, m["pos"][1] - y) < p["engage_range"])
            fight = hpf >= p["flee_hp"] and strength * aggr >= target.get("hp", 1400) and near <= p["max_enemies"]
            mode = "fight" if fight or (hit_recent and self.threat[2] == target["id"] and hpf >= p["flee_hp"]) else "flee"
        if gap_dir is not None and mode in ("fight", "flee") and not berserk:
            mode = "flee"
        if prof:
            # giddy's crowd-hide retreat: leave when swamped, outgunned by a laser or low; come back healed
            if threats or t - self.last_hit < 3:
                self.danger_t = t
            crowd = sum(1 for m in mechs if m["id"] in engaged or math.hypot(m["pos"][0] - x, m["pos"][1] - y) < 150)
            tdist = math.hypot(target["pos"][0] - x, target["pos"][1] - y) if target else 1e9
            outgunned = target is not None and "laser" in target.get("weapons", ()) and "laser" not in kinds \
                and tdist > 55 and hpf < 0.8
            swamped = crowd >= 3 or (crowd == 2 and hpf < 0.65) or outgunned
            if (hpf < 0.38 and t - self.danger_t < 2) or swamped:
                self.ffa_ret = True
            elif self.ffa_ret and (hpf > 0.8 or (t - self.danger_t > 7 and hpf > 0.55)):
                self.ffa_ret = False
            finishing = target is not None and tdist < 120 and (
                target.get("hp", 1400) < 200 or target.get("hp", 1400) < 0.5 * me["hp"])
            if self.ffa_ret and not finishing and obs["alive"] > 2:
                if not threats and swamped:  # crowded by mechs not yet on us: still break their sight lines
                    threats = [(*m["pos"], m.get("hp", 1400)) for m in mechs
                               if math.hypot(m["pos"][0] - x, m["pos"][1] - y) < 150]
                mode = "flee" if threats else "hide"
        # early zone damage is a fraction of one gun's dps: finish breaking contact before running for the zone
        # the lose-rule retreat yields to the zone, as in the reference bot
        flee_first = self.survival and mode == "flee" and zone["dps"] < 50 and me["hp"] > 10 * zone["dps"] and not retreat
        contacts = [m for m in mechs if hostile(m)] if self.encounter else []
        tactic = self.encounter.choose(obs, W, target, contacts, duel, self.attackers) if self.encounter else None
        if tactic == "retreat":
            mode = "flee" if threats else "hide"
        elif tactic == "recover":
            mode = "hide"
        elif tactic in ("angle", "screen"):
            mode = "watch"
        elif tactic == "engage":
            mode = "fight"
        if zone_urgent and (self.encounter or not flee_first):
            mode = "zone"
        self.mode = mode

        mvx = mvy = 0.0
        boost = jump = False
        in_drop = False  # fight_drop: standing in a supply ring; evasion must not walk us out of it
        shoot_obj = None
        goal_z = None
        self.goal = None
        local_peek = False
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
            elif self.econ and target:
                # stay outside an enemy lightning's reach and a mortar's minimum-arc zone
                tw = target.get("weapons", ())
                pref = max(pref, 64 if "lightning" in tw else 0, 80 if "mortar" in tw else 0)
            if self.hunt and target and target.get("hp", 1400) < 0.38 * target.get("max_hp", 1400) and hpf > 0.42 and (
                    sum(1 for m in mechs if m["id"] in engaged or math.hypot(m["pos"][0] - x, m["pos"][1] - y) < 150) <= 2):
                # finisher: close in, but stay out of mortar splash and lightning reach as the reference does
                pref = 20.0
                if "lightning" not in kinds:
                    pref = max(pref, 80.0 if "mortar" in target.get("weapons", ()) else
                               64.0 if "lightning" in target.get("weapons", ()) else 20.0)
            radial = clamp((d - pref) / 15.0, -1.0, 1.0)
            if t >= self.strafe_next:
                if rng.random() < 0.7:
                    self.strafe = -self.strafe
                self.strafe_next = t + p["strafe_period"] * (0.5 + rng.random())
            s = p["strafe"] * self.strafe
            if self.combat_pilot:
                if target and "lightning" in target.get("weapons", ()) and "lightning" not in kinds:
                    radial = min(radial, clamp((d - 60) / 10, -1, 1))
                s = self.strafe
            mvx, mvy = vx_ * radial - vy_ * s, vy_ * radial + vx_ * s
            if p["score_moves"] >= 0.5 and target is not None and me["grounded"]:
                mvx, mvy = self.fight_direction(me, target, mechs, pref, zone, (mvx, mvy))
            if self.encounter and target and me["grounded"]:
                step = choose_position(self, obs, target, contacts, pref, (mvx, mvy))
                if step is not None:
                    mvx, mvy = step
                    local_peek = self._corner_shot
            if self.on["fight_loot"]:
                # keep shooting (aim is independent of movement) while walking over cubes or to a better weapon
                g = None
                cube_r = (70 if hpf > 0.5 else 40) if self.econ else 30
                for c in cubes:
                    dl = math.hypot(c["pos"][0] - x, c["pos"][1] - y)
                    if dl < (110 if c["id"] in cube_far and hpf > 0.3 else cube_r) and abs(c["pos"][2] - z) < 4 and (g is None or dl < g[0]):
                        g = (dl, c["pos"])
                if g is None:
                    cur = wpref[kinds[1]] + p["swap_margin"] if kinds[1] else 0.0
                    for k, ppos, _ in self.loot.values():
                        dl = math.hypot(ppos[0] - x, ppos[1] - y)
                        if k not in kinds and wpref[k] > cur and dl < p["fight_loot_dist"] and (
                                math.hypot(ppos[0] - nc[0], ppos[1] - nc[1]) < nr) and (g is None or dl < g[0]):
                            g = (dl, ppos)
                if g is not None and (not local_peek or g[0] <= self.reach):
                    mvx, mvy = g[1][0] - x, g[1][1] - y
            if p["fight_drop"] >= 0.5:
                # a supply drop is full hp, full energy and 3 cubes: walk to it mid-fight (aim is independent)
                # and hold still in the ring; the reach grows as we get hurt. The clock needs us alone in it.
                for dr in obs["drops"]:
                    px, py, _ = dr["pos"]
                    dl = math.hypot(px - x, py - y)
                    if (dr["holder"] in (None, me["id"]) and dl < 120 + 200 * (1 - hpf)
                            and dr["lands_in"] < dl / walk + 8
                            and all(math.hypot(m["pos"][0] - px, m["pos"][1] - py) > dr["radius"] + 6 for m in mechs)):
                        in_drop = dl < 0.6 * dr["radius"]
                        mvx, mvy = (0.0, 0.0) if in_drop else (px - x, py - y)
                        break
            if self.combat_pilot and target and threats and weapons[0] and weapons[0]["reload"] > 0.5 and not kinds[1]:
                mvx, mvy = self.flee_cover(x, y, z, t, threats, nc, nr, margin, st)
        elif mode == "flee":
            if gap_dir is not None:
                mvx, mvy = gap_dir
            elif self.survival:
                mvx, mvy = self.flee_cover(x, y, z, t, threats, nc, nr, margin, st)
            else:
                mvx, mvy = self.flee_dir(x, y, z, target, nc, nr, margin, st)
            boost = energy > p["boost_energy"]
        elif mode in ("hide", "watch"):
            pass  # stay put; regen needs 5 s without damage
        else:
            goal, gkind, gobj, gbest = None, None, None, 1e18
            safe_r = nr - margin * 0.5

            def safe(px, py, d):
                return math.hypot(px - nc[0], py - nc[1]) < safe_r or tts > p["zone_lead"] + d / walk + 5

            cur_pref = wpref[kinds[1]] + p["swap_margin"] if kinds[1] else 0.0
            reach = max(p["loot_dist"], p["loot_dist_empty"]) if kinds[1] is None else p["loot_dist"]
            near = 60.0 if opening else 1e9
            for pid, (k, ppos, _) in self.loot.items():
                if k in kinds:
                    continue
                gain = wpref[k] - cur_pref
                if gain <= 0:
                    continue
                px, py, pz = ppos
                pk = {"id": pid, "kind": k, "pos": ppos}
                d = math.hypot(px - x, py - y) + 2 * max(0.0, pz - z)
                if d < min(near, 240 if self.econ and k == "laser" else reach) and safe(px, py, d):
                    sc = d / (0.2 + gain)
                    if self.econ:  # econ scores are (d + 18) / value, lower is better
                        sc = (d + 18) / max(150 if k == "laser" else 0, 130 if kinds[1] is None else 60)
                    if sc < gbest:
                        gbest, goal, gkind, gobj = sc, (px, py, pz), "loot", pk
            for dr in obs["drops"]:
                px, py, pz = dr["pos"]
                d = math.hypot(px - x, py - y)
                contested = any(math.hypot(m["pos"][0] - px, m["pos"][1] - py) < p["drop_contest"] for m in mechs)
                far = 200 * (1 - hpf) if p["fight_drop"] >= 0.5 else 0  # a drop refills hp: worth a longer walk when hurt
                if d < p["drop_dist"] + far and dr["holder"] in (None, me["id"]) and dr["lands_in"] < d / walk + 8 \
                        and not contested:
                    sc = (d + 18) / 260 if self.econ else d * 0.7
                    if sc < gbest:
                        gbest, goal, gkind, gobj = sc, (px, py, pz), "drop", dr
            capped = me["cubes"] >= self.cfg["cubes"].get("stat_cap", 10)
            for c in cubes:
                px, py, pz = c["pos"]
                d = math.hypot(px - x, py - y) + 2 * max(0.0, pz - z)
                sc = (d + 18) / (25 if capped else 75) if self.econ else d * 0.8
                if p["cube_cluster"] >= 0.5 and not capped:
                    # a pile is collected in one visit, and the cube that reaches an upgrade threshold buys a pick
                    gain = sum(1 for o in cubes if math.dist(o["pos"], c["pos"]) < self.cfg["cubes"]["pickup_radius"] + 3)
                    nxt = next((k for k in self.cfg["cubes"]["upgrade_thresholds"] if k > me["cubes"]), 1e9)
                    sc /= 1 + 0.4 * (gain - 1) + 0.3 * (me["cubes"] + gain >= nxt)
                if d < (160 if c["id"] in cube_far else min(near, p["cube_dist"])) and safe(px, py, d) and sc < gbest:
                    gbest, goal, gkind, gobj = sc, (px, py, pz), "cube", c
            gun = weapons[0] if kinds[0] == "gun" else None
            if gun and (gun["ammo"] > 0 or gun["reload"] > 0) and (hpf > 0.5 or not self.econ):
                for c in obs["crates"]:
                    px, py, pz = c["pos"]
                    d = math.hypot(px - x, py - y)
                    sc = (d + 18) / (0.6 * (25 if capped else 75)) if self.econ else d + 15
                    if d < min(near, p["crate_dist"]) and safe(px, py, d) and sc < gbest:
                        gbest, goal, gkind, gobj = sc, (px, py, pz), "crate", c
            if hpf > p["npc_hp"] and not (self.econ and capped):
                for n in npcs:
                    px, py, pz = n["pos"]
                    d = math.hypot(px - x, py - y)
                    sc = (d + 18) / 85 if self.econ else d + 25
                    if d < min(near, p["npc_dist"]) and safe(px, py, d) and sc < gbest:
                        gbest, goal, gkind, gobj = sc, (px, py, pz), "npc", n
            # explore: without a laser in the opening, sweep unseen ground for one instead of idling or farming far away
            if (p["explore"] >= 0.5 and t < 125 and self.goal_hint is None and kinds[1] != "laser" and hpf > 0.52
                    and me["in_zone"] and dc < cr - 28
                    and (goal is None or (gkind in ("crate", "npc") and math.hypot(goal[0] - x, goal[1] - y) > 24))
                    and not any(k == "laser" and t - ts < 9 and math.hypot(pp[0] - x, pp[1] - y) < 125
                                for k, pp, ts in self.loot.values())):
                ex = self.explore_goal(t, x, y, zone)
                if ex is not None:
                    goal, gkind, gobj = (ex[0], ex[1], z), "explore", None
            if self.goal_hint is not None:
                goal, gkind = (self.goal_hint[0], self.goal_hint[1], z), "hint"
            if goal is None:
                goal, gkind = (home[0], home[1], z), "home"
                if p["crowd_avoid"] >= 0.5 and mechs:
                    cx_, cy_ = self.calm_point(t, x, y, nc, nr, mechs)
                    goal = (cx_, cy_, z)
                if (opening or p["idle_still"] >= 0.5) and dn < nr - margin:
                    goal = (x, y, z)  # nothing to do inside the next circle: hold still (silent) where we are
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
                scale = 1.0 if gkind in ("drop", "hint", "explore") else 1.0 - 0.6 * p["quiet"]
                if gkind == "home" and d < 6:
                    scale = min(scale, 0.35)
                if t < p["open_boost_s"] and not opening:
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
        if in_drop:
            mode = self.mode = "capture"
            boost = False

        if self.flight is not None and me["grounded"] and t > self.flight_started:
            # An early collision can land us on a different floor. Re-anchor from measured feet.
            landed = (math.dist(me["pos"][:2], self.flight_goal[:2]) <= max(2, st["radius"]) and
                      abs(z-self.flight_goal[2]) <= st["step_up"] + 0.2)
            self.end_flight(t, landed)
        if self.routes and self.flight is None and me["grounded"] and mode != "capture":
            purpose, sources, destination = None, [], home
            recent_contact = self.last_contact and t - self.last_contact[0] < 2.0
            reloading = weapons[0] and weapons[0]["reload"] > 0.5 and not kinds[1]
            if mode != "zone":
                if mode == "watch" and target and len(duel) == 2:
                    purpose = "opportunity"
                    for m in sorted((m for m in contacts if m["id"] in duel), key=lambda m: m is not target):
                        tx, ty, tz = m["pos"]
                        sources.append((tx, ty, tz + 4.5*(1+0.05*m.get("cubes",0))))
                elif mode == "flee" or (mode == "fight" and reloading):
                    purpose = "shelter"
                    sources = [(tx, ty, tz+7) for tx, ty, tz, _ in threats]
                    sources.sort(key=lambda pos: math.hypot(pos[0]-x, pos[1]-y))
                    sources = sources[:3]
                    if sources and self.heard_sector and t-self.heard_sector[0] < 1.0:
                        sx, sy, sz = self.heard_sector[1][1]
                        heard = math.atan2(sy-y, sx-x)
                        if all(abs(wrap(heard-math.atan2(ty-y, tx-x))) > 0.45 for tx, ty, _ in sources):
                            sources.append((sx, sy, sz))
                elif not local_peek and recent_contact and mode in ("fight", "roam") and hpf > 0.55:
                    contact = self.last_contact[1]
                    if not W.los(x, y, z + st["muzzle_height"], *contact):
                        purpose, sources = "flank", [contact]
                if not sources and target is None and self.heard_sector and t-self.heard_sector[0] < 1.5:
                    if mode in ("hide", "flee") or (mode == "roam" and self.goal and self.goal[3] == "home"):
                        purpose, sources = "shelter", self.heard_sector[1]
            if purpose is None and self.goal and mode in ("zone", "roam"):
                destination = self.goal[:3] if self.goal[3] in ("loot", "cube", "drop") else self.goal[:2]
                dx, dy = destination[0]-x, destination[1]-y
                dist = math.hypot(dx, dy)
                elevated = len(destination) == 3 and abs(destination[2]-z) > st["step_up"] + 1
                if mode == "zone" and self.encounter and dist > 12:
                    for m in sorted(contacts, key=lambda e: math.dist(me["pos"],e["pos"]))[:3]:
                        tx, ty, tz = m["pos"]
                        if math.hypot(tx-x, ty-y) < 240:
                            sources.append((tx, ty, tz + 7*(1+0.05*m.get("cubes",0))))
                timed_route = mode == "zone" and self.encounter is not None
                following = self.route_key == self.travel_key(destination, sources, timed_route)
                if sources or elevated or (dist > 20 and (following or
                        not W.clear(x, y, z, dx/dist, dy/dist, min(32, dist), st["step_up"], st["height"]))):
                    purpose = "travel"
            if purpose and (purpose == "travel" or sources):
                circle = (*nc, nr) if dn < nr and nr < cr else (*cc, cr)
                if purpose == "travel" and goal_z is not None and W.ceiling(x, y, z) < goal_z + st["height"]:
                    # Walking out from under a deck comes before gaining altitude.
                    goal_z = None
                budget = max(0, zone["shrink_end"]-t-1) if mode == "zone" and self.encounter else math.inf
                next_move = self.map_move(t, me, purpose, destination, sources, circle, budget)
                if next_move is not None:
                    speed = min(1, math.hypot(mvx, mvy)) if purpose == "travel" else 1
                    if purpose == "travel" and len(destination) == 3 and abs(destination[2]-z) > st["step_up"]+1:
                        speed = 1
                    # Follow the approach to the takeoff point instead of jetting at the loot overhead.
                    goal_z = None
                    mvx, mvy = next_move[0]*speed, next_move[1]*speed
                    if purpose != "travel":
                        mode = self.mode = purpose
                        boost = purpose == "shelter" and energy > p["boost_energy"] and next_move != (0, 0)
            else:
                self.route_key = None

        closing_coil = (self.encounter is not None and mode == "fight" and target is not None
                        and kinds[1] == "lightning" and "lightning" not in target["weapons"]
                        and math.dist(me["pos"], target["pos"]) > 0.9*weapons[1]["stats"]["range"]
                        and not (local_peek and kinds[0] == "gun"))
        if closing_coil:
            boost = me["grounded"] and energy > 3 * weapons[1]["stats"]["energy"]
        evading = False
        if (p["evade_v2"] >= 0.5 and me["grounded"] and self.flight is None and not closing_coil
                and mode not in ("zone", "capture")):
            ev = self.evade_move(obs, me, target, mechs, (mvx, mvy), mode == "fight")
            if ev is not None:
                (mvx, mvy), boost, threatened = ev
                evading = True
                if threatened:
                    mode = self.mode = "dodge"
        # ---- dodge incoming mortar shells (their path is visible and gravity is known)
        shell = None if evading else self.shell_threat(obs["projectiles"], x, y, z)
        if evading:
            pass
        elif shell is not None:
            lx, ly = shell
            d = math.hypot(x - lx, y - ly) or 1e-3
            mvx, mvy = (x - lx) / d, (y - ly) / d
            boost = energy > 10
            mode = self.mode = "dodge"
        elif self.on["sidestep"]:
            side = self.bullet_threat(obs["projectiles"], x, y, z, st["height"],
                                      me["vel"] if self.combat_pilot else (0, 0, 0))
            if side is not None:
                if closing_coil:
                    # Keep gaining range on a retreating rifle. Preserve the perpendicular
                    # component of the evasion without reversing back into the firing lane.
                    norm = max(0.001, math.hypot(mvx, mvy))
                    forward_x, forward_y = mvx/norm, mvy/norm
                    side_amount = -forward_y*side[0] + forward_x*side[1]
                    mvx = forward_x - 0.55*forward_y*side_amount
                    mvy = forward_y + 0.55*forward_x*side_amount
                else:
                    mvx, mvy = side
                if self.combat_pilot:
                    boost = me["grounded"] and energy > 8
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
                # strafe reversals in a fight look like being stuck; the top bots hop only when truly blocked
                jump = me["grounded"] and not (p["grounded"] >= 0.5 and mode in ("fight", "dodge"))
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
        if free is not None and drone is None:
            want_yaw, want_pitch, fire0, fire1 = self.shoot(me, free, kinds, weapons, x, y, z, muz, eye)
        elif drone is not None:  # shoot() aims 4.5 m above feet; a drone's centre is 1.75 m up
            nx, ny, nz = drone["pos"]
            want_yaw, want_pitch, fire0, fire1 = self.shoot(me, dict(drone, pos=(nx, ny, nz - 2.75)), kinds, weapons,
                                                            x, y, z, muz, eye)
        elif target is not None:
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
        elif mode == "flank" and self.last_contact:
            tx, ty, tz = self.last_contact[1]
            want_yaw = math.atan2(ty-y, tx-x)
            want_pitch = math.atan2(tz-muz, max(1, math.hypot(tx-x, ty-y)))
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
            if self.goal and self.goal[3] == "explore":
                want_yaw += 0.9 * math.sin(0.7 * t)  # sweep the view wedge across both sides of the path
        else:
            self.scan += p["scan_speed"] * 0.1
            want_yaw = yaw + p["scan_speed"] * 0.25
        if want_yaw is None:
            want_yaw = yaw

        # dodge: a visible laser charging at us, or fresh damage
        if tactic in ("recover","screen","angle") and mode != "zone":
            fire0 = False
            fire1 = bool(kinds[1] == "laser" and weapons[1].get("charge",0) > 0)
        if me["grounded"] and p["dodge"] > 0:
            danger = hit_recent and t - self.threat[0] < 0.3
            for m in mechs:
                if m.get("charging"):
                    b = math.atan2(y - m["pos"][1], x - m["pos"][0])
                    if abs(wrap(m["aim"][0] - b)) < 0.08:
                        danger = True
            if danger and not closing_coil and rng.random() < p["dodge"]:
                self.strafe = -self.strafe
                # evade_v2: a 1 s ballistic hop is the easiest lead shot (72% of expert rounds hit airborne us)
                if rng.random() < 0.5 and p["evade_v2"] < 0.5 and p["grounded"] < 0.5:
                    jump = True

        gun = weapons[0] if kinds[0] == "gun" else None
        if gun and target is None and gun["reload"] <= 0 and gun["ammo"] < gun["stats"]["mag"] * p["reload_frac"]:
            a["reload"] = True

        if (p["fight_boost"] >= 0.5 and p["grounded"] < 1.5 and mode == "fight" and me["grounded"] and not evading
                and not jump and m_ > 1e-3 and energy > p["boost_energy"]):
            boost = True
        # reference ThreatBoost: boost in any contact above 30 energy; lightning needs its energy reserve
        if (self.hunt and p["grounded"] < 1.5 and me["grounded"] and not jump and m_ > 1e-3 and energy > 30
                and (target is not None or t - self.last_hit < 2.5) and ("lightning" not in kinds or energy > 60)):
            boost = True

        if self.takeoff_approach is not None and mode not in ("dodge", "capture"):
            mvx, mvy = self.takeoff_approach
            boost = jump = False
        if self.flight is not None:
            # Enemy shells and bullets can interrupt a planned flight exactly as they interrupt a walk.
            if mode in ("dodge", "capture"):
                self.end_flight(t, False)
            else:
                command = self.flight_command(t, me)
                if command is not None:
                    (mvx, mvy), boost, jump = command
                    self.mode = "jet_route"
                    self.goal = (*self.flight_goal, "landing")
                    # Preserve the modeled speed and fuel through touchdown. A new laser
                    # charge halves travel speed; lightning and jets share energy.
                    if kinds[1] in ("laser", "lightning"):
                        fire1 = False

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
    def mark_seen(self, t, x, y, yaw):
        """explore: record 28 m cells our view swept (45 m all round, 120 m in the forward wedge), every 0.6 s."""
        if t - self.seen_at < 0.6:
            return
        self.seen_at = t
        i0, j0 = math.floor(x / 28.0), math.floor(y / 28.0)
        ux, uy = math.cos(yaw), math.sin(yaw)
        for i in range(i0 - 5, i0 + 6):
            for j in range(j0 - 5, j0 + 6):
                dx, dy = (i + 0.5) * 28.0 - x, (j + 0.5) * 28.0 - y
                r = math.hypot(dx, dy)
                if r < 45 or (r < 120 and dx * ux + dy * uy > 0.5 * r):
                    self.seen.add((i, j))

    def explore_goal(self, t, x, y, zone):
        """explore: a point 55 m out whose onward view covers the most unswept cells; kept 2.5 s or until reached."""
        pt = self.explore_pt
        if pt is not None and t - self.explore_at < 2.5 and math.hypot(pt[0] - x, pt[1] - y) > 7:
            return pt
        (cx, cy), cr = zone["center"], zone["radius"]
        (nx, ny), nr = zone["next_center"], zone["next_radius"]
        best, pick = -1e9, None
        for k in range(16):
            a = self.phase + k * TAU / 16
            dx, dy = math.cos(a), math.sin(a)
            px, py = x + 55 * dx, y + 55 * dy
            if math.hypot(px - cx, py - cy) > cr - 14:
                continue
            cells = {(math.floor((x + r * dx - s * dy) / 28.0), math.floor((y + r * dy + s * dx) / 28.0))
                     for r in (45, 75, 105, 135) for s in (-35, 0, 35)}
            score = len(cells - self.seen) - 0.015 * max(0.0, math.hypot(px - nx, py - ny) - nr)
            if pt is not None:
                score += 1 - 0.012 * math.hypot(px - pt[0], py - pt[1])
            if score > best:
                best, pick = score, (px, py)
        self.explore_pt, self.explore_at = pick, t
        return pick

    def calm_point(self, t, x, y, nc, nr, mechs):
        """crowd_avoid: of 10 slowly turning points on a ring inside the next circle, the one with least company."""
        R = min(0.62 * nr, 95.0)
        best, pick = 1e18, (nc[0], nc[1])
        for k in range(10):
            a = self.phase + t * 0.035 + k * TAU / 10
            px, py = nc[0] + R * math.cos(a), nc[1] + R * math.sin(a)
            crowd = sum(max(0.0, 72 - math.hypot(px - m["pos"][0], py - m["pos"][1])) for m in mechs)
            s = 0.46 * math.hypot(px - x, py - y) + 0.68 * crowd
            if self.calm_last and math.hypot(px - self.calm_last[0], py - self.calm_last[1]) < 5:
                s -= 8  # keep the same point unless another is clearly calmer
            if s < best:
                best, pick = s, (px, py)
        self.calm_last = pick
        return pick

    def end_flight(self, t, landed):
        self.flight = None
        self.route = []
        self.route_key = None
        self.route_until = t
        if not landed:
            self.air_retry = t + 1.5

    def flight_command(self, t, me):
        frame = round((t-self.flight_started) * 10)
        if frame >= len(self.flight.commands):
            # Contacts and terrain can shift touchdown by a physics step.
            if t-self.flight_started <= self.flight.airtime + 0.35:
                return (0, 0), False, False
            self.end_flight(t, False)
            return None
        if frame:
            expected = self.flight.positions[frame-1]
            if math.dist(me["pos"], expected) > max(3.0, me["stats"]["radius"]):
                self.end_flight(t, False)
                return None
        return self.flight.commands[frame]

    @staticmethod
    def travel_key(destination, sources, timed):
        # A timed zone crossing keeps its route while the goal and the watchers stay in the same cells.
        if not timed:
            return ("travel", destination)
        return ("travel", tuple(round(c/8) for c in destination),
                tuple((round(tx/20), round(ty/20)) for tx, ty, _ in sources))

    def map_move(self, t, me, purpose, destination, sources, circle, budget=math.inf):
        x, y, z = me["pos"]
        if purpose == "travel":
            key = self.travel_key(destination, sources, math.isfinite(budget))
        else:
            key = (purpose, tuple((round(tx/16), round(ty/16)) for tx, ty, _ in sources))
        if key != self.route_key or t >= self.route_until:
            self.route = self.routes.plan((x, y, z), destination, purpose, sources, circle, me["stats"],
                                          energy=me["energy"], allow_air=t >= self.air_retry, travel_budget=budget,
                                          tactical_flank=self.encounter is not None)
            self.route_key = key
            self.route_until = t + (3.0 if purpose == "travel" else 1.2)
        if not self.route:
            return None
        self.goal = (*self.route[-1][:3], purpose)
        while len(self.route) > 1 and not self.route[0].air and math.dist((x,y,z), self.route[0][:3]) < 2:
            if self.route[1].air:
                sx, sy, _ = self.route[0][:3]
                if math.hypot(sx-x, sy-y) > 0.65 or math.hypot(*me["vel"][:2]) > 3.0:
                    # Brake at the modeled launch point; sprinting past a deck lip
                    # changes which side of the wall the jump will intersect.
                    factor = 2.5 / me["stats"]["walk_speed"]
                    self.takeoff_approach = ((sx-x)*factor, (sy-y)*factor)
                    return self.takeoff_approach
            self.route.pop(0)
        waypoint = self.route[0]
        px, py, pz = waypoint[:3]
        if waypoint.air:
            # The grid uses a nominal launch. Simulate again from the observed position,
            # speed, energy and actual mech size before committing to a jump.
            charging = any(w and w.get("charge", 0) > 0 for w in me["weapons"])
            if not charging:
                self.flight = plan_flight(self.world, me["pos"], waypoint[:3], me["stats"],
                                          me["energy"], velocity=me["vel"])
            if self.flight is None:
                self.end_flight(t, False)
                return None
            self.flight_started = t
            self.flight_goal = waypoint[:3]
            return (0, 0)
        dist = math.hypot(px-x, py-y)
        if abs(pz-z) > dist + me["stats"]["step_up"]:
            self.route_until = t
            return None
        if dist < 2 or dist > 18:
            self.route_until = t if purpose == "travel" or dist > 18 else self.route_until
            return (0, 0) if purpose == "shelter" and dist < 2 else None
        return (px-x)/dist, (py-y)/dist

    def shoot(self, me, target, kinds, weapons, x, y, z, muz, eye):
        p, W = self.p, self.world
        yaw, pitch = me["aim"]
        tx, ty, tz = target["pos"]
        tvx, tvy, tvz = target["vel"]
        size = 1 + 0.05 * target.get("cubes", 0)
        tcz = tz + 4.5 * size
        dxy = max(1.0, math.hypot(tx - x, ty - y))
        pilot = p.get("combat_pilot", 0) >= 0.5
        tactical_aim = pilot and p.get("encounter_tactics", 0) >= 0.5
        vis = W.los(x, y, muz, tx, ty, tcz, 1.0) if tactical_aim else W.los(x, y, eye, tx, ty, tcz)
        tol = math.atan2(2.5 * size * p["aim_tol"], dxy)
        gun = weapons[0] if kinds[0] == "gun" else None
        gs = gun["stats"]["speed"] if gun else 252.0
        T = dxy / gs
        lx, ly, lz = tx + tvx * T * p["lead"], ty + tvy * T * p["lead"], tcz + tvz * T * p["lead"] * 0.5
        if pilot:
            # Movement and torso slew precede discharge in the physics step.
            dt = 0.05
            vx, vy, vz = me["vel"]
            x, y, muz = x + vx * dt, y + vy * dt, muz + vz * dt
            tx, ty, tz = tx + tvx * dt, ty + tvy * dt, tz + tvz * dt
            tcz = tz + 4.5 * size
            dxy = max(0.1, math.hypot(tx - x, ty - y))
            flight = math.hypot(dxy, tcz - muz) / gs
            for _ in range(2):
                lx, ly = tx + tvx * flight, ty + tvy * flight
                feet = tz + tvz * flight
                if abs(tvz) > 1 and not target.get("jetting"):
                    feet -= 0.5 * me["stats"]["gravity"] * flight ** 2
                floor = W.floor(lx, ly, tz, me["stats"]["step_up"])
                lz = max(feet, floor) + 4.5 * size if floor <= tz + me["stats"]["step_up"] else feet + 4.5 * size
                flight = math.dist((lx, ly, lz), (x, y, muz)) / gs
            tol = math.atan2(2.5 * size * p["aim_tol"], dxy)
        g_yaw = math.atan2(ly - y, lx - x)
        g_pitch = math.atan2(lz - muz, math.hypot(lx - x, ly - y))
        if tactical_aim and not vis and not (kinds[1] == "mortar" and weapons[1]["cooldown"] < 0.3):
            # A held laser releases on trigger-up, so retain an existing charge.
            charging = kinds[1] == "laser" and weapons[1].get("charge", 0) > 0
            speed = math.hypot(tvx, tvy)
            if gun and not charging and speed > 3 and flight < 0.45 and abs(tvz) < 2:
                # A round can leave just before the enemy does. Require both
                # a clear projectile ray and an unobstructed target movement path.
                st = me["stats"]
                if (W.los(x, y, muz, lx, ly, lz, 1.0)
                        and W.clear(*target["pos"], tvx/speed, tvy/speed,
                                    speed*(dt+flight), st["step_up"], 2*(tcz-tz))):
                    aligned = (abs(wrap(yaw-g_yaw)) < tol + st["turn_rate"]*dt
                               and abs(pitch-g_pitch) < tol + st["pitch_rate"]*dt
                               and st["min_pitch"] <= g_pitch <= st["max_pitch"]
                               and gs * flight < gun["stats"]["range"] * p["fire_range"])
                    return g_yaw, g_pitch, aligned, False
            edge = exit_aim(W, (x,y,muz), target, me["stats"])
            return edge[0], edge[1], False, charging
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
            if th is not None and (ms["splash_radius"] if pilot else 25) < dxy < 240:
                Tf = dxy / (ms["speed"] * math.cos(th))
                mx, my = tx + tvx * Tf * p["lead"], ty + tvy * Tf * p["lead"]
                dm = math.hypot(mx - x, my - y)
                impact_z = tz + 1.0
                if pilot:
                    floor = W.floor(mx, my, tz, me["stats"]["step_up"])
                    impact_z = max(floor, tz + tvz * Tf - 0.5 * me["stats"]["gravity"] * Tf**2) + 3 * size
                th2 = ballistic(ms["speed"], ms["gravity"], dm, impact_z - tube, not vis)
                if th2 is not None:
                    want = (math.atan2(my - y, mx - x), th2)
                    use = "mortar"
                    yaw_tol, pitch_tol = 0.04, 0.03
                    if pilot:
                        yaw_tol += me["stats"]["turn_rate"] * dt
                        pitch_tol += me["stats"]["pitch_rate"] * dt
                    fire1 = abs(wrap(yaw - want[0])) < yaw_tol and abs(pitch - want[1]) < pitch_tol
                    if tactical_aim and not vis:
                        legal = me["stats"]["min_pitch"] <= th2 <= me["stats"]["max_pitch"]
                        if not legal or not clear_mortar_arc(W, (x,y,tube), want[0], th2,
                                                            ms["speed"], ms["gravity"], dm):
                            use, fire1 = "gun", False
            if tactical_aim and not vis and use != "mortar":
                want = exit_aim(W, (x,y,muz), target, me["stats"])
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
                    if p["lethal_k"] * damage >= target["hp"] and W.los(x, y, muz, tx, ty, tcz):
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
            if tactical_aim:
                # The cone originates at the barrel after movement and torso slew.
                # Gun lead can share that cone; with an empty primary just track the body.
                st = me["stats"]
                if not gun:
                    want = (d_yaw, d_pitch)
                sy = yaw + clamp(wrap(want[0]-yaw), -st["turn_rate"]*dt, st["turn_rate"]*dt)
                sp = pitch + clamp(clamp(want[1],st["min_pitch"],st["max_pitch"])-pitch,
                                   -st["pitch_rate"]*dt, st["pitch_rate"]*dt)
                length = st["radius"] + 0.5
                ox = x + length*math.cos(sp)*math.cos(sy)
                oy = y + length*math.cos(sp)*math.sin(sy)
                oz = muz + length*math.sin(sp)
                fire1 = (math.dist((ox,oy,oz),(tx,ty,tcz)) <= ls["range"]
                         and abs(wrap(sy-math.atan2(ty-oy,tx-ox))) <= math.radians(ls["cone_deg"])/2
                         and W.los(x,y,muz,ox,oy,oz,0.5) and W.los(ox,oy,oz,tx,ty,tcz,1.0))
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

    def evade_move(self, obs, me, target, mechs, want, fighting):
        """((mx, my), boost, threatened) for the safest candidate move, or None when nothing threatens us.
        Simulates 1 s of our ground motion for 16 directions, standing and the wanted move, each with and
        without boost, and scores visible rounds and shells that would hit, the aim points of mechs facing
        us (our position now and its linear lead at their round's arrival), progress, zone and line of sight."""
        W, st = self.world, me["stats"]
        x, y, z = me["pos"]
        r, height = st["radius"], st["height"]
        p0 = np.array((x, y))
        # rounds that can reach us: coming our way, within 14 m of our line, not stopped behind us
        rounds = []
        for pr in obs["projectiles"]:
            if pr["kind"] == "mortar":
                continue
            px, py, pz = pr["pos"]
            vx, vy, vz = pr["vel"]
            sp = math.hypot(vx, vy)
            rx, ry = x - px, y - py
            if sp > 1 and (rx * vx + ry * vy) / sp > -3 and abs(rx * vy - ry * vx) / sp < 14 + r:
                rounds.append((px, py, pz, vx, vy, vz, 45.0 if pr["kind"] == "gun" else 25.0))
        shells = []
        ws = self.cfg["weapons"]["mortar"]
        tt = np.arange(1, 71) * 0.1
        for pr in obs["projectiles"]:
            if pr["kind"] != "mortar":
                continue
            (px, py, pz), (vx, vy, vz) = pr["pos"], pr["vel"]
            qx, qy, qz = px + vx * tt, py + vy * tt, pz + vz * tt - 0.5 * ws["gravity"] * tt * tt
            i = np.clip((qx - W.x0 + 0.5).astype(int), 0, W.nx - 1)
            j = np.clip((qy - W.y0 + 0.5).astype(int), 0, W.ny - 1)
            down = np.flatnonzero(qz <= W.top[j, i])
            if len(down):
                k = down[0]
                if math.hypot(qx[k] - x, qy[k] - y) < 50:
                    shells.append((qx[k], qy[k], qz[k], tt[k]))
        shooters = []
        for m in mechs:
            mx, my, mz = m["pos"]
            d = math.hypot(x - mx, y - my)
            if 1 < d < 230 and abs(wrap(m["aim"][0] - math.atan2(y - my, x - mx))) < 0.35 and W.los(
                    mx, my, mz + st["muzzle_height"], x, y, z + 0.5 * height):
                # round flight + one decision of latency, and the unit line of fire
                shooters.append((d / 252.0 + 0.1, (x - mx) / d, (y - my) / d))
        vis = {m["id"] for m in mechs}
        for aid, (ta, b) in self.attackers.items():  # unseen rifles that hit us lately: assume 80 m out
            if aid not in vis and obs["t"] - ta < 1.5:
                shooters.append((80 / 252.0 + 0.1, -math.cos(b), -math.sin(b)))
        # arc_avoid: lightning cannot miss inside its reach (240 dps), so stay out of it for every carrier in view
        arcs = []
        if self.p["arc_avoid"] >= 0.5 and not any(w and w["kind"] == "lightning" for w in me["weapons"]):
            reach = self.cfg["weapons"]["lightning"]["range"] + 12
            arcs = [m["pos"][:2] for m in mechs if "lightning" in m.get("weapons", ())
                    and math.hypot(m["pos"][0] - x, m["pos"][1] - y) < reach + 20]
        if not (rounds or shells or shooters or arcs or (fighting and target is not None)):
            return None
        # candidates
        wm = math.hypot(*want)
        wu = np.array(want) / wm if wm > 1e-6 else np.zeros(2)
        D = np.vstack([EV_DIRS, wu[None]]) if wm > 1e-6 else EV_DIRS
        n = len(D)
        slow = st["charge_move_mult"] if me["weapons"][1] and me["weapons"][1].get("charge", 0) > 0 else 1.0
        can_boost = me["energy"] > 12
        D2 = np.vstack([D, D[1:]]) if can_boost else D
        boost = np.arange(len(D2)) >= n
        speed = np.where(boost, st["boost_speed"], st["walk_speed"]) * slow
        T = D2 * speed[:, None]
        V = np.tile(np.array(me["vel"][:2], float), (len(D2), 1))
        acc = st["ground_accel"] * EV_DT
        P = np.empty((len(D2), EV_K, 2))
        pos = np.tile(p0, (len(D2), 1))
        for k in range(EV_K):
            dv = T - V
            dl = np.hypot(dv[:, 0], dv[:, 1])
            V += dv * np.minimum(1.0, acc / np.maximum(dl, 1e-9))[:, None]
            pos = pos + V * EV_DT
            P[:, k] = pos
        score = np.zeros(len(D2))
        threatened = False
        if rounds:
            R = np.array(rounds)
            ks = np.arange(EV_K + 1) * EV_DT
            B = R[:, None, :2] + R[:, None, 3:5] * ks[None, :, None]  # (b, K+1, 2)
            A, AB = B[:, :-1], B[:, 1:] - B[:, :-1]
            AP = P[:, None] - A[None]  # (c, b, K, 2)
            s = np.clip((AP * AB[None]).sum(-1) / np.maximum((AB * AB).sum(-1), 1e-9)[None], 0, 1)
            miss = np.hypot(*np.moveaxis(AP - s[..., None] * AB[None], -1, 0))
            bz = R[:, None, 2] + R[:, None, 5] * (ks[:-1] + 0.5 * EV_DT)[None]
            zok = (bz > z - 0.6) & (bz < z + height + 0.6)
            hit = ((miss < r + 0.85) & zok[None]).any(-1)  # (c, b)
            pen = hit @ R[:, 6]
            threatened = bool(pen.any())
            score -= pen
        for sx, sy, sz, ts in shells:
            k = min(EV_K, max(1, int(round(ts / EV_DT)))) - 1
            d = np.sqrt(((P[:, k] - (sx, sy)) ** 2).sum(-1) + (z + 0.5 * height - sz) ** 2)
            reach = ws["splash_radius"] * 1.3 + r
            pen = np.where(d < reach, ws["splash"] * (1 - 0.7 * d / reach) + 40, 0.0)
            threatened |= bool(pen.any())
            score -= pen
        v0 = np.array(me["vel"][:2])
        for tf, fx, fy in shooters:
            # Aimbots fire at our position or its linear lead; only the miss across the line of fire counts.
            k = min(EV_K, max(1, int(round(tf / EV_DT)))) - 1
            # Weights follow the sim's hits on us by house aim: full lead about 60%, half and none about 20% each.
            for lead, w in ((0.0, 6.0), (0.5, 6.0), (1.0, 18.0)):
                off = P[:, k] - (p0 + v0 * tf * lead)
                lat = np.abs(off[:, 0] * fy - off[:, 1] * fx)
                score -= w * np.exp(-(lat / 3.0) ** 2)
        # progress along the wanted move, zone, energy, heading
        prog = P[:, EV_PROG - 1]
        score += ((prog - p0) @ wu) * min(1.0, wm)
        zc, zr = obs["zone"]["center"], obs["zone"]["radius"]
        d_now = math.hypot(x - zc[0], y - zc[1]) - zr
        d_out = np.hypot(prog[:, 0] - zc[0], prog[:, 1] - zc[1]) - zr
        if d_now < -0.5:
            score -= np.where((d_out > -4) & (d_out > d_now), (d_out - d_now) * 8, 0.0)
        score -= np.where(boost, 1.5 if me["energy"] < self.p["boost_energy"] else 0.3, 0.0)
        for ax, ay in arcs:
            score -= 3.0 * np.maximum(0.0, reach - np.hypot(prog[:, 0] - ax, prog[:, 1] - ay))
        # Commit to a move for 0.3-0.6 s: every candidate assumes its command is held, and re-picking each
        # 0.1 s jitters around the straight line a lead shot aims at. Only rounds and shells outweigh this.
        last, until = getattr(self, "ev_last", ((0.0, 0.0), False)), getattr(self, "ev_until", -1.0)
        same = (D2 @ np.array(last[0]) > 0.98) & (boost == last[1])
        # The hold ends once we reach the commanded speed: constant velocity is what a lead shot predicts.
        accelerating = np.hypot(*(T - np.array(me["vel"][:2])).T) > 3.0
        # jink_hold > 0: hold each leg for its full time (Epic, 5,193 real rounds: 0.6-1.4 s boosted legs cut hits at
        # 78-150 m from 54% to 26%; legs under 0.6 s barely help because a reversal takes 0.4 s)
        jink = self.p["jink_hold"] > 0
        score += np.where(same, np.where((accelerating | jink) & (obs["t"] < until), 12.0, 0.25), 0.0)
        if target is not None:
            tx, ty, tz = target["pos"]
            a = np.column_stack([prog, np.full(len(D2), z + st["muzzle_height"])])
            b = np.tile((tx, ty, tz + 4.5 * (1 + 0.05 * target.get("cubes", 0))), (len(D2), 1))
            score -= 8.0 * ~W.los_many(a, b)
        # walls: take the best candidate whose first 6 m are walkable
        for c in np.argsort(-score)[:8]:  # boxed in: fall back to the old dodge and steering
            ux, uy = D2[c]
            if (ux == 0 and uy == 0) or W.clear(x, y, z, ux, uy, 6.0, st["step_up"], height):
                if not same[c] or obs["t"] >= until:
                    lo = self.p["jink_hold"] if jink else 0.3
                    self.ev_until = obs["t"] + lo + (1.33 * lo if jink else 0.3) * self.rng.random()
                self.ev_last = ((ux, uy), bool(boost[c]))
                return (float(ux), float(uy)), bool(boost[c]), threatened
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

    def bullet_threat(self, projectiles, x, y, z, height, velocity=(0, 0, 0)):
        """Unit move direction out of the path of a gun round passing within 3.5 m in the next 1.2 s, else None.
        jack (practice board): removing this dodge cost 0.5 points."""
        best = None
        for pr in projectiles:
            if pr["kind"] != "gun":
                continue
            px, py, pz = pr["pos"]
            vx, vy, vz = pr["vel"]
            vx, vy, vz = vx - velocity[0], vy - velocity[1], vz - velocity[2]
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
        def corridor(dx, dy, distance):
            if not self.combat_pilot:
                return W.clear(x, y, z, dx, dy, distance, su)
            radius = st["radius"]
            xmin, xmax, ymin, ymax = W.bounds
            ex, ey = x + dx * distance, y + dy * distance
            if not (xmin + radius < ex < xmax - radius and ymin + radius < ey < ymax - radius):
                return False
            return all(W.clear(x - dy * offset, y + dx * offset, z, dx, dy, distance, su, st["height"])
                       for offset in (0, radius * 0.85, -radius * 0.85))

        if corridor(ux, uy, 6.0):
            return mvx, mvy, False
        best, detour = -math.inf, None
        for k in (1, 2, 3, 4, 5):
            for sgn in (self.detour, -self.detour):
                ang = sgn * k * 0.5236
                c, s = math.cos(ang), math.sin(ang)
                rx, ry = ux * c - uy * s, ux * s + uy * c
                if corridor(rx, ry, 6.0):
                    if not self.combat_pilot:
                        self.detour = sgn
                        return rx * m, ry * m, False
                    score = c + 0.2 * (sgn == self.detour) + 0.7 * corridor(rx, ry, 12)
                    if score > best:
                        best, detour = score, (rx, ry, sgn)
        if detour:
            rx, ry, self.detour = detour
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
        if self.combat_pilot:
            solid_cover = ~W.los_many(np.repeat(E, len(P), 0), np.tile(P, (len(E), 1)), 2.0)
            cover += 4 * solid_cover.reshape(len(E), len(P)).mean(0)
        best, bdir = -1e9, (math.cos(away), math.sin(away))
        for k, a in enumerate(angs):
            c, s = math.cos(a), math.sin(a)
            sc = math.cos(a - away) + p["cover"] * 1.5 * cover[k]
            if self.combat_pilot and (W.blocked(*pts[k], z, st["height"], su)
                                      or not W.clear(x, y, z, c, s, 18.0, su, st["height"])):
                sc -= 5.0
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
