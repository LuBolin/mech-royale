"""Headless approximate mechfight/1 match simulator for training. It is NOT the official server.

Built from protocol.md on the real arena (bot/world.py: buildings, bridge decks, trees, slopes).
Known gaps (shortcuts, upgrade if evolved bots behave oddly on the server):
- physics uses the kit terrain and prisms; item sample heights for loose loot are approximate;
- house bots are our own Bot with fixed params, not the real ones.
"""
import gzip
import json
import math
import os
import random
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "bot"))
from world import get_world  # noqa: E402  (grid: only used to list roof cells for loot spawns)
from geom import NOHIT, Geom  # noqa: E402

_GEOM = {}


def get_geom(map_dir, cfg):
    key = (map_dir, cfg["mech"]["step_up"], cfg["mech"]["max_slope_deg"])
    if key not in _GEOM:
        _GEOM[key] = Geom(map_dir, key[1], key[2])
    return _GEOM[key]

MAP_DIR = os.path.join(ROOT, "kit", "arena")
DT = 0.05
TAU = 2 * math.pi


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def load_config():
    with gzip.open(os.path.join(ROOT, "kit", "check", "stream.jsonl.gz"), "rt") as f:
        return json.loads(f.readline())["config"]


class Mech:
    kind = "mech"

    def __init__(self, i, x, y, z, yaw, cfg, upg, team):
        self.id, self.x, self.y, self.z = i, x, y, z
        self.vx = self.vy = self.vz = 0.0
        self.yaw, self.pitch, self.cyaw, self.cpitch = yaw, 0.0, yaw, 0.0
        self.cfg, self.upg, self.team = cfg, upg, team
        self.upc, self.upgrades, self.cubes = {}, [], 0
        self.max_hp = 0.0
        self.hp = 0.0
        self.recalc()
        self.hp = self.max_hp
        self.energy = self.stats["energy"]
        self.slots = [{"kind": "gun", "cooldown": 0.0, "ammo": int(self.wst["gun"]["mag"]), "reload": 0.0,
                       "charge": 0.0, "equip": 0.0}, None]
        self.grounded, self.alive, self.in_zone = True, True, True
        self.last_reply_tick = None
        self.last_dmg = self.last_spend = -99.0
        self.jetting = False
        self.short_until = -1.0
        self.events, self.mem = [], {}
        self.offer, self.pending, self.offer_seq = None, 0, 0
        self.move, self.boost, self.jump, self.fire, self.reload_req = (0.0, 0.0), False, False, [False, False], False
        self.step_acc = self.zone_acc = 0.0
        self.kills, self.place, self.bot, self.last_cont = 0, None, None, {}

    def recalc(self):
        base = dict(self.cfg["mech"])
        w = {k: dict(v) for k, v in self.cfg["weapons"].items()}
        for uid, n in self.upc.items():
            for e in self.upg[uid]["effects"]:
                path = e["path"].split(".")
                d = base if path[0] == "mech" else w[path[1]]
                for _ in range(n):
                    if "mul" in e:
                        d[path[-1]] *= e["mul"]
                    if "add" in e:
                        d[path[-1]] += e["add"]
        cu = self.cfg["cubes"]
        size = 1 + cu["size_per_cube"] * self.cubes
        for k in ("radius", "height", "eye_height", "muzzle_height"):
            base[k] *= size
        self.stats, self.wst, self.r, self.h = base, w, base["radius"], base["height"]
        old = self.max_hp
        capped = min(self.cubes, cu["stat_cap"])
        self.max_hp = base["hp"] * (1 + cu["hp_per_cube"] * capped)
        if old:
            self.hp += max(0.0, self.max_hp - old)
        self.dmg_mult = 1 + cu["damage_per_cube"] * capped

    def charging(self):
        return any(w and w["kind"] == "laser" and w["charge"] > 0 for w in self.slots)

    def kinds(self):
        return [w["kind"] for w in self.slots if w]


class Thing:
    """NPCs, crates, cubes, pickups, projectiles, drops: plain attribute bags."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class Match:
    def __init__(self, cfg, seed, factories, teams=None, track=None, map_dir=MAP_DIR, shared=None, levels=None):
        self.cfg, self.map_dir = cfg, map_dir
        self.W = get_world(map_dir)
        self.G = G = get_geom(map_dir, cfg)
        self.rng = rng = random.Random(seed)
        self.t, self.tick = 0.0, 0
        self.upg = {u["id"]: u for u in cfg["upgrades"]}
        self.nid = {"npc": 100, "crate": 1000, "cube": 5000, "pickup": 8000, "drop": 20000, "proj": 1}
        self.track, self.shared, self.levels = track, shared, levels or {}
        self.sounds, self.beams, self.projs, self.cubes, self.pickups, self.drops = [], [], [], [], [], []
        self.drop_i = 0
        self.errors = 0
        self._zone_setup()
        mc = cfg["match"]
        sc, sr = cfg["zone"]["start_center"], cfg["zone"]["start_radius"]
        pts = self._spawn_points(sc, sr, mc)
        self.mechs = [Mech(i, px, py, G.surface(px, py), rng.uniform(-math.pi, math.pi), cfg, self.upg,
                           teams[i] if teams else None) for i, (px, py) in enumerate(pts)]
        n = cfg["npc"]
        self.npcs = []
        for _ in range(int(n["count"])):
            px, py = self._open_point(sc[0], sc[1], sr * 0.9, 20)
            self.npcs.append(Thing(id=self._id("npc"), kind="npc", x=px, y=py, z=G.surface(px, py), vx=0.0, vy=0.0, vz=0.0,
                                   hx=px, hy=py, hp=n["hp"], max_hp=n["hp"], r=n["radius"], h=n["height"],
                                   alive=True, target=None, cool=0.0, gx=px, gy=py, gt=0.0, yaw=0.0, team=None,
                                   grounded=True))
        cr = cfg["crates"]
        self.crates = []
        for k in range(int(cr["count"])):
            px, py, pz = self._loot_point(k < cr["count"] * cr["roof_fraction"])
            self.crates.append(Thing(id=self._id("crate"), kind="crate", x=px, y=py, z=pz, hp=cr["hp"], r=cr["radius"],
                                     h=cr["height"], alive=True, team=None))
        lo = cfg["loot"]
        kinds = [k for k, v in lo["weights"].items() if v > 0]
        wts = [lo["weights"][k] for k in kinds]
        for k in range(int(lo["floor_count"])):
            px, py, pz = self._loot_point(k < lo["floor_count"] * lo["floor_roof_fraction"])
            self._add_pickup(rng.choices(kinds, wts)[0], px, py, pz)
        self.bots_state = []
        slot_rng = random.Random("mechfight/1 simulator slot seeds")
        for m, f in zip(self.mechs, factories):
            hello = {"type": "hello", "protocol": "mechfight/1", "slot": m.id, "seed": slot_rng.randrange(2 ** 32),
                     "map": {"dir": map_dir, "files": ["v2/world.json", "v2/terrain.i16"]}, "config": cfg}
            m.bot = f(hello)

    # ------------------------------------------------------------------ setup helpers
    def _id(self, k):
        self.nid[k] += 1
        return self.nid[k] - 1

    def _spawn_points(self, center, radius, match_cfg):
        radius *= match_cfg["spawn_zone_fraction"]
        separation = match_cfg["spawn_min_separation"]
        body = self.cfg["mech"]
        for _ in range(32):
            pts = []
            for _ in range(int(match_cfg["mechs"])):
                for _ in range(1000):
                    x, y = self._open_point(*center, radius, match_cfg["arena_margin"])
                    if all(math.hypot(x-px, y-py) >= separation for px, py in pts):
                        ground = self.G.ground(x, y)
                        if self.G.body_clear(x, y, ground, body["radius"], body["height"]):
                            pts.append((x, y))
                            break
                else:
                    break  # A crowded partial layout can leave no legal next seat.
            if len(pts) == match_cfg["mechs"]:
                return pts
        raise ValueError("Could not place all mechs with the configured spawn separation")

    def _open_point(self, cx, cy, r, margin=30):
        x0, x1, y0, y1 = self.G.bounds
        G = self.G
        for _ in range(2000):
            a = self.rng.uniform(0, TAU)
            rr = math.sqrt(self.rng.random()) * r
            px, py = cx + math.cos(a) * rr, cy + math.sin(a) * rr
            if x0 + margin < px < x1 - margin and y0 + margin < py < y1 - margin and \
                    G.surface(px, py) - G.ground(px, py) < 0.05:
                return px, py
        return cx, cy

    def _loot_point(self, roof):
        W, G = self.W, self.G
        if roof:
            if not hasattr(W, "_roofs"):
                d = (W.top - W.ground).ravel()
                W._roofs = np.flatnonzero((d > 3) & (d < 55))
            idx = int(W._roofs[self.rng.randrange(len(W._roofs))])
            px, py = W.x0 + idx % W.nx, W.y0 + idx // W.nx
            return px, py, G.surface(px, py)
        sc, sr = self.cfg["zone"]["start_center"], self.cfg["zone"]["start_radius"]
        px, py = self._open_point(sc[0], sc[1], sr * 0.95, 20)
        return px, py, G.surface(px, py)

    def _add_pickup(self, kind, x, y, z, state=None):
        self.pickups.append(Thing(id=self._id("pickup"), kind=kind, x=x, y=y, z=z, state=state))

    def _zone_setup(self):
        z = self.cfg["zone"]
        c, r, T = tuple(z["start_center"]), z["start_radius"], 0.0
        self.phases = []
        for ph in z["phases"]:
            r2 = ph["radius"]
            a = self.rng.uniform(0, TAU)
            rr = math.sqrt(self.rng.random()) * (r - r2)
            c2 = (c[0] + math.cos(a) * rr, c[1] + math.sin(a) * rr)
            self.phases.append((T + ph["hold_s"], T + ph["hold_s"] + ph["shrink_s"], c, r, c2, r2, ph["dps"]))
            T += ph["hold_s"] + ph["shrink_s"]
            c, r = c2, r2

    def zone_at(self, t):
        for ts, te, c0, r0, c1, r1, dps in self.phases:
            if t < te:
                if t < ts:
                    return c0, r0, (ts, te, c0, r0, c1, r1, dps)
                f = (t - ts) / (te - ts)
                return (c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f), r0 + (r1 - r0) * f, \
                    (ts, te, c0, r0, c1, r1, dps)
        ph = self.phases[-1]
        return ph[4], ph[5], ph

    # ------------------------------------------------------------------ loop
    def run(self, max_t=None):
        end = max_t or self.cfg["match"]["duration_s"]
        while self.t < end - 1e-9:
            if self.tick % 2 == 0:
                self.decide()
            self.step()
            alive = [m for m in self.mechs if m.alive]
            if len(alive) <= 1:
                break
            if self.track is not None and not self.mechs[self.track].alive:
                break
        alive = sorted((m for m in self.mechs if m.alive), key=lambda m: -m.hp)
        if self.track is None or self.mechs[self.track].alive:
            for k, m in enumerate(alive):
                m.place = k + 1
        return {"place": [m.place for m in self.mechs], "kills": [m.kills for m in self.mechs], "t": self.t,
                "errors": self.errors}

    def step(self):
        self._move()
        self._weapons()
        self._projectiles()
        if self.tick % 2 == 0:
            self._npcs_think()
            self._cubes_pickup()
            self._offers()
        self._npcs_move()
        self._zone_regen()
        self._drops_step()
        self._deaths()
        self.tick += 1
        self.t = self.tick * DT

    # ------------------------------------------------------------------ decisions
    def decide(self):
        t, cfg = self.t, self.cfg
        alive = [m for m in self.mechs if m.alive]
        actors = alive + [n for n in self.npcs if n.alive]
        vis = self._visibility(alive, actors)
        self.item_vis = self._item_visibility(alive)
        team_seen = {}
        for m in alive:
            if m.team is not None:
                team_seen.setdefault(m.team, set()).update(vis[m.id])
        if self.shared is not None:
            self._update_shared(alive, team_seen)
        zc, zr, ph = self.zone_at(t)
        zone = {"center": [zc[0], zc[1]], "radius": zr, "next_center": [ph[4][0], ph[4][1]], "next_radius": ph[5],
                "shrink_start": ph[0], "shrink_end": ph[1], "dps": ph[6]}
        drops = [{"id": d.id, "pos": [d.x, d.y, d.z], "radius": cfg["drops"]["radius"],
                  "lands_in": max(0.0, d.lands_at - t), "capture_s": cfg["drops"]["capture_s"],
                  "progress": d.progress, "holder": d.holder} for d in self.drops]
        actions = []
        for m in alive:
            if hasattr(m.bot, "send"):  # external bots get every obs first and think in parallel
                m.bot.send(self._observe(m, vis, team_seen, actors, zone, drops))
        for m in alive:
            try:
                act = m.bot.recv() if hasattr(m.bot, "send") else \
                    m.bot.act(self._observe(m, vis, team_seen, actors, zone, drops))
            except Exception:
                self.errors += 1
                if self.errors <= 3:
                    traceback.print_exc()
                act = None
            if isinstance(act, dict) and act.get("tick") == self.tick:
                m.last_reply_tick = self.tick
                m.last_cont = {k: act[k] for k in ("move", "aim", "boost") if k in act}
            elif m.last_reply_tick is not None and (self.tick - m.last_reply_tick) * DT <= cfg["bot"]["coast_s"]:
                act = m.last_cont
            else:
                act = {"aim": [m.yaw, m.pitch]}
            actions.append((m, act))
        # All observers see the same pre-action world. Resolve contested takes by distance.
        pickups = {p.id: p for p in self.pickups}
        takes = []
        for m, act in actions:
            pid = act.get("take")
            if pid is not None:
                pk = pickups.get(pid)
                distance = math.hypot(pk.x - m.x, pk.y - m.y) if pk else math.inf
                takes.append((distance, m.id, m, pid, act.get("drop")))
                act = {k: v for k, v in act.items() if k not in ("take", "drop")}
            self._apply(m, act)
        for _, _, m, pid, slot in sorted(takes):
            self._take(m, pid, slot)
        self.sounds, self.beams = [], []

    def _observe(self, m, vis, team_seen, actors, zone, drops):
        for aid in vis[m.id]:
            m.mem[aid] = self.t
        if m.team is not None:
            for aid in team_seen[m.team]:
                m.mem[aid] = self.t
        if m.zone_acc > 0:
            m.events.append({"type": "damaged", "amount": round(m.zone_acc, 1), "weapon": "zone",
                             "bearing": None, "by": None})
            m.zone_acc = 0.0
        obs = self._obs(m, actors, zone, drops)
        m.events = []  # obs keeps the old list
        return obs

    def _update_shared(self, alive, team_seen):
        sh = self.shared
        pl = self.mechs[sh["player"]]
        if pl.alive and any(pl.id in s for s in team_seen.values()):
            sh["last"] = (self.t, (pl.x, pl.y, pl.z))
        house = sorted((m for m in alive if m.team is not None), key=lambda m: self.levels.get(m.id, 0))
        n = len(house) if self.t >= sh["all_hunt_t"] else 1 + (15 - len(house))
        sh["hunters"] = {m.id for m in house[:n]}

    def _visibility(self, alive, actors):
        vc = self.cfg["vision"]
        half = math.radians(vc["fov_deg"]) / 2
        A, B, pairs, ignore_canopy = [], [], [], []
        for o in alive:
            st = o.stats
            ex, ey, ez = o.x, o.y, o.z + st["eye_height"]
            nr = vc["near_range"] + st["near_range_bonus"]
            rr = vc["range"] + st["vision_bonus"]
            nr2, rr2 = nr * nr, rr * rr
            for a in actors:
                if a is o:
                    continue
                dx, dy = a.x - o.x, a.y - o.y
                d2 = dx * dx + dy * dy
                if d2 <= nr2 or (d2 <= rr2 and abs(wrap(math.atan2(dy, dx) - o.yaw)) <= half):
                    pairs.append((o.id, a.id))
                    near = d2 <= nr2 or not vc["canopy_blocks"]
                    ignore_canopy.extend((near, near))
                    A.append((ex, ey, ez))
                    B.append((a.x, a.y, a.z + a.h * 0.5))
                    A.append((ex, ey, ez))
                    B.append((a.x, a.y, a.z + a.h))
        vis = {o.id: set() for o in alive}
        if pairs:
            ok = self.G.sight_many(np.array(A, float), np.array(B, float), np.array(ignore_canopy)).reshape(-1, 2).any(1)
            for (oid, aid), v in zip(pairs, ok):
                if v:
                    vis[oid].add(aid)
        return vis

    def _item_visibility(self, alive):
        """Per mech, the set of item ids (crates, cubes, pickups, projectiles) in view with line of sight."""
        vc = self.cfg["vision"]
        half = math.radians(vc["fov_deg"]) / 2
        items = [(("crate", c.id), c.x, c.y, c.z + offset)
                 for c in self.crates if c.alive for offset in (c.h / 2, c.h)]
        items += [(("cube", c.id), c.x, c.y, c.z + 0.3) for c in self.cubes]
        items += [(("pickup", p.id), p.x, p.y, p.z + 0.5) for p in self.pickups]
        items += [(("projectile", p.id), p.x, p.y, p.z) for p in self.projs]
        A, B, pairs, ignore_canopy = [], [], [], []
        for o in alive:
            st = o.stats
            ex, ey, ez = o.x, o.y, o.z + st["eye_height"]
            nr = vc["near_range"] + st["near_range_bonus"]
            rr = vc["range"] + st["vision_bonus"]
            nr2, rr2 = nr * nr, rr * rr
            for iid, x, y, z in items:
                dx, dy = x - o.x, y - o.y
                d2 = dx * dx + dy * dy
                if d2 <= nr2 or (d2 <= rr2 and abs(wrap(math.atan2(dy, dx) - o.yaw)) <= half):
                    pairs.append((o.id, iid))
                    ignore_canopy.append(d2 <= nr2 or not vc["canopy_blocks"])
                    A.append((ex, ey, ez))
                    B.append((x, y, z))
        vis = {o.id: set() for o in alive}
        if pairs:
            ok = self.G.sight_many(np.array(A, float), np.array(B, float), np.array(ignore_canopy))
            for (oid, iid), v in zip(pairs, ok):
                if v:
                    vis[oid].add(iid)
        return vis

    def _obs(self, m, actors, zone, drops):
        t, cfg = self.t, self.cfg
        vc, sc = cfg["vision"], cfg["sound"]
        mem_s = vc["memory_s"]
        st = m.stats
        nr = vc["near_range"] + st["near_range_bonus"]
        rr = vc["range"] + st["vision_bonus"]
        iv = self.item_vis.get(m.id, ())

        def inview(iid):
            return iid in iv

        seen = []
        for a in actors:
            if a is m or t - m.mem.get(a.id, -99.0) > mem_s:
                continue
            if a.kind == "mech":
                if m.team is not None and a.team == m.team:
                    continue
                seen.append({"id": a.id, "kind": "mech", "pos": [a.x, a.y, a.z], "vel": [a.vx, a.vy, a.vz],
                             "aim": [a.yaw, a.pitch], "hp": a.hp, "max_hp": a.max_hp, "cubes": a.cubes,
                             "weapons": a.kinds(), "charging": a.charging()})
            else:
                seen.append({"id": a.id, "kind": "npc", "pos": [a.x, a.y, a.z], "vel": [a.vx, a.vy, 0.0],
                             "aim": [a.yaw, 0.0], "hp": a.hp, "max_hp": a.max_hp})
        noise = math.radians(sc["bearing_noise_deg"]) * st["hearing"]
        sounds = []
        for kind, sx, sy, src, ts in self.sounds:
            if src == m.id:
                continue
            d = math.hypot(sx - m.x, sy - m.y)
            if d <= sc["range"][kind]:
                sounds.append({"kind": kind, "bearing": wrap(math.atan2(sy - m.y, sx - m.x) + self.rng.gauss(0, noise)),
                               "dist": "near" if d < sc["near"] else "mid" if d < sc["mid"] else "far",
                               "age": round(t - ts, 2)})
        return {
            "type": "obs", "tick": self.tick, "t": round(t, 2), "alive": sum(1 for q in self.mechs if q.alive),
            "self": {"id": m.id, "pos": [m.x, m.y, m.z], "vel": [m.vx, m.vy, m.vz], "aim": [m.yaw, m.pitch],
                     "hp": m.hp, "max_hp": m.max_hp, "energy": m.energy, "max_energy": st["energy"],
                     "grounded": m.grounded, "cubes": m.cubes, "upgrades": list(m.upgrades),
                     "weapons": [None if w is None else self._weapon_obs(m, w) for w in m.slots],
                     "in_zone": m.in_zone, "stats": st},
            "mechs": seen,
            "projectiles": [{"id": p.id, "kind": p.kind, "pos": [p.x, p.y, p.z], "vel": [p.vx, p.vy, p.vz]}
                            for p in self.projs if p.owner is not m and inview(("projectile", p.id))],
            "crates": [{"id": c.id, "pos": [c.x, c.y, c.z], "hp": c.hp} for c in self.crates
                       if c.alive and inview(("crate", c.id))],
            "cubes": [{"id": c.id, "pos": [c.x, c.y, c.z]} for c in self.cubes if inview(("cube", c.id))],
            "pickups": [{"id": p.id, "kind": p.kind, "pos": [p.x, p.y, p.z]} for p in self.pickups if inview(("pickup", p.id))],
            "sounds": sounds,
            "beams": [{"by": b[0], "kind": b[1], "path": b[2], "age": round(t - b[3], 2)} for b in self.beams
                      if b[0] != m.id and self._beam_visible(m, b[2], nr, rr)],
            "events": m.events, "drops": drops, "zone": zone, "offer": m.offer,
        }

    def _beam_visible(self, m, path, near, far):
        eye = (m.x, m.y, m.z + m.stats["eye_height"])
        half = math.radians(self.cfg["vision"]["fov_deg"]) / 2
        midpoints = (tuple((x + y) / 2 for x, y in zip(a, b)) for a, b in zip(path, path[1:]))
        for point in (*path, *midpoints):
            dx, dy = point[0] - m.x, point[1] - m.y
            d = math.hypot(dx, dy)
            if d <= near:
                if self.G.ray(eye, point, trunks=True) > 1:
                    return True
            elif d <= far and abs(wrap(math.atan2(dy, dx) - m.yaw)) <= half and self.G.sight(eye, point):
                return True
        return False

    @staticmethod
    def _weapon_obs(m, w):
        kind = w["kind"]
        obs = {"kind": kind, "stats": m.wst[kind],
               "cooldown": max(0.0, w["equip"], w["cooldown"])}
        if kind == "gun":
            obs.update(ammo=w["ammo"], reload=w["reload"])
        elif kind == "laser":
            obs["charge"] = min(w["charge"], m.wst[kind]["charge"])
        return obs

    def _apply(self, m, act):
        st = m.stats
        mv = act.get("move") or (0.0, 0.0)
        mx, my = float(mv[0]), float(mv[1])
        L = math.hypot(mx, my)
        if L > 1:
            mx, my = mx / L, my / L
        m.move = (mx, my)
        m.boost = bool(act.get("boost", False))
        if act.get("aim"):
            m.cyaw = float(act["aim"][0])
            m.cpitch = min(st["max_pitch"], max(st["min_pitch"], float(act["aim"][1])))
        m.jump = bool(act.get("jump", False))
        f = act.get("fire") or (False, False)
        m.fire = [bool(f[0]), bool(f[1]) if len(f) > 1 else False]
        m.reload_req = bool(act.get("reload", False))
        if m.offer is not None and act.get("pick") is not None:
            i = int(act["pick"])
            if 0 <= i < len(m.offer["choices"]):
                self._upgrade(m, m.offer["choices"][i]["id"])
        if act.get("take") is not None:
            self._take(m, act["take"], act.get("drop"))
        elif act.get("drop") is not None:
            s = int(act["drop"])
            w = m.slots[s] if s in (0, 1) else None
            if w and not (w["kind"] == "laser" and w["charge"] > 0):
                m.slots[s] = None
                self._add_pickup(w["kind"], m.x, m.y, m.z, w)
                m.events.append({"type": "dropped", "id": self.pickups[-1].id, "kind": w["kind"], "slot": s})

    def _take(self, m, pid, drop):
        pk = next((p for p in self.pickups if p.id == pid), None)
        reason = None
        if pk is None:
            reason = "gone"
        elif math.hypot(pk.x - m.x, pk.y - m.y) > self.cfg["loot"]["pickup_radius"] or abs(pk.z - m.z) > 4:
            reason = "range"
        elif pk.kind in m.kinds():
            reason = "have"
        else:
            slot = 1 if m.slots[1] is None else 0 if m.slots[0] is None else drop
            if slot not in (0, 1):
                reason = "full"
            elif m.slots[slot] and m.slots[slot]["kind"] == "laser" and m.slots[slot]["charge"] > 0:
                reason = "busy"
        if reason:
            m.events.append({"type": "take_failed", "id": pid, "reason": reason})
            return
        old = m.slots[slot]
        if old:
            self._add_pickup(old["kind"], m.x, m.y, m.z, old)
            m.events.append({"type": "dropped", "id": self.pickups[-1].id, "kind": old["kind"], "slot": slot})
        self.pickups.remove(pk)
        ws = m.wst[pk.kind]
        w = pk.state or {"kind": pk.kind, "cooldown": 0.0, "ammo": int(ws.get("mag", 0)), "reload": 0.0,
                         "charge": 0.0, "equip": 0.0}
        w["charge"] = 0.0
        w["equip"] = max(self.cfg["loot"]["equip_s"], ws.get("cooldown", 0.0))
        m.slots[slot] = w
        m.events.append({"type": "took", "id": pk.id, "kind": pk.kind, "slot": slot})

    # ------------------------------------------------------------------ physics
    def _slide(self, bodies, dz_list):
        """Exact move-and-slide for bodies (mechs or NPCs) by their velocity over one tick."""
        S = np.array([(b.x, b.y, b.z, b.r, b.h, b.vx * DT, b.vy * DT, dz, 1.0 if b.grounded else 0.0)
                      for b, dz in zip(bodies, dz_list)], float)
        out = self.G.move_many(S)
        for b, row, s in zip(bodies, out, S):
            ax, ay = row[0] - b.x, row[1] - b.y
            if ax * ax + ay * ay < s[5] * s[5] + s[6] * s[6] - 1e-9:  # walls took part of the move
                b.vx, b.vy = ax / DT, ay / DT
            b.x, b.y, b.z = row[0], row[1], row[2]
            b.grounded = bool(row[3] > 0.5)
            if row[4] > 0.5 or b.grounded:
                b.vz = 0.0

    def _move(self):
        t = self.t
        sc = self.cfg["sound"]
        alive = [m for m in self.mechs if m.alive]
        boosting_ids = set()
        for m in alive:
            st = m.stats
            dyaw = wrap(m.cyaw - m.yaw)
            lim = st["turn_rate"] * DT
            m.yaw = wrap(m.yaw + max(-lim, min(lim, dyaw)))
            lim = st["pitch_rate"] * DT
            m.pitch += max(-lim, min(lim, m.cpitch - m.pitch))
            mx, my = m.move
            moving = mx * mx + my * my > 1e-4
            spd = st["walk_speed"]
            boosting = m.boost and m.grounded and not m.jump and moving and m.energy > 0
            if boosting:
                boosting_ids.add(m.id)
                spd = st["boost_speed"]
                m.energy -= st["boost_drain"] * DT
                m.last_spend = t
            if m.charging():
                spd *= st["charge_move_mult"]
            acc = (st["ground_accel"] if m.grounded else st["air_accel"]) * DT
            dvx, dvy = mx * spd - m.vx, my * spd - m.vy
            dl = math.hypot(dvx, dvy)
            if dl > acc:
                dvx, dvy = dvx * acc / dl, dvy * acc / dl
            m.vx += dvx
            m.vy += dvy
            if m.jump and m.grounded:
                m.vz = st["jump_speed"]
                m.grounded = False
            m.jump = False
            jetting = not m.grounded and m.boost and m.energy > 0
            if jetting and not m.jetting:
                self.sounds.append(("jet", m.x, m.y, m.id, t))
            m.jetting = jetting
            if not m.grounded:
                if jetting:
                    m.vz = min(m.vz + st["jet_accel"] * DT, st["max_rise_speed"])
                    m.energy -= st["jet_drain"] * DT
                    m.last_spend = t
                m.vz -= st["gravity"] * DT
            if t - m.last_spend >= st["energy_regen_delay"]:
                m.energy = min(st["energy"], m.energy + st["energy_regen"] * DT)
            m.energy = max(0.0, m.energy)
        if alive:
            self._slide(alive, [m.vz * DT for m in alive])
        for m in alive:
            if m.grounded and math.hypot(m.vx, m.vy) > sc["step_speed"]:
                m.step_acc += DT
                if m.step_acc + 1e-9 >= sc["step_every"]:
                    m.step_acc = 0.0
                    self.sounds.append(("boost" if m.id in boosting_ids else "step", m.x, m.y, m.id, t))
        for i, a in enumerate(alive):  # overlapping mechs are pushed apart
            for b in alive[i + 1:]:
                dx, dy = b.x - a.x, b.y - a.y
                d = math.hypot(dx, dy)
                lim = a.r + b.r
                if d < lim and a.z < b.z + b.h and b.z < a.z + a.h:
                    if d < 1e-6:
                        dx, dy, d = 1e-6, 0.0, 1e-6
                    push = (lim - d) / 2 / d
                    rows = np.array([(m.x, m.y, m.z, m.r, m.h, sign * dx * push,
                                      sign * dy * push, 0.0, float(m.grounded))
                                     for m, sign in ((a, -1), (b, 1))])
                    for m, row in zip((a, b), self.G.move_many(rows)):
                        m.x, m.y, m.z = row[:3]
                        m.grounded = bool(row[3])

    def _dir(self, m, spread_deg=0.0):
        yaw, pitch = m.yaw, m.pitch
        if spread_deg:
            s = math.radians(spread_deg)
            yaw += self.rng.gauss(0, s)
            pitch += self.rng.gauss(0, s)
        cp = math.cos(pitch)
        return cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch)

    def _weapons(self):
        t = self.t
        for m in self.mechs:
            if not m.alive:
                continue
            for si, w in enumerate(m.slots):
                if not w:
                    continue
                k, ws, firing = w["kind"], m.wst[w["kind"]], m.fire[si]
                # Preserve sub-tick cadence while firing; never bank time spent blocked.
                w["cooldown"] -= DT
                if not firing or w["reload"] > 0 or w["equip"] > 0 or w["cooldown"] <= -DT + 1e-9:
                    w["cooldown"] = max(0.0, w["cooldown"])
                ready = w["cooldown"] <= 1e-9 and w["equip"] <= 1e-9
                w["equip"] = max(0.0, w["equip"] - DT)
                if k == "gun":
                    if w["reload"] > 0:
                        w["reload"] -= DT
                        if w["reload"] <= 1e-9:
                            w["reload"], w["ammo"] = 0.0, int(ws["mag"])
                    elif m.reload_req and w["ammo"] < ws["mag"]:
                        w["reload"] = ws["reload"]
                        self.sounds.append(("reload", m.x, m.y, m.id, t))
                    if firing and ready and w["ammo"] > 0 and w["reload"] <= 0:
                        dx, dy, dz = self._dir(m, ws["spread_deg"])
                        self._spawn_proj("gun", m, dx, dy, dz, ws["speed"], ws["damage"] * m.dmg_mult, ws["range"],
                                         self.cfg["projectile"]["gun_radius"], m.stats["muzzle_height"])
                        w["cooldown"] += ws["cooldown"]
                        w["ammo"] -= 1
                        if w["ammo"] <= 0:
                            w["reload"] = ws["reload"]
                        self.sounds.append(("gun", m.x, m.y, m.id, t))
                elif k == "mortar":
                    if firing and ready:
                        dx, dy, dz = self._dir(m)
                        p = self._spawn_proj("mortar", m, dx, dy, dz, ws["speed"], ws["damage"] * m.dmg_mult, 1e9,
                                             0.3, m.stats["height"] + 0.5)
                        p.g, p.ttl, p.ws, p.mult = ws["gravity"], ws["ttl"], dict(ws), m.dmg_mult
                        w["cooldown"] += ws["cooldown"]
                        self.sounds.append(("mortar", m.x, m.y, m.id, t))
                elif k == "laser":
                    if t < m.short_until:
                        w["charge"] = 0.0
                    elif w["charge"] > 0:
                        if firing or w["charge"] < ws["min_charge"] - 1e-9:
                            w["charge"] += DT
                        full = ws["charge"]
                        if (not firing and w["charge"] + 1e-9 >= ws["min_charge"]) or w["charge"] + 1e-9 >= full + ws["hold_s"]:
                            frac = ws["min_frac"] + (1 - ws["min_frac"]) * min(1.0, w["charge"] / full)
                            self._laser(m, ws, frac)
                            w["charge"], w["cooldown"] = 0.0, ws["cooldown"]
                    elif firing and ready:
                        w["charge"] = DT
                        self.sounds.append(("laser_charge", m.x, m.y, m.id, t))
                elif k == "lightning":
                    if firing and ready and m.energy >= ws["energy"]:
                        m.energy -= ws["energy"]
                        m.last_spend = t
                        w["cooldown"] += ws["cooldown"]
                        self._lightning(m, ws)

    def _spawn_proj(self, kind, owner, dx, dy, dz, speed, dmg, rng, rad, height):
        off = self._barrel_offset(owner, height, dx, dy, dz)
        p = Thing(id=self._id("proj"), kind=kind, owner=owner, x=owner.x + dx * off, y=owner.y + dy * off,
                  z=owner.z + height + dz * off, vx=dx * speed, vy=dy * speed, vz=dz * speed, dmg=dmg, range=rng,
                  rad=rad, dist=0.0, age=0.0, g=0.0, ttl=1e9)
        self.projs.append(p)
        return p

    def _targets(self, owner, mechs_only=False):
        out = [m for m in self.mechs if m.alive and m is not owner
               and not (owner.team is not None and m.team == owner.team)]
        if not mechs_only:
            out += [n for n in self.npcs if n.alive] + [c for c in self.crates if c.alive]
        return out

    def _projectiles(self):
        keep = []
        if not self.projs:
            return
        for p in self.projs:
            if p.g:
                p.vz -= p.g * DT
        A = np.array([(p.x, p.y, p.z) for p in self.projs], float)
        B = A + np.array([(p.vx, p.vy, p.vz) for p in self.projs], float) * DT
        for i, p in enumerate(self.projs):
            remaining = p.range - p.dist
            length = math.hypot(p.vx, p.vy, p.vz) * DT
            if length > remaining:
                B[i] = A[i] + (B[i] - A[i]) * (max(0.0, remaining) / length)
        walls = self.G.ray_many(A, B, False)
        for p, wt, endpoint in zip(self.projs, walls, B):
            ox, oy, oz = p.x, p.y, p.z
            ex, ey, ez = map(float, endpoint)
            L = math.hypot(ex - ox, ey - oy)
            seg = math.sqrt(L * L + (ez - oz) ** 2)
            wall = None if wt > 1.0 else max(0.0, wt - 0.3 / max(seg, 1e-6))  # stop just before the surface
            sx, sy = ex - ox, ey - oy
            ll = sx * sx + sy * sy
            best, hit = 2.0, None
            mx_, my_ = (ox + ex) / 2, (oy + ey) / 2
            for a in self._targets(p.owner, p.kind == "npc"):
                if abs(a.x - mx_) > L / 2 + a.r + 1 or abs(a.y - my_) > L / 2 + a.r + 1:
                    continue
                s = 0.0 if ll < 1e-9 else max(0.0, min(1.0, ((a.x - ox) * sx + (a.y - oy) * sy) / ll))
                cx, cy = ox + sx * s - a.x, oy + sy * s - a.y
                if cx * cx + cy * cy <= (a.r + p.rad) ** 2:
                    zz = oz + (ez - oz) * s
                    if a.z - 0.1 <= zz <= a.z + a.h and s < best:
                        best, hit = s, a
            if hit is not None and (wall is None or best <= wall):
                if p.kind == "mortar":
                    self._explode(p, ox + sx * best, oy + sy * best, oz + (ez - oz) * best, hit)
                else:
                    self._damage(hit, p.dmg, p.owner, p.kind, (ox, oy))
                continue
            if wall is not None:
                if p.kind == "mortar":
                    self._explode(p, ox + sx * wall, oy + sy * wall, oz + (ez - oz) * wall, None)
                continue
            p.x, p.y, p.z = ex, ey, ez
            p.dist += seg
            p.age += DT
            if p.dist < p.range - 1e-9 and p.age <= p.ttl:
                keep.append(p)
        self.projs = keep

    def _explode(self, p, ex, ey, ez, direct):
        ws, owner = p.ws, p.owner
        if direct is not None:
            self._damage(direct, ws["damage"] * p.mult, owner, "mortar", (ex, ey))
        R = ws["splash_radius"]
        for a in [m for m in self.mechs if m.alive] + [n for n in self.npcs if n.alive] + [c for c in self.crates if c.alive]:
            dh = max(0.0, math.hypot(a.x - ex, a.y - ey) - a.r)
            dz = 0.0 if a.z <= ez <= a.z + a.h else min(abs(ez - a.z), abs(ez - a.z - a.h))
            d = math.hypot(dh, dz)
            if d > R or (a is not direct and self.G.ray((ex, ey, ez), (a.x, a.y, a.z + a.h / 2)) <= 1.0):
                continue  # splash is blocked by walls
            f = 1 - (1 - ws["splash_edge"]) * d / R
            amt = ws["splash"] * f * p.mult
            if a is owner:
                amt *= owner.stats["self_damage_mult"]
            elif owner.team is not None and getattr(a, "team", None) == owner.team:
                continue
            self._damage(a, amt, owner, "mortar", (ex, ey), effect="concuss" if a.kind == "mech" else None, self_ok=True)
            if a.kind == "mech" and a.alive:
                dx, dy = a.x - ex, a.y - ey
                dl = math.hypot(dx, dy) or 1.0
                a.vx += dx / dl * ws["knock"] * f
                a.vy += dy / dl * ws["knock"] * f
                a.vz += ws["knock_up"] * f
                a.grounded = False
                a.energy = max(0.0, a.energy - ws["drain"] * f)
                a.events.append({"type": "effect", "effect": "concuss", "by": owner.id})
        self.sounds.append(("explosion", ex, ey, owner.id, self.t))

    def _barrel_offset(self, owner, height, dx, dy, dz):
        length = owner.r + 0.5
        start = (owner.x, owner.y, owner.z + height)
        end = (start[0] + dx * length, start[1] + dy * length, start[2] + dz * length)
        hit = self.G.ray(start, end)
        return max(0.0, length * hit - 1e-5) if hit <= 1.0 else length

    def _ray_wall(self, ox, oy, oz, dx, dy, dz, rng):
        t = self.G.ray((ox, oy, oz), (ox + dx * rng, oy + dy * rng, oz + dz * rng))
        return rng if t > 1.0 else t * rng

    def _laser(self, m, ws, frac):
        dx, dy, dz = self._dir(m)
        st = m.stats
        off = self._barrel_offset(m, st["muzzle_height"], dx, dy, dz)
        ox, oy, oz = m.x + dx * off, m.y + dy * off, m.z + st["muzzle_height"] + dz * off
        wall = self._ray_wall(ox, oy, oz, dx, dy, dz, ws["range"])
        h2 = math.hypot(dx, dy) or 1e-6
        hits = []
        for a in self._targets(m):
            rx, ry = a.x - ox, a.y - oy
            along = (rx * dx + ry * dy) / h2  # horizontal metres along the beam
            if along <= 0:
                continue
            s = along / h2
            if s > wall:
                continue
            perp = abs(rx * dy - ry * dx) / h2
            zz = oz + dz * s
            if perp <= a.r and a.z <= zz <= a.z + a.h:
                hits.append((s, a))
        hits.sort(key=lambda q: q[0])
        end = wall
        for s, a in hits[:1 + int(ws.get("pierce", 0))]:
            dmg = ws["damage"] * frac * m.dmg_mult
            eff = None
            if a.kind == "mech" and "mortar" in a.kinds():
                dmg *= 1 + ws["cookoff"]
                eff = "cookoff"
                a.events.append({"type": "effect", "effect": "cookoff", "by": m.id})
            self._damage(a, dmg, m, "laser", (ox, oy), effect=eff)
            end = s
        self.beams.append((m.id, "laser", [[ox, oy, oz], [ox + dx * end, oy + dy * end, oz + dz * end]], self.t))
        self.sounds.append(("laser", m.x, m.y, m.id, self.t))

    def _lightning(self, m, ws):
        st = m.stats
        dx, dy, dz = self._dir(m)
        off = self._barrel_offset(m, st["muzzle_height"], dx, dy, dz)
        ox, oy, oz = m.x + dx * off, m.y + dy * off, m.z + st["muzzle_height"] + dz * off
        half = math.radians(ws["cone_deg"]) / 2
        best, tgt = ws["range"] ** 2, None
        for a in self._targets(m):
            dx, dy, dz = a.x - ox, a.y - oy, a.z + a.h / 2 - oz
            d2 = dx * dx + dy * dy + dz * dz
            if d2 <= best and abs(wrap(math.atan2(dy, dx) - m.yaw)) <= half and \
                    self.G.ray((ox, oy, oz), (a.x, a.y, a.z + a.h / 2)) > 1.0:
                best, tgt = d2, a
        path = [[ox, oy, oz]]
        if tgt is None:
            dx, dy, dz = self._dir(m)
            s = self._ray_wall(ox, oy, oz, dx, dy, dz, ws["range"])
            path.append([ox + dx * s, oy + dy * s, oz + dz * s])
        else:
            hit, dmg, cur = {tgt.id}, ws["damage"] * m.dmg_mult, tgt
            for jump in range(int(ws["chain"]) + 1):
                eff = None
                if cur.kind == "mech" and "laser" in cur.kinds():
                    eff = "short"
                    cur.short_until = self.t + ws["short_s"]
                    for w in cur.slots:
                        if w and w["kind"] == "laser":
                            w["charge"] = 0.0
                    cur.events.append({"type": "effect", "effect": "short", "by": m.id})
                self._damage(cur, dmg, m, "lightning", (path[-1][0], path[-1][1]), effect=eff)
                path.append([cur.x, cur.y, cur.z + cur.h / 2])
                if jump == int(ws["chain"]):
                    break
                nxt, nd = None, ws["chain_range"] ** 2
                for a in self._targets(m):
                    if a.id in hit:
                        continue
                    d2 = ((a.x - cur.x) ** 2 + (a.y - cur.y) ** 2
                          + (a.z + a.h / 2 - cur.z - cur.h / 2) ** 2)
                    if d2 <= nd:
                        nxt, nd = a, d2
                if nxt is None:
                    break
                hit.add(nxt.id)
                cur, dmg = nxt, dmg * ws["chain_falloff"]
        self.beams.append((m.id, "lightning", path, self.t))
        self.sounds.append(("lightning", m.x, m.y, m.id, self.t))

    def _damage(self, tgt, amt, att, weapon, src, effect=None, self_ok=False):
        if not tgt.alive or tgt.hp <= 0 or amt <= 0:
            return
        t = self.t
        att_id = att.id if att is not None else None
        rear = False
        if tgt.kind == "mech":
            if att is tgt and not self_ok:
                return
            if att is not None and att is not tgt and att.team is not None and att.team == tgt.team:
                return
            b = math.atan2(src[1] - tgt.y, src[0] - tgt.x)
            rear = abs(wrap(b - tgt.yaw)) > math.radians(tgt.stats["front_arc_deg"]) / 2
            if rear:
                amt *= tgt.stats["rear_mult"]
            amt *= max(0.0, 1 - tgt.stats["armor"])
            tgt.hp -= amt
            tgt.last_dmg = t
            ev = {"type": "damaged", "amount": round(amt, 1), "weapon": weapon, "bearing": round(b, 3), "by": att_id}
            if rear:
                ev["rear"] = True
            tgt.events.append(ev)
            if tgt.hp <= 0:
                tgt.killer = att
        elif tgt.kind == "npc":
            tgt.hp -= amt
            if att is not None and att.kind == "mech":
                tgt.target = att
            if tgt.hp <= 0:
                tgt.alive = False
                for _ in range(int(self.cfg["npc"]["cubes"])):
                    self._spawn_cube(tgt.x, tgt.y, tgt.z)
        else:
            tgt.hp -= amt
            if tgt.hp <= 0:
                tgt.alive = False
                self._spawn_cube(tgt.x, tgt.y, tgt.z)
                lo = self.cfg["loot"]
                if self.rng.random() < lo["crate_weapon_chance"]:
                    kinds = [k for k, v in lo["weights"].items() if v > 0]
                    self._add_pickup(self.rng.choices(kinds, [lo["weights"][k] for k in kinds])[0], tgt.x, tgt.y, tgt.z)
                self.sounds.append(("crate", tgt.x, tgt.y, -1, t))
        if att is not None and att.kind == "mech" and att is not tgt:
            att.last_dmg = t
            ev = {"type": "hit", "target": tgt.id, "amount": round(amt, 1), "weapon": weapon}
            if rear:
                ev["rear"] = True
            if effect:
                ev["effect"] = effect
            att.events.append(ev)

    def _spawn_cube(self, x, y, z):
        a = self.rng.uniform(0, TAU)
        r = self.rng.uniform(0, 2.5)
        self.cubes.append(Thing(id=self._id("cube"), x=x + math.cos(a) * r, y=y + math.sin(a) * r, z=z))

    # ------------------------------------------------------------------ world systems
    def _npcs_think(self):
        nc = self.cfg["npc"]
        alive = [m for m in self.mechs if m.alive]
        for n in self.npcs:
            if not n.alive:
                continue
            dh = math.hypot(n.x - n.hx, n.y - n.hy)
            tg = n.target
            if tg is not None and (not tg.alive or math.hypot(tg.x - n.x, tg.y - n.y) > nc["deaggro_range"]
                                   or dh > nc["chase_leash"]):
                n.target = tg = None
                n.gx, n.gy, n.gt = n.hx, n.hy, self.cfg["match"]["duration_s"] + DT
            if tg is None and dh <= nc["leash"]:
                for m in alive:
                    if math.hypot(m.x - n.x, m.y - n.y) <= nc["aggro_range"] and \
                            self.G.sight((n.x, n.y, n.z + n.h), (m.x, m.y, m.z + m.h / 2)):
                        n.target = tg = m
                        break
            n.cool -= 0.1
            if tg is not None:
                n.gx, n.gy = (n.x, n.y) if math.hypot(tg.x - n.x, tg.y - n.y) < 15 else (tg.x, tg.y)
                tz = tg.z + tg.h / 2
                d = math.hypot(tg.x - n.x, tg.y - n.y)
                n.yaw = math.atan2(tg.y - n.y, tg.x - n.x)
                if n.cool <= 1e-9 and d < 120 and self.G.ray((n.x, n.y, n.z + n.h), (tg.x, tg.y, tz)) > 1.0:
                    n.cool = nc["cooldown"]
                    s = math.radians(nc["spread_deg"])
                    yaw = n.yaw + self.rng.gauss(0, s)
                    pitch = math.atan2(tz - n.z - n.h, max(d, 1)) + self.rng.gauss(0, s)
                    cp = math.cos(pitch)
                    self._spawn_proj("npc", n, cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch),
                                     nc["projectile_speed"], nc["damage"], 200, self.cfg["projectile"]["npc_radius"], n.h)
                    self.sounds.append(("npc_gun", n.x, n.y, n.id, self.t))
            elif self.t >= n.gt or math.hypot(n.gx - n.x, n.gy - n.y) < 2:
                a = self.rng.uniform(0, TAU)
                r = self.rng.uniform(0, nc["leash"])
                n.gx, n.gy, n.gt = n.hx + math.cos(a) * r, n.hy + math.sin(a) * r, self.t + self.rng.uniform(3, 8)

    def _npcs_move(self):
        spd = self.cfg["npc"]["speed"]
        moving = []
        for n in self.npcs:
            if not n.alive:
                continue
            dx, dy = n.gx - n.x, n.gy - n.y
            d = math.hypot(dx, dy)
            if d < 0.5:
                n.vx = n.vy = 0.0
                continue
            n.vx, n.vy = dx / d * spd, dy / d * spd
            moving.append(n)
        if moving:
            self._slide(moving, [-0.05] * len(moving))  # drones stay on the ground
            for n in moving:
                if math.hypot(n.vx, n.vy) < spd * 0.3:
                    n.gt = 0.0  # blocked: pick a new wander goal

    def _cubes_pickup(self):
        cu = self.cfg["cubes"]
        r2 = cu["pickup_radius"] ** 2
        keep = []
        alive = [m for m in self.mechs if m.alive and m.hp > 0]
        for c in self.cubes:
            best, who = r2, None
            for m in alive:
                d2 = (m.x - c.x) ** 2 + (m.y - c.y) ** 2
                if d2 <= best and abs(m.z - c.z) <= 4 and (who is None or d2 < best or m.id < who.id):
                    best, who = d2, m
            if who is None:
                keep.append(c)
            else:
                self._give_cubes(who, 1)
                who.events.append({"type": "pickup", "cubes": who.cubes})
        self.cubes = keep

    def _give_cubes(self, m, n):
        th = self.cfg["cubes"]["upgrade_thresholds"]
        for _ in range(n):
            m.cubes += 1
            if m.cubes in th:
                m.pending += 1
        m.recalc()
        if n:
            # Growth can put the enlarged body through a wall before its next move.
            row = self.G.move_many(np.array([(m.x, m.y, m.z, m.r, m.h, 0, 0, 0, float(m.grounded))]))[0]
            m.x, m.y, m.z = row[:3]
            m.grounded = bool(row[3])

    def _offers(self):
        t = self.t
        for m in self.mechs:
            if not m.alive or m.hp <= 0:
                continue
            if m.offer is not None and t >= m.offer["expires"]:
                self._upgrade(m, m.offer["choices"][0]["id"])
            if m.offer is None and m.pending > 0:
                held = set(m.kinds())
                opts = [u for u in self.cfg["upgrades"] if m.upc.get(u["id"], 0) < u["max"]
                        and (not u.get("requires") or u["requires"] in held)]
                if not opts:
                    m.pending = 0
                    continue
                m.pending -= 1
                m.offer_seq += 1
                ch = self.rng.sample(opts, min(3, len(opts)))
                m.offer = {"id": m.offer_seq, "expires": t + self.cfg["bot"]["offer_timeout_s"],
                           "choices": [{"id": u["id"], "name": u["name"], "desc": u["desc"]} for u in ch]}

    def _upgrade(self, m, uid):
        m.upc[uid] = m.upc.get(uid, 0) + 1
        m.upgrades.append(uid)
        m.recalc()
        m.offer = None
        m.events.append({"type": "upgrade", "id": uid})

    def _zone_regen(self):
        t = self.t
        zc, zr, ph = self.zone_at(t)
        dps = ph[6]
        for m in self.mechs:
            if not m.alive or m.hp <= 0:
                continue
            out = math.hypot(m.x - zc[0], m.y - zc[1]) > zr
            m.in_zone = not out
            if out:
                m.hp -= dps * DT
                m.zone_acc += dps * DT
                m.last_dmg = t
                if m.hp <= 0:
                    m.killer = None
            elif t - m.last_dmg >= m.stats["regen_delay"] and m.hp < m.max_hp:
                m.hp = min(m.max_hp, m.hp + m.stats["regen_per_s"] * DT)

    def _drops_step(self):
        t, dc = self.t, self.cfg.get("drops")
        if not dc:
            return
        if self.drop_i < len(dc["times"]) and t >= dc["times"][self.drop_i]:
            _, _, ph = self.zone_at(t)
            px, py = self._open_point(ph[4][0], ph[4][1], max(5.0, ph[5] * dc["spread"]), 20)
            self.drops.append(Thing(id=self._id("drop"), x=px, y=py, z=self.G.surface(px, py), lands_at=t + dc["fall_s"],
                                    progress=0.0, holder=None, landed=False))
            self.drop_i += 1
        keep = []
        for d in self.drops:
            if t >= d.lands_at:
                if not d.landed:
                    d.landed = True
                    self.sounds.append(("drop", d.x, d.y, -1, t))
                occ = [m for m in self.mechs if m.alive and m.hp > 0 and math.hypot(m.x - d.x, m.y - d.y) <= dc["radius"]
                       and abs(m.z - d.z) < 6]
                if len(occ) == 1:
                    if d.holder != occ[0].id:
                        d.holder, d.progress = occ[0].id, 0.0
                    d.progress += DT
                    if d.progress + 1e-9 >= dc["capture_s"]:
                        m = occ[0]
                        self._give_cubes(m, int(dc["cubes"]))
                        m.hp, m.energy = m.max_hp, m.stats["energy"]
                        m.events.append({"type": "drop", "id": d.id, "cubes": m.cubes})
                        continue
                elif not occ:
                    d.holder, d.progress = None, 0.0
            keep.append(d)
        self.drops = keep

    def _deaths(self):
        dead = [m for m in self.mechs if m.alive and m.hp <= 0]
        if not dead:
            return
        place = sum(1 for m in self.mechs if m.alive)
        extra = int(self.cfg["cubes"]["death_drop_extra"])
        for m in dead:
            m.hp = 0.0
            m.alive, m.place = False, place
            for _ in range(m.cubes + extra):
                self._spawn_cube(m.x, m.y, m.z)
            for w in m.slots:
                if w and w["kind"] != "gun":
                    self._add_pickup(w["kind"], m.x, m.y, m.z, w)
            self.sounds.append(("death", m.x, m.y, m.id, self.t))
            k = getattr(m, "killer", None)
            if k is not None and k.kind == "mech" and k is not m:
                k.kills += 1
                k.events.append({"type": "kill", "target": m.id})
