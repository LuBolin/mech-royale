"""Stand-ins for the practice house bots and the two match modes used as fitness.

The real house bots are unknown. Per the track page they share vision, never attack each other, the
weakest hunt first, more join as the player kills them, and all hunt late. Difficulty here only
changes aim noise, lead and loot use. shortcut: all bots hunt from ALL_HUNT_T; tune if practice
scores on the server disagree with the simulator.
"""
import os
import random

import params as PR
from brain import Bot
from engine import Match

ALL_HUNT_T = 240.0
LEVELS = ["easy"] * 4 + ["medium"] * 4 + ["hard"] * 4 + ["expert"] * 3
RANK = {"easy": 0, "medium": 1, "hard": 2, "expert": 3}
SKILL = {  # aim noise (rad), lead, weapon preference
    "easy": (0.08, 0.0, -1.0),
    "medium": (0.045, 0.5, 0.5),
    "hard": (0.022, 0.9, 0.6),
    "expert": (0.01, 1.0, 0.7),
}


class HouseBot(Bot):
    name = "house"

    def __init__(self, hello, level, shared):
        noise, lead, wp = SKILL[level]
        prm = dict(lead=lead, w_mortar=wp, w_laser=wp, w_lightning=wp, npc_dist=30, crate_dist=30,
                   loot_dist=50, camp=0.0, quiet=0.0, face_sound=1.0, flee_hp=0.2, engage_range=110,
                   survival=0.0)  # keep the calibrated house logic; brain.py's survival layer is ours only
        prm.update(PR.OFF)
        super().__init__(hello, prm, aim_noise=noise)
        self.base_noise = noise
        self.shared = shared

    def act(self, obs):
        if self.shared is None:
            return super().act(obs)  # Independent expert filler in a tournament FFA.
        hunting = obs["self"]["id"] in self.shared["hunters"]
        p = self.p
        if self.shared.get("v2"):
            return self._act_v2(obs, hunting)
        p["engage_range"] = 230 if hunting else 110
        p["aggr"] = 50.0 if hunting else 1.2
        p["flee_hp"] = 0.0 if hunting else 0.2
        last = self.shared["last"]
        self.goal_hint = (last[1][0], last[1][1]) if hunting and last and obs["t"] - last[0] < 30 else None
        return super().act(obs)

    def _act_v2(self, obs, hunting):
        """House v2 (practice_match(..., house="v2")): evidence from other teams' notes, sources inline."""
        sh, p, t = self.shared, self.p, obs["t"]
        # First hunter at ~20 s, not at t=0 (nullat0m7 sim/house.ts joinStart; track page: "one early").
        hunting = hunting and t >= V2_FIRST_HUNT_T
        pm = sh["match"].mechs[sh["player"]]
        d = max(1.0, ((pm.x - obs["self"]["pos"][0]) ** 2 + (pm.y - obs["self"]["pos"][1]) ** 2) ** 0.5)
        # Aim noise grows with range: real house gun hits fall from ~88% under 40 m to ~15% at 120-180 m, faster
        # than constant angular noise gives (sim/house_aim.py measures it).
        self.aim_noise = self.base_noise * AIM_A * (d / 80.0) ** AIM_B
        # Non-hunters never start or return fights (pedduvedcharan NOTES.md cal-A, joekritenbrink README: passive
        # non-hunters put the kit starter at 0.8-1.1 vs real 1.15; itsyashk crates/sim/README: retaliation gives
        # 0.34-0.54). Here, retaliating for 6 s (jack tools/house.py) put the starter at 0.22 and all bots at ~0.4x.
        # Waking non-hunters when shot or approached within 45-60 m (ch2li's model) lowered board agreement
        # (rank correlation 0.74 -> 0.64-0.69, sim/rank_check.py), so they stay passive.
        pl = sh["player"]
        p["engage_range"] = 230 if hunting else 0
        p["aggr"] = 50.0 if hunting else 1.2
        p["flee_hp"] = 0.0 if hunting else 0.2
        # Hunters learn the player's true position (+-20 m) every 8 s when nobody sees it (nullat0m7 house.ts
        # intelEvery; jack LEARNINGS: hiding to heal fails because "the hunters always know where we are";
        # Infamustdev hide probe scored 0.04 for real). One update per squad, not per bot.
        M, last = sh["match"], sh["last"]
        me = M.mechs[pl]
        if hunting and me.alive and t - max(sh["intel_t"], last[0] if last else -1e9) >= V2_INTEL_S:
            sh["intel_t"] = t
            r = self.rng
            sh["last"] = last = (t, (me.x + r.uniform(-20, 20), me.y + r.uniform(-20, 20), me.z))
        self.goal_hint = (last[1][0], last[1][1]) if hunting and last and t - last[0] < 30 else None
        return Bot.act(self, obs)


V2_FIRST_HUNT_T, V2_INTEL_S = 20.0, 8.0
# House v2 aim: noise = tier noise * AIM_A * (range / 80 m) ** AIM_B. Env overrides are for calibration sweeps.
# 0.6 / 0.5 (league.py --house v2c) gives 95/57/30/16/2% at <40/40-80/80-120/120-180/180+ m (real: 88/55/30/15/3%)
# and raises rank correlation with the board from 0.66 to 0.74 (sim/house_aim.py, sim/rank_check.py).
AIM_A, AIM_B = float(os.environ.get("HOUSE_AIM_A", 1.0)), float(os.environ.get("HOUSE_AIM_B", 0.0))


def practice_match(cfg, seed, params, player=None, house="v1"):
    """Unrun practice match: params (or player(hello), if given) vs 15 cooperating house bots.
    house="v2" selects the evidence-based squad (HouseBot._act_v2). Returns (match, our slot, seat labels)."""
    rng = random.Random(seed)
    slot = rng.randrange(16)
    levels = LEVELS[:]
    rng.shuffle(levels)
    shared = {"player": slot, "last": None, "hunters": set(), "all_hunt_t": ALL_HUNT_T}
    if house == "v2":
        shared.update(v2=True, intel_t=-1e9)
    facs, teams, lv, labels = [], [], {}, []
    for i in range(16):
        if i == slot:
            facs.append(player or (lambda h: Bot(h, params)))
            teams.append(None)
            labels.append("ours")
        else:
            L = levels.pop()
            lv[i] = RANK[L]
            facs.append(lambda h, L=L: HouseBot(h, L, shared))
            teams.append(1)
            labels.append(L)
    M = Match(cfg, seed, facs, teams, track=slot, shared=shared, levels=lv)
    if house == "v2":
        shared["match"] = M  # hunter intel reads the player's true position
        damage = M._damage

        def _damage(tgt, amt, att, *a, **k):
            # Drones never hurt the squad: a real never-shoot hide probe scored 0.04, so house bots only die to
            # the player (Infamustdev sim/ASSUMPTIONS.md, experiments.md P1).
            if getattr(tgt, "team", None) is not None and getattr(att, "kind", None) == "npc":
                return None
            return damage(tgt, amt, att, *a, **k)
        M._damage = _damage
    return M, slot, labels


def practice(cfg, seed, params):
    """One practice match. Returns placement points (16 - place)."""
    M, slot, _ = practice_match(cfg, seed, params)
    res = M.run()
    return 16 - res["place"][slot], res


def ffa_match(cfg, seed, params_list):
    """Unrun free-for-all among 16 genomes. Returns (match, order) with order[slot] = index into params_list."""
    rng = random.Random(seed)
    order = list(range(16))
    rng.shuffle(order)
    if len(params_list) > 16:
        raise ValueError("FFA has 16 seats")
    facs = []
    for entry in order:
        if entry < len(params_list) and params_list[entry] is not None:
            facs.append(lambda h, p=params_list[entry]: Bot(h, p))
        else:
            facs.append(lambda h: HouseBot(h, "expert", None))
    return Match(cfg, seed, facs), order


def ffa(cfg, seed, params_list):
    """One free-for-all. Returns placement points per entry (same order)."""
    M, order = ffa_match(cfg, seed, params_list)
    res = M.run()
    pts = [0.0] * 16
    for s in range(16):
        pts[order[s]] = 16 - res["place"][s]
    return pts[:len(params_list)], res
