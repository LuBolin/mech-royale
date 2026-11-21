"""Stand-ins for the practice house bots and the two match modes used as fitness.

The real house bots are unknown. Per the track page they share vision, never attack each other, the
weakest hunt first, more join as the player kills them, and all hunt late. Difficulty here only
changes aim noise, lead and loot use. shortcut: all bots hunt from ALL_HUNT_T; tune if practice
scores on the server disagree with the simulator.
"""
import random

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
        super().__init__(hello, prm, aim_noise=noise)
        self.shared = shared

    def act(self, obs):
        if self.shared is None:
            return super().act(obs)  # Independent expert filler in a tournament FFA.
        hunting = obs["self"]["id"] in self.shared["hunters"]
        p = self.p
        p["engage_range"] = 230 if hunting else 110
        p["aggr"] = 50.0 if hunting else 1.2
        p["flee_hp"] = 0.0 if hunting else 0.2
        last = self.shared["last"]
        self.goal_hint = (last[1][0], last[1][1]) if hunting and last and obs["t"] - last[0] < 30 else None
        return super().act(obs)


def practice(cfg, seed, params):
    """One practice match: params vs 15 cooperating house bots. Returns placement points (16 - place)."""
    rng = random.Random(seed)
    slot = rng.randrange(16)
    levels = LEVELS[:]
    rng.shuffle(levels)
    shared = {"player": slot, "last": None, "hunters": set(), "all_hunt_t": ALL_HUNT_T}
    facs, teams, lv = [], [], {}
    for i in range(16):
        if i == slot:
            facs.append(lambda h: Bot(h, params))
            teams.append(None)
        else:
            L = levels.pop()
            lv[i] = RANK[L]
            facs.append(lambda h, L=L: HouseBot(h, L, shared))
            teams.append(1)
    res = Match(cfg, seed, facs, teams, track=slot, shared=shared, levels=lv).run()
    return 16 - res["place"][slot], res


def ffa(cfg, seed, params_list):
    """One free-for-all among 16 genomes. Returns placement points per entry (same order)."""
    rng = random.Random(seed)
    order = list(range(16))
    rng.shuffle(order)  # order[slot] = index into params_list
    if len(params_list) > 16:
        raise ValueError("FFA has 16 seats")
    facs = []
    for entry in order:
        if entry < len(params_list) and params_list[entry] is not None:
            facs.append(lambda h, p=params_list[entry]: Bot(h, p))
        else:
            facs.append(lambda h: HouseBot(h, "expert", None))
    res = Match(cfg, seed, facs).run()
    pts = [0.0] * 16
    for s in range(16):
        pts[order[s]] = 16 - res["place"][s]
    return pts[:len(params_list)], res
