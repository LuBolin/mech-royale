"""The genome: every behaviour knob the bot exposes, with search bounds and a hand-set default.

Evolution searches x in [0, 1]^n; decode() maps it to these ranges. genome.json (next to this file)
holds the current best values and overrides the defaults.
"""
import json
import os

# name, low, high, default, meaning
SPEC = [
    # zone and positioning
    ("zone_margin", 0, 80, 25, "metres to stay inside the next circle"),
    ("zone_lead", 0, 40, 12, "extra seconds of travel slack before a shrink starts"),
    ("center_bias", 0, 1, 0.35, "home point: 0 = next centre, 1 = margin from its edge"),
    ("quiet", 0, 1, 0.3, "slow roaming; 1 = 40% speed, below the step-sound threshold"),
    ("camp", 0, 1, 0.6, ">= 0.5: stand still at the home point"),
    ("scan_speed", 0, 3, 0.8, "rad/s torso sweep when idle"),
    ("face_threat_s", 0, 8, 3, "seconds to keep facing the last attacker's bearing"),
    ("face_sound", 0, 1, 0.5, ">= 0.5: turn toward loud sounds heard in the last 2 s"),
    # combat
    ("engage_range", 30, 230, 160, "max distance to pick a mech as target"),
    ("tgt_w_hp", 0, 3, 0.6, "target score weight on enemy hp fraction"),
    ("tgt_w_threat", 0, 3, 1.0, "target score bonus for whoever hit us last"),
    ("aggr", 0.3, 3, 1.0, "fight if our strength * aggr >= enemy hp, else flee"),
    ("max_enemies", 1, 4, 1.5, "flee when more mechs than this are within engage range"),
    ("dodge_shell_t", 0, 3, 1.5, "dodge mortar shells landing near us within this many seconds"),
    ("flee_hp", 0, 0.8, 0.3, "flee below this hp fraction"),
    ("late_alive", 2, 10, 5, "late game starts when this many mechs are left"),
    ("late_aggr", 0.2, 2, 1.0, "aggression multiplier in the late game"),
    ("pref_dist", 15, 180, 70, "preferred fighting distance"),
    ("lightning_dist", 10, 45, 30, "preferred distance while holding lightning"),
    ("strafe", 0, 1, 0.6, "sideways movement share while fighting"),
    ("strafe_period", 0.3, 3, 1.0, "mean seconds between strafe direction changes"),
    ("lead", 0, 1.3, 1.0, "fraction of target velocity to lead"),
    ("aim_tol", 0.3, 3, 1.2, "fire when aim error < aim_tol * target radius"),
    ("fire_range", 0.3, 1, 0.9, "fire the gun within this fraction of its range"),
    ("cover", 0, 1, 0.5, "weight for breaking line of sight when fleeing"),
    ("dodge", 0, 1, 0.3, "chance to jump / flip strafe when shot at or laser-aimed"),
    ("boost_energy", 0, 100, 35, "boost only above this energy"),
    ("reload_frac", 0, 1, 0.5, "reload early below this fraction of the magazine"),
    ("laser_release", 0.3, 1, 1.0, "release the laser at this fraction of full charge"),
    # loot and economy
    ("w_mortar", 0, 1, 0.5, "preference for the mortar"),
    ("w_laser", 0, 1, 0.7, "preference for the laser"),
    ("w_lightning", 0, 1, 0.4, "preference for lightning"),
    ("swap_margin", 0, 0.6, 0.2, "preference gap needed to swap slot-1 weapons"),
    ("loot_dist", 0, 150, 60, "max detour to a weapon pickup"),
    ("cube_dist", 0, 100, 35, "max detour to a power cube"),
    ("crate_dist", 0, 120, 40, "max detour to shoot a crate"),
    ("npc_dist", 0, 120, 0, "max detour to fight an NPC drone"),
    ("npc_hp", 0.3, 1, 0.8, "min hp fraction to fight drones"),
    ("drop_dist", 0, 300, 120, "max distance to go for a supply drop"),
    ("drop_contest", 0, 80, 40, "skip a drop if a visible mech is this close to its ring"),
    # competitor-survey features (docs/competitor-survey.md); OFF below restores the pre-survey bot.
    # Defaults follow the sim A/Bs in runs/ab (2026-11-21): Tier 2 knobs other than fire_early showed no gain.
    ("sidestep", 0, 1, 1, ">= 0.5: step sideways out of incoming gun rounds"),
    ("hide", 0, 1, 0, ">= 0.5: after contact breaks, stand still until regen refills hp"),
    ("hostile_only", 0, 1, 1, ">= 0.5: only target mechs that hit us, aim at us or come close"),
    ("loot_mem_s", 0, 200, 120, "seconds to remember weapon pickups after they leave view"),
    ("loot_dist_empty", 0, 250, 160, "max detour to a weapon pickup while slot 1 is empty"),
    ("finish_hp", 0, 0.5, 0.2, "always fight a target below this hp fraction"),
    ("stick", 0, 2, 0.5, "target score bonus for keeping the current target"),
    ("open_boost_s", 0, 10, 4, "boost while roaming for this many seconds after the start"),
    ("mortar_laser", 0, 1, 1, ">= 0.5: drop the mortar when a laser holder is in sight"),
    ("ffa_detect", 0, 1, 0, ">= 0.5: detect a free-for-all lobby and switch to the ffa profile"),
    ("ffa_aggr", 0.3, 1.5, 0.8, "aggression multiplier once a free-for-all is detected"),
    ("third_party", 0, 1, 0.5, "target score bonus for mechs busy fighting someone else (ffa)"),
    ("fire_early", 0, 1, 1, ">= 0.5: start the laser charge on the tick the cooldown ends"),
    ("laser_smart", 0, 1, 0, ">= 0.5: release the laser on a kill, before losing LOS, on predicted aim"),
    ("ghost_fire", 0, 1, 0, ">= 0.5: fire back at an unseen shooter's estimated position"),
    ("pincer", 0, 1, 0, ">= 0.5: flee through the largest gap when threats surround us"),
    ("final_r", 0, 60, 0, "stand on the exact next centre once its radius is at most this"),
]

# Knob values that reproduce the pre-survey bot: house bots and A/B baselines use them.
OFF = {"sidestep": 0, "hide": 1, "hostile_only": 0, "loot_mem_s": 0, "loot_dist_empty": 0, "finish_hp": 0,
       "stick": 0, "open_boost_s": 0, "mortar_laser": 0, "ffa_detect": 0, "ffa_aggr": 1, "third_party": 0,
       "fire_early": 0, "laser_smart": 0, "ghost_fire": 0, "pincer": 0, "final_r": 0}

UPGRADES = [
    "plating", "armor", "servos", "capacitor", "reactor", "jets", "nanites", "gyros", "sensors",
    "gun_damage", "gun_rate", "gun_velocity", "gun_mag", "mortar_splash", "mortar_reload",
    "mortar_velocity", "laser_charge", "laser_damage", "laser_pierce", "lightning_chain",
    "lightning_range", "lightning_efficiency",
]
_UP_DEFAULT = {"armor": 0.9, "plating": 0.85, "nanites": 0.7, "gun_rate": 0.65, "gun_damage": 0.6,
               "servos": 0.55, "laser_damage": 0.6, "gyros": 0.5}
SPEC += [("up_" + u, 0, 1, _UP_DEFAULT.get(u, 0.3), "upgrade preference") for u in UPGRADES]
# Appended last so genome vectors saved before these knobs existed still decode.
# Arnav's "v5" package (real board 3.81 -> 5.13); defaults keep the old behaviour, genome.json turns them on.
SPEC += [
    ("berserk", 0, 1, 0, ">= 0.5: never flee from a fight"),
    ("tgt_hpd", 0, 1, 0, ">= 0.5: target the lowest hp * (1 + dist / 120) in line of sight"),
    ("laser_dist", 0, 180, 0, "preferred fighting distance while holding a laser; 0 = pref_dist"),
    ("fight_loot", 0, 1, 0, ">= 0.5: while fighting, walk over cubes within 30 m and fetch a better weapon"),
    ("fight_loot_dist", 0, 250, 200, "max distance to fetch a better weapon mid-fight"),
]

NAMES = [s[0] for s in SPEC]
GENOME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "genome.json")


def defaults():
    return {s[0]: float(s[3]) for s in SPEC}


def decode(x):
    return {n: lo + (hi - lo) * min(1.0, max(0.0, float(v))) for (n, lo, hi, _, _), v in zip(SPEC, x)}


def encode(p):
    return [(float(p.get(n, d)) - lo) / (hi - lo) for n, lo, hi, d, _ in SPEC]


def load(path=GENOME_PATH):
    p = defaults()
    try:
        with open(path) as f:
            p.update({k: float(v) for k, v in json.load(f).items() if k in p})
    except (OSError, ValueError):
        pass
    return p
