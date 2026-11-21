# Mech Royale: data and tunable parameters

Sources: [protocol.md](../protocol.md), the kit wrapper (`kit/python/mechfight.py`), the recorded match in `kit/check/stream.jsonl.gz`, and the code in `bot/` and `sim/`.

## 1. Data the bot receives

### Once per match: `hello`
| Field | Use |
| --- | --- |
| `slot` | Our mech id (0–15). It varies per match. |
| `seed` | Per-slot RNG seed. It says nothing about the match. |
| `map.dir` | Read-only map: `v2/world.json` (259 prisms: 223 buildings, 27 stadium, 9 bridges; 1560 trees) and `v2/terrain.i16` (1070 × 724, 1 m grid). `bot/world.py` turns it into a height grid in 1.4 s, or 0.13 s from the `/tmp` cache. |
| `config` | The complete rule set (section 2). Read numbers from it; balance can change. |

### 10 times per second: `obs`
| Field | What it gives | Notes for strategy |
| --- | --- | --- |
| `t`, `tick`, `alive` | Match clock and mechs left. `alive` = our place if we died now. | Drives late-game logic. |
| `self` | pos, vel, actual aim, hp/max_hp, energy, grounded, cubes, upgrades, in_zone, `stats` (our mech numbers after upgrades), `weapons[2]` (kind, cooldown, ammo, reload, laser charge, per-weapon stats) | Everything about us, exact. |
| `mechs` | Seen mechs and NPC drones: id, pos, vel, torso aim, hp, max_hp, cubes, held weapons, `charging` | **Seen mechs stay tracked at their real position for 5 s** after they leave view. Enemy hp and loadout are visible, so we can pick fights. `charging` + aim telegraphs a laser shot. |
| `projectiles` | Enemy bullets and shells in view, with velocity | Lets the bot dodge shells (gravity is known). |
| `crates`, `cubes`, `pickups` | Loot in view: crates (hp), power cubes, weapons on the ground (kind) | Listed only while seen. No memory. |
| `sounds` | Kind, bearing (σ = 7° noise), dist bucket (near < 60 m, mid < 160 m, far), age | The kind tells us the shooter's weapon. Walking under 5 m/s or standing still is silent. |
| `beams` | Laser and lightning paths we saw, with shooter id | Shows who fights whom. |
| `events` | `damaged` (amount, weapon, **attacker bearing and id**, rear hit), `hit`, `kill`, `pickup`, `took`, `dropped`, `take_failed`, `upgrade`, `effect` (cookoff / short / concuss), `drop` | `damaged.bearing` is the best signal for facing an unseen attacker (rear hits do ×1.5 damage). |
| `drops` | Supply drops from the moment they are called: pos, landing time, capture progress, holder | Always known, like the zone. A capture gives 3 cubes and full HP/energy. |
| `zone` | Current circle, **next circle**, shrink start/end, dps | Always known, so the bot can be in the next circle early. |
| `offer` | Upgrade choice (3 options, 5 s timeout, otherwise choice 0) | Answer with `pick`. |

### What the bot sends
`move` (world-frame direction, length ≤ 1), `boost`, `jump`, `aim` [yaw, pitch], `fire` [slot0, slot1] (held for 2 ticks), `reload`, `pick`, `take` + optional `drop`, `debug`. The reply deadline is 50 ms. A missed reply coasts on the last move/aim for 1 s, then the mech idles.

## 2. Fixed rules (from `hello.config`, not tunable by us)
| Group | Key numbers |
| --- | --- |
| Mech | 1400 HP, walk 11.7 m/s, boost 19.5 m/s (17 energy/s), jet 15 energy/s (100 energy ≈ 60 m of climb), turn 4 rad/s, pitch 3 rad/s, regen 45 HP/s after 5 s without damage, rear hits ×1.5 outside the 180° front arc |
| Vision | 200 m inside a 120° wedge around the torso aim, 45 m all around, 5 s memory |
| Gun | 45 dmg, 0.24 s, 252 m/s, 230 m, 12-round magazine, 2.4 s reload |
| Mortar | 200 direct + 160 splash (7 m, down to 30% at the edge), 72 m/s, g = 22, 2.6 s cooldown, knockback, −45 energy |
| Laser | Up to 315 dmg at 0.75 s charge, hitscan 280 m, 1.65 s cooldown, half speed while charging, +60% against mortar carriers |
| Lightning | 60 dmg, 45 m, 30° cone, chains to 2 more targets (×0.7 each), 3 energy, 0.25 s; shorts lasers |
| Economy | 36 crates (1 cube, 50% weapon), 40 floor weapons, 12 drones (350 HP, 2 cubes), kills drop all cubes + 2. Each cube: +4% HP and damage (cap 10), +5% size (no cap). Offers at 1, 3, 6, 10, 15, 21 … cubes |
| Zone | 560 m → 380 → 240 → 140 → 70 → 25 → 0 m; 20 → 260 dps; closed by about 6:55; 8:00 limit |
| Drops | Called at 70, 140, 210 s; land 25 s later inside the next circle; capture by standing alone in an 8 m ring for 4 s |
| Scoring | 16 − place per match. Survival is everything. Kills only help through cubes. |

## 3. Tunable bot parameters (the genome, `bot/params.py`)
Evolution searches the normalised box [0, 1]ⁿ and decodes it to these ranges. `bot/genome.json` overrides the defaults.

| Parameter | Range | Default | Meaning |
| --- | --- | --- | --- |
| `zone_margin` | 0 – 80 | 25 | metres to stay inside the next circle |
| `zone_lead` | 0 – 40 | 12 | extra seconds of travel slack before a shrink starts |
| `center_bias` | 0 – 1 | 0.35 | home point: 0 = next centre, 1 = margin from its edge |
| `quiet` | 0 – 1 | 0.3 | slow roaming; 1 = 40% speed, below the step-sound threshold |
| `camp` | 0 – 1 | 0.6 | >= 0.5: stand still at the home point |
| `scan_speed` | 0 – 3 | 0.8 | rad/s torso sweep when idle |
| `face_threat_s` | 0 – 8 | 3 | seconds to keep facing the last attacker's bearing |
| `face_sound` | 0 – 1 | 0.5 | >= 0.5: turn toward loud sounds heard in the last 2 s |
| `engage_range` | 30 – 230 | 160 | max distance to pick a mech as target |
| `tgt_w_hp` | 0 – 3 | 0.6 | target score weight on enemy hp fraction |
| `tgt_w_threat` | 0 – 3 | 1.0 | target score bonus for whoever hit us last |
| `aggr` | 0.3 – 3 | 1.0 | fight if our strength * aggr >= enemy hp, else flee |
| `max_enemies` | 1 – 4 | 1.5 | flee when more mechs than this are within engage range |
| `dodge_shell_t` | 0 – 3 | 1.5 | dodge mortar shells landing near us within this many seconds |
| `flee_hp` | 0 – 0.8 | 0.3 | flee below this hp fraction |
| `late_alive` | 2 – 10 | 5 | late game starts when this many mechs are left |
| `late_aggr` | 0.2 – 2 | 1.0 | aggression multiplier in the late game |
| `pref_dist` | 15 – 180 | 70 | preferred fighting distance |
| `lightning_dist` | 10 – 45 | 30 | preferred distance while holding lightning |
| `strafe` | 0 – 1 | 0.6 | sideways movement share while fighting |
| `strafe_period` | 0.3 – 3 | 1.0 | mean seconds between strafe direction changes |
| `lead` | 0 – 1.3 | 1.0 | fraction of target velocity to lead |
| `aim_tol` | 0.3 – 3 | 1.2 | fire when aim error < aim_tol * target radius |
| `fire_range` | 0.3 – 1 | 0.9 | fire the gun within this fraction of its range |
| `cover` | 0 – 1 | 0.5 | weight for breaking line of sight when fleeing |
| `dodge` | 0 – 1 | 0.3 | chance to jump / flip strafe when shot at or laser-aimed |
| `boost_energy` | 0 – 100 | 35 | boost only above this energy |
| `reload_frac` | 0 – 1 | 0.5 | reload early below this fraction of the magazine |
| `laser_release` | 0.3 – 1 | 1.0 | release the laser at this fraction of full charge |
| `w_mortar` | 0 – 1 | 0.5 | preference for the mortar |
| `w_laser` | 0 – 1 | 0.7 | preference for the laser |
| `w_lightning` | 0 – 1 | 0.4 | preference for lightning |
| `swap_margin` | 0 – 0.6 | 0.2 | preference gap needed to swap slot-1 weapons |
| `loot_dist` | 0 – 150 | 60 | max detour to a weapon pickup |
| `cube_dist` | 0 – 100 | 35 | max detour to a power cube |
| `crate_dist` | 0 – 120 | 40 | max detour to shoot a crate |
| `npc_dist` | 0 – 120 | 0 | max detour to fight an NPC drone |
| `npc_hp` | 0.3 – 1 | 0.8 | min hp fraction to fight drones |
| `drop_dist` | 0 – 300 | 120 | max distance to go for a supply drop |
| `drop_contest` | 0 – 80 | 40 | skip a drop if a visible mech is this close to its ring |

Upgrade preferences: one weight `up_<id>` in [0, 1] for each of 22 upgrades (`plating`, `armor`, `servos`, `capacitor`, `reactor`, `jets`, `nanites`, `gyros`, `sensors`, `gun_damage`, `gun_rate`, `gun_velocity`, `gun_mag`, `mortar_splash`, `mortar_reload`, `mortar_velocity`, `laser_charge`, `laser_damage`, `laser_pierce`, `lightning_chain`, `lightning_range`, `lightning_efficiency`). On an offer the bot picks the choice with the highest weight.

Total: **62 genes**.

## 4. Training hyperparameters (`sim/evolve.py`)
| Flag | Default | Meaning |
| --- | --- | --- |
| `--gens` | 2000 | CMA-ES generations (iterations) |
| `--pop` | 16 | Candidates per generation (a multiple of 16: one FFA group = 16 mechs) |
| `--sigma` | 0.2 | Initial step size in the [0, 1] box (sweep: 0.1 / 0.2 / 0.35) |
| `--ffa` | 12 | FFA matches each candidate plays per generation |
| `--hof` | 4 | Seats per FFA match given to hall-of-fame opponents (the default genome and past benchmark means) |
| `--prac` | 2 | Practice matches per candidate per generation (shared seeds) |
| `--w_ffa` | 0.5 | Fitness weight on FFA points; the rest goes to practice points |
| `--bench` | 25 | Score the distribution mean on fixed seeds every N generations |

Fitness is mean placement points (0–15). The benchmark plays 32 practice matches and 16 FFA matches (1 mean genome vs 15 default genomes) on fixed seeds, so its numbers compare across runs.

## 5. Simulator fidelity (`sim/engine.py`, `sim/geom.py`, `sim/house.py`)
- **Geometry is exact.** `sim/geom.py` is a numba port of the official web client's world code, covering polygon prisms, bilinear terrain, bridge decks, trunk cylinders, crown ellipsoids, move-and-slide, and the canopy rule. `sim/test_geom.py` compares it with answers from the client itself: 0 mismatches in 1000 queries each of surface height, sight, canopy transmittance, shot raycast and move-and-slide.
- **Where the client and protocol.md differ:** step height comes from `hello.config` (1.2 m; the client's world rules say 1.5). Trunks block movement (the client's push-out always includes them) but not shots. Slopes over 50° can't be walked up.
- **Rules come from protocol.md.** That covers weapons, damage, rear armor, vision wedge and memory, sounds, the zone, drops, drones, crates, cubes and upgrades. The server's own code for these is not public.
- **House bots are stand-ins:** our `Bot` with fixed parameters, scaled by aim noise and lead. They share vision, don't hurt each other, hunt the player weakest-first (one more per kill), and all hunt from 240 s.
- **Bot vs simulator:** the bot's own map (`bot/world.py`) is a fast 1 m grid with a trunk mask, used for steering and cover decisions. Only the simulator needs exact geometry.
- Evolved genomes can still exploit gaps in the rules or in the house-bot model. Confirm gains with real practice submissions.
