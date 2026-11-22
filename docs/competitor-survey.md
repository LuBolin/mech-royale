# Competitor survey (2026-11-21)

Source: 57 cloned repos in `W:\Builders Hackathon\References\mech-royale\`. "Real" means the official practice board: our bot against a 15-bot house squad over 48 fixed seeds. Score claims come from each team's own logs and commit messages, and we have not verified them.

## Facts about the scorer and the house bots
- **Practice scoring is deterministic.** Viktor, Lance, Arnav, jack and nullat0m7 each resubmitted an identical bot and got an identical score. Re-salting only the bot's RNG moves the score by about ±0.06 to ±0.2. Read any delta under about 0.15 as noise.
- **Board top is about 5.2 to 5.6.** himanshu (C++) claims 5.58, Arnav got 5.13, poyraz 5.27.
- **Local sims mis-rank tactics.** Arnav found his sim anti-correlated with the board for combat changes. Economy farming was +1.03 in the sim and −0.57 real. Lance's gun+laser loadout was +1.1 in the sim and lost real. jack's sim ranked v7 best, and v7 scored worst officially. Use our sim to check that a mechanism fires. Do not use it to rank tactics.
- **House squad** (nullat0m7 `sim/house.ts:1-31`): 4 easy, 4 medium, 4 hard and 3 expert bots. The first hunter joins at about 20 s and more join per kill; all hunt by 240 s. Hunters learn our true position about every 8 s. That is why hiding to heal does not work in practice.
- **House laser** releases at exactly 2.75 s: 0.75 s charge plus 2 s hold (ch2li).
- **House gun accuracy** (jack): 90% under 40 m, 61% at 80–120 m, 36% at 120–160 m.
- **Mortar risk:** lasers do +60% damage to mortar carriers, and the laser can cook off the mortar. Strong bots drop the mortar or never hold it.
- **Plating beats armor:** ×1.15 max HP against ×1.087 effective HP (nullat0m7). The upgrade order otherwise made no measurable difference (jack).
- **Trees block sight, not shots.** Beyond 45 m, more than about 4 m of canopy hides a mech (xero `world.rs:229`, Warrior-hound, Hassene).
- **No clock-out:** the zone reaches radius 0 at about 415 s, before the 480 s limit.
- **Zone obs:** the obs gives the current circle while it is already shrinking (Arnav `bot.py:298`).

## Real-board results for behaviours we have or are considering
| Behaviour | Real delta | Source |
|---|---|---|
| Sidestep incoming bullets | **+0.5** (removing it: 4.31 → 3.81) | jack |
| Opening boost for the first 4 s | +0.21 | poyraz |
| Engage only mechs that shot at or aimed at us | +0.23 | jack (poyraz `eco_t` is the same idea) |
| Weapon pickup memory, 160 m search | +0.19 (a shorter radius cost −0.27) | jack |
| Commit to a fight and finish targets under 20% HP | +0.11 | jack |
| Stay put when idle | +0.10 | jack |
| Berserk: never retreat, hold 110 m, target hp×(1+d/120) | +1.3 combined | Arnav |
| Shorter ranges: laser 110 m, gun 90 m | 3.42 → 4.13 | alexwilldev |
| Looser gun gate, tolerance 1.6·atan(2/d) | 3.25 → 3.38 | viktor |
| **Hide or heal while hurt** | **−0.19, −0.45 ×2** | jack, nullat0m7 |
| Hide from lasers | −0.27 | Russell-Day |
| Cautious or patient variants | −0.5 to −0.6 | viktor, Lance |
| Long-range backpedal or kite | −0.77 | jack |
| Farming drones and crates | −0.23, −0.57 | jack, Arnav |
| Random strafe flips every 0.35–1.05 s | −0.09 | jack |
| Roof camping | 3.52 real, worse in every sim | alexwilldev, xero, niranjan |

## What this means for our bot
Our survival layer (commit 041922c) moved the sim from 1.11 to 1.41. It adds hide-until-regen (`bot/brain.py:12`, `:150`, `:188`) and broad flee logic. Two teams measured exactly these behaviours as losing on the real board, and their scores are far above our sim numbers. Our sim reward for passive play is probably the same mis-ranking the other teams saw.

## Candidates, by evidence and effort
**Tier 1: strong real evidence, small change**
1. Bullet sidestep: move perpendicular to incoming gun rounds, with a 1.2 s look-ahead and a 3.5 m miss distance. Today we only dodge mortar shells (`brain.py:424`). Reference: swan28pande `bot.py:756-782`, blended into movement at `:591-594`.
2. Gate the hide mode behind a genome knob, default off. Keep idle camping.
3. Hostility filter (practice): ignore mechs that have not shot at us, aimed at us within 0.12 rad (0.25 rad while charging) at under 220 m, or come within 45 m. Reference: jack `bot.py:169-180`, Abhiraj `bot.py:791-819`.
4. Weapon and loot memory with expiry, cleared once the spot is in LOS and empty. Fetch from up to 160 m when a slot is empty. Reference: himanshu `tracker.hpp:625-672`, Abhiraj `bot.py:715-735`.
5. Commit to a target and finish it below 20% HP. Bias toward never retreating in practice.
6. Opening boost for 4 s. Widen the genome bounds so shorter ranges and a looser gun gate are reachable.
7. Mortar: drop it or rank it last while a laser holder is visible (himanshu `combat.hpp:147`).

**Tier 2: good logic, medium evidence**
- FFA detector that switches the profile (final rounds are FFA, where caution pays). Signals: at least 2 deaths we did not cause before about 110 s, beams ending on other mechs, HP chunks of 44 or more we did not deal. References: niranjan `brain.rs:758-830`, Abhiraj `bot.py:585-615`, himanshu `tracker.hpp:524-558`, orbit `bot.py:212-285`.
- Hold the trigger when the cooldown ends inside the next tick. We fire at `cooldown <= 0.05` (`brain.py:407`); the tick is 0.1 s. Sim evidence only: xero `bot.rs:2131`.
- Laser release: predict aim one tick ahead, release early when partial charge kills, release before losing LOS, and never waste the 2 s auto-fire (himanshu `combat.hpp:476-540`, bshaurya `policy.py:568-581`).
- Ghost tracks: place an unseen shooter from the beam origin or from the damage bearing at a weapon-typical range, and fire back (xero `bot.rs:497-543`, Vivek `brain.py:262-280`).
- Anti-pincer: when threats wrap more than 150° around us, leave through the largest gap (jmonked `example_bot.py:1253`, himanshu `movement.hpp:668-800`).
- Final circle: stand on the exact centre once the next radius is 25 m or less, and track c(t+1.5 s).

**Tier 3: large work, unclear payoff**
- Grid A* with component labels and a jet-over-wall unstick (Abhiraj `tools/build_nav.py`, `nav.bin`). It fixes getting stuck in courtyards and the stadium bowl, not tactics.
- Roof climb state machine: Abhiraj turns it on only in FFA, and the board results are negative.

**Tooling**
- Real-engine harnesses that play private rooms on play.joinbuilders.org against the site's house bots: jack `tools/live_browser.js`, Arnav `dev/live/bridge.mjs`, Russell `devtools/play_ws.mjs`. The scripts' own notes say to use the organisers' server only with permission.
- Calibrate the sim's house squad to nullat0m7's model (tiers, join timing, intel every 8 s) and jack's tier aim numbers.
- Because the board is deterministic, every submission is an exact paired A/B. Change one thing per submission and log it.

## Sim A/B results (2026-11-21)
Paired seeds, `sim/ab.py`, 200 practice and 60 FFA matches per variant. The value after ± is one standard error. Raw tables are in `runs/ab/tier1.txt` and `runs/ab/tier2.txt`.

**Tier 1.** Base is the pre-survey bot (`params.OFF`). The FFA field is 15 copies of the pre-survey bot.
| Variant | Practice Δ | FFA Δ |
|---|---|---|
| sidestep | −0.10 ± 0.10 | **+2.48 ± 0.60** |
| no_hide | −0.07 ± 0.07 | +0.15 ± 0.19 |
| hostile | **−0.28 ± 0.11** | +0.57 ± 0.49 |
| loot_mem | −0.07 ± 0.09 | +0.37 ± 0.34 |
| finish | −0.03 ± 0.07 | +0.67 ± 0.32 |
| open_boost | −0.01 ± 0.14 | +0.78 ± 0.55 |
| mortar_laser | −0.04 ± 0.02 | −0.05 ± 0.16 |
| all Tier 1 | −0.25 ± 0.14 | **+2.90 ± 0.68** |

Decision: all of Tier 1 is on. The sim's practice mode disputes `hostile`, but jack measured +0.23 for it on the real board, and other teams found their sims wrong in practice mode. The board submission decides.

**Tier 2.** Base is the Tier 1 bot. The FFA field is 15 copies of the Tier 1 bot.
| Variant | Practice Δ | FFA Δ |
|---|---|---|
| ffa_detect (+ffa_aggr 0.8, third_party 0.5) | +0.01 ± 0.01 | +0.03 ± 0.27 |
| fire_early | **+0.15 ± 0.07** | 0.00 ± 0.27 |
| laser_smart | +0.04 ± 0.06 | −0.15 ± 0.17 |
| ghost_fire | −0.03 ± 0.09 | **−0.62 ± 0.41** |
| pincer | +0.02 ± 0.04 | +0.08 ± 0.09 |
| final_r 25 | 0 (never changed the outcome) | 0 |
| all Tier 2 | +0.06 ± 0.10 | −0.75 ± 0.40 |

Decision: only `fire_early` is on by default. The other Tier 2 knobs stay in the genome at off, so a GA can still try them.

A second Tier 2 run used 200 practice and 100 FFA matches, with 15 pre-survey bots as the FFA field. It is the current `runs/ab/tier2.txt`, which replaced the first run's table. It agrees with the first run:
- fire_early: +0.15 ± 0.07 practice, +0.27 ± 0.16 FFA.
- ghost_fire: **−0.82 ± 0.33** FFA.
- laser_smart: +0.04 / +0.07, noise.
- pincer: +0.02 / −0.12, noise.
- ffa_detect: +0.01 / +0.02, noise.
- final_r: exactly 0.
- All Tier 2: +0.06 / −0.59 ± 0.38.

## Second survey, all ~160 repos (2026-11-22)
Board: we are 4.73; top is Hermes-25 6.58. Scores below are each team's own claims.

**What the 6+ bots share.** giddy's Go core (6.06–6.19), andrew-zhouu/azing (6.23–6.29), automata (6.29, Rust) and Hermes' Afterglow (V201 branch `practice-loadout-v4`, 686 lines of Python). All of them: laser first, plating-first upgrades, fight everything in line of sight, never hide, short retreats only. Hermes adds a 3-step pickup itinerary planner (cubes, drops, lasers, with contest and upgrade-threshold value), shoots every visible mech in range, and picks the laser target by fewest shots to kill.

**Noise.** Arnav redrew giddy's 6.17 build with 15 RNG salts: mean about 5.65, sd 0.25. Lance saw 5.42 and 5.15 for the same commit. Read any single board delta under about 0.3 as noise. Teams draw extra samples by changing an RNG salt; we now have a `salt` knob for that.

**Real-board results worth acting on.**
- doni9977 on giddy's engine: no fight/threat boost, `LethalK` 0.7, target `HPw` 0.18 took 5.83 → 6.06 → 6.19. Arnav's giddy2 → giddy3 (same three changes) 5.25 → 6.17 on one draw.
- ParishruthiGanesh: mid-fight fetch of a better weapon within 110 m +0.44; 200 m −0.21. Fight range peak at 110 m (35 m 4.67, 80 m 4.65, 110 m 4.90, 150 m 4.50).
- Practice score is mostly kills in the first 3 minutes: idling from 180 s still scored 5.02 against 5.25 (ParishruthiGanesh).
- nullat0m7: survival play from the first tick 1.75 against 4.46. Every hide, heal or cover variant in this survey lost on the board.
- Contradiction: jackbalseca measured never-mortar and farming at −0.36 each when ported alone onto a Python bot.

**Measured house facts.** House gun hit rate (andrew-zhouu, 1,457 shots): 85% <40 m, 52% at 40–80, 26% at 80–120, 15% at 120–180. Epic (5,193 real bullets): long 0.6–1.4 s jink legs while boosting cut hits at 78–150 m from 54% to 26%; legs under 0.6 s barely help because reversing takes 0.4 s. Lightning is 240 dps inside 45 m and cannot miss.

**New knobs (`bot/params.py`, end of `defaults()`), sim A/B, house v2, 150 paired seeds 9000–9149, base 5.52:**
| Knob | Δ | Note |
|---|---|---|
| `arc_avoid` 1 (evade_v2 keeps out of lightning reach) | **+0.11 ± 0.07** | lightning deaths 18 → 10. **On in genome.json.** |
| `free_fire` 1 (laser only: shoot any visible mech when no target) | +0.01 ± 0.16 | fires on 0.9% of ticks in the sim; the sim's passive house rarely shows up in range |
| `hunt_hpw` 0.18 + `lethal_k` 0.7 + `grounded` 2 (doni's package) | +0.07 ± 0.20 | |
| above + `fight_loot_dist` 110 | +0.02 ± 0.20 | |
| above + `free_fire` + `fight_drop` | +0.11 ± 0.22 | salt 1 redraw: +0.09 |
| above + `arc_avoid` | +0.05 ± 0.21 | |

Knobs at their defaults reproduce the previous bot exactly (12 of 12 seeds identical). Only `arc_avoid` is clearly above sim noise. The doni package has the strongest real-board evidence of anything we found, so it is the first board A/B to run. Both pass the kit's recorded stream with 0 errors (p99 20 ms on a busy machine).

### Combined bot (2026-11-22, 12:30)
Five more knobs carry ideas from the 6+ bots: `laser_pulses` (Hermes: target the mech the fewest full beams finish), `explore` (Hermes: before 125 s without a laser, walk toward unswept 28 m cells and sweep the view), `cube_cluster` (Hermes: value piles and upgrade thresholds), `crowd_avoid` (Hermes: idle on the calmest of 10 ring points), `jink_hold` 0.6 (Epic: 0.6–1.4 s dodge legs).

Paired house-v2 practice; reference = previous bot with `arc_avoid` on. Seeds 9000–9149 and 9150–9299, pooled n = 300:
| Variant | 9000 set | 9150 set | Pooled |
|---|---|---|---|
| `laser_pulses` alone | +0.11 | +0.17 | +0.14 ± 0.09 |
| the five new knobs | +0.03 | +0.07 | +0.05 ± 0.17 |
| board package (doni's three edits, `fight_loot_dist` 110, `free_fire`, `fight_drop`, `laser_pulses`) | +0.13 | +0.20 | +0.16 ± 0.15 |
| **everything (board package + the five)** | **+0.32** | **+0.39** | **+0.36 ± 0.17** |

Alone, `explore`, `cube_cluster`, `crowd_avoid` and `jink_hold` each scored −0.15 to −0.27 on the first set, all within about 1.5 se. They only pay inside the full package. **`bot/genome.json` now holds the full package.** Against the bot at the start of the day (5.52 on seeds 9000–9149) it scores 5.95.

Free-for-all, 15 competitor images, 24 seeds (5100–5123): 7.38 → 7.42 (se 0.9), alive 101 → 114 s, kills 0.92 → 1.12. No loss. Kit stream: 0 errors, p99 3.4 ms on an idle machine. Unit tests pass.

If the board score drops, switch off the five Hermes/Epic knobs first (back to the board package), then the board package (back to `arc_avoid` only).

### Sim calibration, pruning and tuning (2026-11-22, 14:30)
**House aim calibrated.** `sim/house_aim.py` measures house gun hits on us by range. House v2 hit 69/46/33/24/9% at <40/40-80/80-120/120-180/180+ m; real measurements say about 88/55/30/15/3%. Aim noise that grows with range (`--house v2c`: tier noise × 0.6 × (range/80)^0.5) gives 95/57/30/16/2%.

**Board agreement** (`sim/rank_check.py`, 9 competitor images and 3 of our commits with known board scores, 32 seeds): rank correlation 0.66 for v2, **0.74 for v2c**. Tried and dropped, each worse: house bots that wake when shot or approached (0.64–0.69); an independent scripted house (0.59, far too easy). Every model still rates our bot about 0.8–1.3 above its board score and giddy, kanya and mike 0.6–1.8 below. The most likely cause is that the sim's house bots run our own brain code, so our bot is partly tuned against itself. Treat sim gains for our bot as optimistic.

**Knockouts of the combined bot** (v2c, seeds 9000–9149; positive = removing it helps): `grounded`→1 +0.43, farming off +0.19, `crowd_avoid` +0.18, `fight_drop` +0.17, `laser_pulses` +0.15, `free_fire` +0.15, `cube_cluster` +0.14, `jink_hold` off −0.38, `explore` off −0.21, the rest within ±0.06.

**Fresh-seed check** (9150–9299), against the deployed 4.73 bot:
| Bot | v2c | v2 |
|---|---|---|
| combined | +0.19 | +0.39 |
| pruned, `grounded` 1 | +0.18 | −0.10 |
| **pruned, `grounded` 2** | **+0.25** | **+0.19** |

The `grounded` knockout gain did not replicate, so `grounded` stays 2. **`bot/` now runs the pruned bot with `grounded` 2:** `arc_avoid`, `explore`, `jink_hold` 0.6, `hunt_hpw` 0.18, `lethal_k` 0.7, `fight_loot_dist` 110, farming off (`npc_dist` 0, `crate_dist` 40). `laser_pulses` stays in the code but off (it helped on v2 and hurt on v2c). The code for `free_fire`, `fight_drop`, `cube_cluster` and `crowd_avoid` is deleted: each lost in both houses. `bot/` reproduces the tested variant on 10 of 10 seeds.

**Tuning** (v2c, seeds 9300–9449, against the pruned bot at 5.13): `jink_hold` 0.4 −0.12, 0.9 +0.03; `laser_dist` 100 −0.08, 140 −0.06; `pref_dist` 70 −0.05, 100 −0.05. All flat. The values stay, and they agree with the board evidence (laser about 110–120 m, gun about 85 m).

## Arnav "v5" package (2026-11-21, late)
Our board score with Tier 1 on is 3.50; the top-32 bar is above 5.3. Practice score is close to a kill count, because house bots never attack each other. Three teams' real-board logs agree on one recipe:
- **Arnav** (`dev/OVERNIGHT.md`): never retreat + hold 110 m + target lowest hp×(1+d/120) in line of sight + mid-fight looting moved 3.81 → 5.13 as one package. Single knobs added on top (looser gun gate, slower strafe, selective engagement, sticky targets, economy) all scored lower.
- **alexwilldev** (`src/params.ts:29-30`): fight distance by weapon, gun 90 m and laser 110 m, scored 4.13. Laser distance 255 m gave 3.42, 150 m gave 3.85, 50 m gave 4.04. Lightning for close fights scored lower.
- **giddygiddyboy** (5.54, as read by Arnav): laser 120 m, gun 85 m, seek a laser first from up to 240 m.

All weapons outrange these distances (gun 230 m, laser 280 m). The distance sets hit rates, not reach: a gun round takes 0.4 s to cross 100 m, and the house gun hits 90% under 40 m but 61% at 80–120 m (jack).

New knobs, default off (`bot/params.py`, end of `SPEC`): `berserk`, `tgt_hpd`, `laser_dist`, `fight_loot`, `fight_loot_dist`. `bot/genome.json` turns the package on: those knobs plus `engage_range` 200, `pref_dist` 90, `laser_dist` 110, `w_laser` 1.0 and `w_lightning` 0. Our gun gate is already atan(3.0/d), about viktor's 1.6·atan(2/d), so it is unchanged.

Sim smoke test (`runs/ab/v5pkg.txt`, 96 paired practice seeds, base = defaults at 1.50): berserk −0.32 ± 0.17, range −0.18 ± 0.18, tgt_hpd +0.15 ± 0.12, fight_loot +0.20 ± 0.16, package −0.03 ± 0.17. Every knob fires and nothing crashes. The sim scores the same base at 1.50 that the board scores at 3.50, and Arnav's sim ranked 70 m against 110 m the opposite way (−0.91), so the board decides. The package passes the kit check in the match container: 300 of 300 answers in time, p99 2.8 ms.

Board submissions, one commit each, all on top of the package: package; package without `berserk`; package with `pref_dist` 110; package with `hostile_only` 0.

Other checks:
- **FFA detector:** at least 2 foreign deaths before 150 s latched in 0 of 64 practice matches and 4 of 4 FFA matches. A single early death misfired in 6 of 64 practice matches, so that rule was dropped.
- **Latency per tick** (6 practice matches): mean 0.14 ms, p99 1.3 ms, max 2.2 ms. That is unchanged from the pre-survey bot and well under the 50 ms limit.
