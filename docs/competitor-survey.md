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

Other checks:
- **FFA detector:** at least 2 foreign deaths before 150 s latched in 0 of 64 practice matches and 4 of 4 FFA matches. A single early death misfired in 6 of 64 practice matches, so that rule was dropped.
- **Latency per tick** (6 practice matches): mean 0.14 ms, p99 1.3 ms, max 2.2 ms. That is unchanged from the pre-survey bot and well under the 50 ms limit.
