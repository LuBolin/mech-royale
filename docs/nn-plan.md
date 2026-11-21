# Neural policy: build plan

Goal: a neural-network bot that first copies `bot/brain.py` (with the GA-tuned genome), then improves through
self-play in the simulator. It ships only if it beats the GA bot on real practice submissions.

Work through the steps in order. Each step ends with a **gate**. Do not start the next step until the gate passes.
If a gate fails, fix the cause in that step. Report the gate numbers after each step.

## 0. Ground rules

- **Files.** New code goes in `nn/`. In `bot/`, add only `bot/features.py`, `bot/netbot.py` and the weights file,
  plus the small logging hook in `brain.py` (step 1). Do not change `brain.py` decisions. Do not change
  `Match`, `ffa` or `practice` signatures in `sim/`.
- **Python.** Use `.venv\Scripts\python.exe`. The sim needs numba, which is only in the venv.
- **Torch.** Install torch with CUDA into `.venv` (approved). It is for training only. It must never go into the
  Docker image. The image keeps numpy only.
- **Hardware.** 32 cores and an RTX 4060 Laptop GPU (8 GB). The GA run uses 24 worker processes
  (`sim/train_main.cmd`). Do not kill it. Ask the user how many cores you may use before long runs.
- **One feature module.** Training (in the sim) and the real bot must compute features with the same code:
  `bot/features.py`. Never write a second copy. A mismatch does not crash; the bot just plays badly.
- **Shortcuts.** Mark any known limit in code as `# shortcut: <limit>, <when to upgrade>`.

## 1. Facts already measured (do not re-measure)

| Fact | Value |
|---|---|
| Sim FFA match, 16 default genomes | ~3.3 s wall on one core, ~13,500 mech decisions, ends at t ≈ 200–250 s |
| Network size chosen | "2×": about 1.34M parameters, ~5.4 MB float32, ~2.7 MB float16 |
| Latency in the 1-CPU match container (bot + 1× net + features) | p99 6.9 ms end to end; 4× net p99 8.0 ms; budget 50 ms |
| Feature extraction (8 LOS + 16 rays + 16 cover LOS) | p50 0.7 ms, p99 1.7 ms |
| Repo limit | 20 MB at the submitted commit |
| Dockerfile | already sets `OPENBLAS_NUM_THREADS=1` etc. Keep it. |

Latency test kit: `nn/bench/net_bot.py` (bot + random-weight net, `NET_SCALE` env sets width) and
`nn/bench/check_posix.py` (works around `kit/check/check.py` turning `/arena` into `W:\arena` on Windows).
Command (PowerShell, from the repo root, after `docker build -t mech-bot .`):

```powershell
$arena = "$PWD\kit\arena"; .venv\Scripts\python.exe nn\bench\check_posix.py kit\check\check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "${arena}:/arena:ro" mech-bot
```

To time the bench bot instead, replace the final `mech-bot` with
`-e NET_SCALE=2 -v "$PWD\nn\bench:/bench:ro" --entrypoint python3 mech-bot -u /bench/net_bot.py`.
It prints per-step timings (brain, features, net) to stderr at exit.

## 2. Network design (fixed; do not redesign)

The network makes the **decisions**. Existing `brain.py` code does the **execution**: aim and lead (`shoot()`),
wall avoidance (`steer()`), shell dodging, reload, protocol.

**Inputs** (all positions and velocities relative to our mech, rotated into our yaw frame):
- Self (~95 numbers): hp fraction, energy, velocity, grounded, cubes, aim yaw/pitch as sin/cos, 2 weapons
  (kind one-hot, cooldown, ammo fraction, reloading, laser charge), upgrades bit mask (22), in_zone,
  zone (vector to current and next centre / radius, distance to edge, time to shrink start and end, dps),
  `t / 480`, `alive / 16`.
- Mechs and drones, up to 8 nearest (~25 numbers each): rel pos, distance, rel vel, hp fraction, cubes,
  weapon one-hots, `charging`, "aiming at me" (cos of their aim to us), line of sight now, seconds since seen,
  is_drone, is_ghost, ghost uncertainty radius.
- **Ghost tracker** (in `bot/features.py`): when a mech leaves `obs.mechs` (after the server's 5 s memory), keep
  it: last pos, last vel, hp, weapons, time since seen. Predicted pos = last pos + vel × dt, clamped to walkable
  ground. Uncertainty radius grows at 11.7 m/s. A `damaged` event from that id, or a matching sound bearing,
  refreshes it. Drop it after 20 s. Ghosts fill free mech slots.
- Projectiles, up to 8 (8 numbers each): rel pos, rel vel, time to closest approach, miss distance.
- Loot, up to 8 (10 each): rel pos, type one-hot, weapon kind. Drops, 3 (6 each): rel pos, time to land,
  capture progress, contested.
- Events (30): last damage bearing sin/cos, amount, rear flag, age; 4 most recent sounds (bearing sin/cos,
  distance bucket, kind, age).
- Terrain (32): 16 rays: free distance from `World.clear`, and cover (no LOS from the nearest threat to a point
  20 m along the ray).

**Layers** (width "2×"; see `nn/bench/net_bot.py` with `NET_SCALE=2` for the exact shapes):
self MLP 95→256→256; shared mech MLP 25→128→128, max and mean pooled; shared MLPs 64 wide for projectiles,
loot, drops (max and mean pooled); events 30→64; rays 32→64; concat → 512 (ReLU) → GRU 256.

**Heads** (all discrete):

| Head | Output | Label from `brain.py` | Execution |
|---|---|---|---|
| target | pointer over mech slots + "none" | `target["id"]` or none | `shoot()` aims and leads at it |
| move | 16 directions + stop | final `a["move"]` direction, rotated into our frame | direction → `steer()` |
| speed | quiet / walk / boost | move length < 0.42 → quiet, `a["boost"]` → boost | move length and boost |
| jump | yes / no | `a["jump"]` | `jump` |
| fire | allow, per slot (2 sigmoids) | `a["fire"]` | fire only if allowed **and** `shoot()` wants to fire |
| loot | none / take / take + drop slot 1 | `a["take"]`, `a["drop"]` | protocol |
| pick | pointer over the 3 offer choices | `a["pick"]` | protocol |
| mode (aux) | fight / flee / dodge / roam | `self.mode` | training only |
| value | final placement points 0–15 | match result | training only |

## Step 1: decision log and shared features

1. In `brain.py`, store the decision variables on `self` each tick (`self.last_target_id`, `self.mode`; the
   action dict already holds the rest). This is the only change to `brain.py`.
2. Write `bot/features.py`: `Encoder(world)` with `encode(obs) -> dict of numpy arrays` and
   `labels(obs, action, bot) -> dict`. It holds the ghost tracker, so it is stateful: one per mech per match.
3. Write `nn/test_features.py`: run one sim match, encode every observation, assert shapes, no NaN, all values in
   sane ranges, ghosts appear after a mech is out of view > 5 s, and labels match the action dict.

**Gate:** tests pass; `encode` p99 < 3 ms on one core.

## Step 2: imitation data

`nn/gen_data.py`: run sim matches (mix of `ffa` and `practice`) with the GA best genome and hall-of-fame /
perturbed genomes in the seats. Record (features, labels) for **all 16 mechs**, plus the final placement of each.
- With probability 0.2 per tick, execute a randomly perturbed action but keep the clean teacher label. This puts
  recovery states in the data.
- Save shards as `.npz` under `runs/nn/data/` (git-ignored). Target 1–2M samples to start.

**Gate:** dataset written; label distribution printed (target / mode / move / fire frequencies look sane).

## Step 3: imitation training (GPU)

`nn/model.py` (torch, layers as section 2) and `nn/train_bc.py`.
- Sequences of 32 ticks, GRU state carried within a sequence. Cross-entropy per head; mask heads that do not
  apply (fire when no target, pick when no offer). Hold out 10% of matches (not ticks) for validation.
- `nn/export.py`: write weights to `bot/policy.npz` as float16. Write a numpy forward pass in `bot/netbot.py`
  and a test that numpy and torch outputs agree (max abs diff < 1e-2).

**Gate:** in the sim, the BC network placed in a seat scores within noise of the teacher on held-out seeds
(≥ 300 matches each, matched seeds). If the training loss stays high, the network is too small: distill into a
wider one (train the wider net to copy this net's outputs). If train loss is low but validation is high, get more
data.

## Step 4: bot wrapper and container check

`bot/netbot.py`: a `Bot` subclass. Each tick: `features.encode` → numpy forward → execute the heads with the
existing `brain.py` helpers. Run one warm-up forward pass while handling `hello`. If anything throws, answer that
tick with `brain.act` (costs ~0.07 ms). `bot/main.py` picks the network bot only if `bot/policy.npz` exists.

**Gate:** the kit check with the command in section 1 passes, 0 late answers, p99 < 15 ms. Repo stays < 20 MB.

## Step 5: DAgger

The network drives; a shadow `brain.Bot` gets the same observations and labels each state. Add to the dataset,
retrain. 3–5 rounds, mixing in the teacher's action with probability 0.5 → 0.25 → 0.

**Gate:** sim placement ≥ teacher's on held-out seeds.

## Step 6: self-play fine-tuning (PPO)

Start only after the sim calibration fixes are merged. Their changes alter what the network learns.
- Start from the DAgger network. Add a KL penalty to the BC policy, so it stays near the expert and does not drift
  into sim quirks.
- Reward: placement points at match end (main), plus small shaped terms: damage dealt − damage taken (scaled),
  zone damage penalty, cubes gained. Keep shaped terms small next to placement.
- Opponents per match: GA hall-of-fame genomes, house bots (`practice`), frozen past network snapshots.
  About 30–50% self-play.
- Budget: about 2 hours to start (≈ 7–11k matches, ~100–150M decisions). Plot placement vs. training time.
  Continue only if the curve still rises.

**Gate:** beats the DAgger network and the GA bot in the sim on held-out seeds and opponent mixes.

## Step 7: decide what ships

1. Kit container check again (section 1).
2. Real practice submissions: GA bot vs. network bot. The network ships only if it scores higher there. The sim
   score alone is not enough.

## Do not

- Do not let the network control aim angles or raw trigger timing. That stays in `shoot()`.
- Do not put torch, or any package except numpy, in the Docker image.
- Do not commit data or checkpoints (`runs/` is git-ignored). Only `bot/policy.npz` is committed.
- Do not run PPO before the sim calibration fixes land.
