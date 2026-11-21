# Measuring aim, movement and duel outcomes

The runner uses the production Python bot with its saved parameters, the kit arena, the local simulator and an independently running copy of the Go reference bot. Baseline revisions are e889cb0a1d936a45d5caace9a5080d60f83694d4 for our bot and eb0aaabe80f288026be7082edee82725756458b6 for [the reference bot](https://github.com/justnau/giddygiddyboy-mech-royale/tree/eb0aaabe80f288026be7082edee82725756458b6).

## Open the saved results

[Download the interactive gallery](combat-gallery.zip?raw=true), extract the entire ZIP, and open index.html in your browser. The archive includes all 540 replays, the summary tables and the per-trial results. [View the same results as a CSV](combat-baseline.csv).

## Run

From the mech-royale repository, with Python and Go installed, run the following. The kit directory must contain arena/v2 and check/stream.jsonl.gz. The commands below extract the bundled kit into a local kit directory. Use a Python environment with NumPy and Numba installed. The baseline used Python 3.12, NumPy 2.4.6 and Numba 0.66.0.

```sh
mkdir -p kit
tar -xzf mechfight-kit.tar.gz --strip-components=1 -C kit
git clone https://github.com/justnau/giddygiddyboy-mech-royale ../reference-mech-bot
git -C ../reference-mech-bot checkout eb0aaabe80f288026be7082edee82725756458b6
go -C ../reference-mech-bot build -buildvcs=false -o ../reference-bot.exe ./cmd/bot
python sim/test_combat_skills.py
python sim/combat_trials.py --kit kit --reference ../reference-bot.exe --output runs/combat-current.csv
```

The final command prints the results and writes one CSV row per trial. Compare rows with the same setup, seed, observed player count, weapon, distance and movement in docs/combat-baseline.csv. Each seed chooses a map position and the simulator's random spread. The same pair starts with the same map position, facing and equipment.

## What each row measures

To compare range control, add `--weapon-swap --seeds 42` to the trial command. This runs two 65 m duels against the reference bot on the same firing lane. Our bot carries the gun in the first duel and lightning in the second. The CSV records seconds inside lightning range (3D barrel-to-target distance), time to first range entry, and health the lightning carrier lost before that first entry. If it never reaches range, the closing damage covers the entire duel.

- **Aim** holds our bot's feet still while it decides where to aim and when to fire. One weapon can fire per trial. The default target is the reference bot running its own decisions with its firing masked. Two repeatable targets cross the firing lane while jumping or using short jet bursts. The simulator applies the same acceleration, terrain collisions and energy rules to every target. The trial ends at death or 15 seconds. Gun, mortar and laser start at 30, 80 and 150 metres. Lightning starts at 15, 28 and 40 metres because its maximum range is 45 metres.
- **Open and cover** measure survival against each weapon. The reference bot controls its movement and the selected weapon. The default uses our production bot. After it chooses its action, the trial masks both fire buttons and passes its movement, aim, jump, boost and other commands through. The optional stationary control suppresses movement commands as well. Impacts can still push that bot. An open start has a clear sideways firing lane. A cover start has solid geometry that blocks a shot within 12 metres sideways of our spawn. Both bots can leave their starting area. The trial ends at death or 60 seconds. All three experiments use the same three ranges for each weapon.
- **Duel** lets both bots move and shoot, starting face-to-face with the same selected weapon at each range. The shooter columns describe our bot. A timeout with both alive is reported separately from a kill.
- **Observed alive** is the number sent in each observation. Two actors are physically present throughout. The sixteen-alive case measures how the same encounter changes when the bots believe other players are alive. There are no third-party attackers, pickups, drones or shrinking zone during the timed trials.
- **Shots and hits** count actual weapon discharges and shots that damaged the designated target before the trial ended. A mortar shell that deals both direct and splash damage counts once. Subtract direct_mortar from mortar hits for splash-only hits. The in_flight column counts unresolved shots when the trial stopped. A blank hit_pct means no shots were fired. A blank first_hit_s means no damage landed.
- **Damage** is health removed, including health recovered and removed again. It excludes damage beyond the target's remaining health. Damage per second includes time spent aiming, reloading or out of sight. visible_s counts simulated seconds when the shooter could see the target on the latest observation.
- **Movement** is measured after every physics step. Horizontal travel sums the distance along the path, including displacement from impacts. Airborne time includes jumps, falls and knockback. Jet time counts steps when the jet actually fired. These columns describe the opposing target in aiming and duels, and our defending bot in dodge trials.
- **Survival time** includes time spent behind cover. A 60-second survivor lived at least that long. Compare frozen and normal rows by seed and inspect incoming hits and visibility alongside survival. A control with zero hits provides no evidence of effective dodging.

## Interactive gallery

Run the same trial command with the gallery directory as an additional argument.

```sh
python sim/combat_trials.py --kit kit --reference ../reference-bot.exe --gallery runs/combat-gallery --output runs/combat-gallery/results.csv
```

Open runs/combat-gallery/index.html directly in a browser. Choose the experiment, weapon, starting range, movement, number of players reported alive, and map seed. The measurements and interactive replay use the same run. Press Play or Space to pause or resume. At the end, either starts the replay again. The 3D view shows the commands requested by the bot before the experiment disables movement or firing. In a dodge case, the paired-run link switches between normal and disabled movement at the same starting position. Keep the generated folder together when copying the gallery.
