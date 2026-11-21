# Mech Royale

Build the AI for one of 16 mechs competing on the OSU Oval in a 3D battle royale. Buildings, rooftops, and trees provide cover as the playable zone shrinks over about seven minutes. Every mech starts with an arm gun and can pick up a mortar, charged laser, or lightning weapon. Kills and crates supply power cubes for upgrades. The last mech standing wins. [Play the arena in a browser](https://play.joinbuilders.org).

**Submission cutoff** is Monday at 00:00 GMT+8 (midnight at the start of Monday, Singapore time). The track lists a $5,000 prize.

## Scoring

Each match awards **16 − finishing place** points. First place earns 15, second earns 14, and sixteenth earns 0. Mechs destroyed on the same tick share a place. Kills earn no placement points directly.

A run's score is its mean placement points across all matches. Higher is better. The Python starter bot's practice baseline is **1.15** under scorer version `builders-3`. Earlier submissions win all score ties.

## Competition format

| Stage | Matches per entry | Opponents and outcome |
| --- | ---: | --- |
| Practice | 48 per submission | One team bot faces 15 cooperating house bots. The leaderboard shows each team's best score alongside the starter. At cutoff, the team's chosen submission locks with its own practice score. The best is selected by default. |
| Qualifying | 16 | The top 32 teams enter 16-mech free-for-alls against other qualifiers on new seeds. The top 16 by qualifying mean advance. Organizers announce their names without scores. |
| Final series | 64 | All 16 finalists play together in free-for-alls on new seeds. The last four matches run live at the venue and carry the same weight as the others. The series mean alone determines finalist ranks. |

**Practice opponents.** The squad has four easy, four medium, four hard, and three expert bots. They share vision and never attack each other. The weakest hunt first. One pursues you early, more join as you thin their ranks, and all remaining bots hunt you late. Practice seeds stay private and fixed throughout the event. Your bot never sees the seed, and its slot and spawn vary by match.

**Tournament exceptions and standings.**

- With fewer than 32 teams, everyone enters qualifying. With 16 or fewer, everyone plays the final series. Expert house bots fill empty slots.
- If a qualifying entry cannot be fetched or built, it sits out and the next team on the practice board takes its seat. An entry that cannot be fetched or built for the final scores nothing.
- Organizers may change match counts only before the final starts. Standings stay hidden until the last match ends.
- All finalists rank above eliminated qualifiers, who rank by qualifying score. Remaining teams follow in practice-board order, shown with negative scores.

## Submission and runtime

1. Put the bot in a public GitHub repository with a top-level `Dockerfile`. The starter archive includes Python and TypeScript examples. Any language, libraries, or AI are allowed. The repository limit is 20 MB at the submitted commit.
2. Build the image and run the kit's `check/check.py` under match limits. The event builds on `linux/amd64` with network access and a 10-minute limit. Pin dependency versions because finalists are rebuilt.
3. Submit the repository URL and optionally a commit. Leaving the commit blank selects the newest commit on the default branch. Teams may submit up to four times per hour. Keep the repository public and every submitted commit available until results are published.

Scored matches are bot-only. At 10 observations per simulated second, the bot reads what its mech sees and hears and sends movement, aim, and fire actions via `mechfight/1` JSON lines on stdin and stdout. Each reply has a **50 ms deadline**. A late reply is skipped. Without fresh input, the mech holds its last movement and aim for one second without firing, then stands still. Response times vary by a few milliseconds across runs and machines.

The runtime provides one CPU, 512 MB of memory, and a read-only map at `/arena`. The bot runs as UID 65534 without network access. Only `/tmp` is writable (64 MB). Install packages system-wide during the build.

Failed builds or conformance checks are rejected with a reason and leave the team's best score intact. A bot that passes the check but crashes or stops answering during a match stays in the arena as a target and receives its placement score.

The competition rules above come from the saved Builders Mech Royale track page. For the Docker test command, message schemas, movement, weapons, sensing, zone behavior, upgrades, and map format, see [protocol.md](protocol.md).
