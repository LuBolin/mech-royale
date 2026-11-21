"""Create a portable case gallery from the same frames the merged 3D viewer consumes."""
import json
import shutil
import subprocess
from pathlib import Path

from viewer import MODES, static_map

HERE = Path(__file__).resolve().parent


def dump(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


class Gallery:
    def __init__(self, output, arena):
        self.out = output.resolve()
        self.cases = self.out / "cases"
        self.cases.mkdir(parents=True, exist_ok=True)
        world = static_map(str(arena))
        (self.out/"arena.js").write_text("const arena="+dump(world), encoding="utf-8")
        shutil.copyfile(HERE/"vendor/three.min.js", self.out/"three.min.js")
        source = (HERE/"viewer_template.html").read_text(encoding="utf-8")
        self.viewer = source.replace('<script>/*THREE*/</script>', '<script src="../three.min.js"></script><script src="../arena.js"></script>')
        self.viewer = self.viewer.replace('/*DATA*/null', 'Object.assign({map:arena}, replay)')
        self.viewer = self.viewer.replace('<span>Final place</span>', '<span>Outcome</span>')
        self.viewer = self.viewer.replace('${D.place} (${D.points} pts)', '${D.outcome}').replace('our place ${D.outcome}', '${D.outcome}')
        self.entries = []

    def save(self, row, recording, slot, cfg):
        frames,deaths,hits,shots = recording
        key = "-".join(str(row[k]) for k in ("setup","weapon","distance","movement","observed_alive","seed"))
        labels = ["rival","rival"]
        labels[slot] = "ours"
        alive = [bool(m[4]) for m in frames[-1]["m"]]
        if all(alive):
            outcome = "Both alive at limit"
        elif not any(alive):
            outcome = "Both bots down"
        else:
            outcome = "Our bot survived" if alive[slot] else "Our bot down"
        data = dict(mode=row["setup"], seed=row["seed"], genome="current bot", slot=slot, labels=labels,
                    frames=frames,deaths=deaths,hits=hits,shots=shots,logs={},outcome=outcome,modes=list(MODES),
                    dt=.1,mech={k:cfg["mech"][k] for k in ("radius","height","eye_height","muzzle_height")},
                    dropR=cfg["drops"]["radius"])
        (self.cases/(key+".js")).write_text("const replay="+dump(data),encoding="utf-8")
        self.write_case_page(key)
        self.entries.append(dict(**row,key=key,outcome=outcome))


    def write_case_page(self, key):
        html = self.viewer.replace('<script src="../arena.js"></script>',f'<script src="../arena.js"></script><script src="{key}.js"></script>')
        redirect = f'<script>if (window === window.top) location.replace("../index.html#{key}");</script>'
        html = html.replace("<head>", "<head>" + redirect, 1)
        (self.cases/(key+".html")).write_text(html,encoding="utf-8")

    def finish(self, rows, csv_path):
        assert len(self.entries)==len(rows)
        self.write_index(self.out, self.entries, csv_path)

    @staticmethod
    def write_index(output, entries, csv_path):
        results = output / "results.csv"
        if csv_path.resolve() != results.resolve():
            shutil.copyfile(csv_path, results)
        baseline = HERE.parent / "docs" / "combat-baseline.csv"
        matched = baseline.exists() and baseline.read_text() == results.read_text()
        root = HERE.parent.resolve()
        revision = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        info = dict(revision=revision, matched=matched)
        html = (HERE / "combat_gallery.html").read_text(encoding="utf-8")
        html = html.replace("/*CASES*/[]", dump(entries)).replace("/*BUILD*/{}", dump(info))
        (output / "index.html").write_text(html, encoding="utf-8")
