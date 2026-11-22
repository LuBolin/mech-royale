"""Arena model from the real map files, shared by the bot and the training simulator.

Built from v2/world.json + v2/terrain.i16 on a 1 m grid (the terrain's own vertex grid):
- `top`: highest ground-attached solid (terrain, buildings, stadium) per cell;
- `slab_lo/slab_hi`: elevated decks (prism cells whose base is above the ground: the 9 bridges).
  Mechs walk on them or under them; their underside stops a jet; they stop shots and sight;
- `trunk`: tree trunk tops (stop sight, not shots); `crown_lo/crown_hi`: crown ellipsoid span per
  column (soft: sight passes while exp(-canopyDensity * metres in crowns) >= canopyBlockT);
- `steep`: terrain cells steeper than max_slope_deg.
Tree and sight rules follow the official web client's world code (play.joinbuilders.org).
shortcut: overlapping crowns are merged into one span per 1 m column, and terrain is sampled at the
nearest vertex, not bilinearly; refine if sight checks disagree with server observations.
"""
import json
import hashlib
import math
import os
import tempfile
from array import array

import numpy as np

GRID_VERSION = 3
NONE_LO, NONE_HI = 1e9, -1e9
_CACHE = {}


def get_world(map_dir):
    w = _CACHE.get(map_dir)
    if w is None:
        w = _CACHE[map_dir] = World(map_dir)
    return w


class World:
    def __init__(self, map_dir, max_slope_deg=50.0):
        with open(os.path.join(map_dir, "v2", "world.json"), "rb") as f:
            world_bytes = f.read()
        w = json.loads(world_bytes)
        t = w["terrain"]
        digest = hashlib.sha256(world_bytes)
        with open(os.path.join(map_dir, "v2", t["file"]), "rb") as f:
            digest.update(f.read())
        self.map_digest = digest.hexdigest()
        self.nx, self.ny, self.x0, self.y0 = t["nx"], t["ny"], t["minX"], t["minY"]
        b = w["bounds"]
        self.bounds = (b["minX"], b["maxX"], b["minY"], b["maxY"])
        r = w.get("rules", {})
        self.canopy_density = r.get("canopyDensity", 0.35)
        self.canopy_block_t = r.get("canopyBlockT", 0.25)
        self.max_canopy_m = -math.log(self.canopy_block_t) / self.canopy_density
        names = ("ground", "top", "slab_lo", "slab_hi", "trunk", "crown_lo", "crown_hi", "steep", "tmask")
        cache = os.path.join(tempfile.gettempdir(), f"mech_grid_v{GRID_VERSION}_{max_slope_deg}_{self.map_digest}.npz")
        try:
            d = np.load(cache)
            g = {k: d[k] for k in names}
            assert g["top"].shape == (self.ny, self.nx)
        except Exception:
            g = self._build(map_dir, w, t, max_slope_deg)
            try:
                np.savez_compressed(cache, **g)
            except OSError:
                pass
        for k in names:
            setattr(self, k, g[k])
        self.sight_top = np.maximum(self.top, self.trunk)
        self._h = self.top.ravel().tolist()  # hot path: python floats
        f = lambda a: array("f", a.astype(np.float32).ravel().tobytes())  # noqa: E731
        self._slo, self._shi = f(self.slab_lo), f(self.slab_hi)
        self._st, self._clo, self._chi = f(self.sight_top), f(self.crown_lo), f(self.crown_hi)
        self._steep = array("b", self.steep.astype(np.int8).ravel().tobytes())
        self._tmask = array("b", self.tmask.astype(np.int8).ravel().tobytes())

    # ------------------------------------------------------------------ build
    def _build(self, map_dir, w, t, max_slope_deg):
        nx, ny = self.nx, self.ny
        raw = np.fromfile(os.path.join(map_dir, "v2", t["file"]), "<i2")
        ground = raw.reshape(ny, nx).astype(np.float32) * t["scale"]
        top = ground.copy()
        slo = np.full((ny, nx), NONE_LO, np.float32)
        shi = np.full((ny, nx), NONE_HI, np.float32)
        for p in w["prisms"]:
            box = self._box(p["ring"])
            if box is None:
                continue
            m = self._fill(p["ring"], box)
            holes = p.get("holes") or []
            if holes and not isinstance(holes[0], list):
                holes = [holes]
            for hole in holes:
                if len(hole) >= 6:
                    m &= ~self._fill(hole, box)
            j0, j1, i0, i1 = box
            s = (slice(j0, j1 + 1), slice(i0, i1 + 1))
            raised = p["base"] > ground[s] + 0.5
            solid, deck = m & ~raised, m & raised
            top[s][solid] = np.maximum(top[s][solid], p["top"])
            slo[s][deck] = np.minimum(slo[s][deck], p["base"])
            shi[s][deck] = np.maximum(shi[s][deck], p["top"])
        trunk = np.full((ny, nx), NONE_HI, np.float32)
        tmask = np.zeros((ny, nx), np.float32)  # cells a mech centre should avoid: trunk + 1.2 m
        clo = np.full((ny, nx), NONE_LO, np.float32)
        chi = np.full((ny, nx), NONE_HI, np.float32)
        tr = w["trees"]
        fi = {n: k for k, n in enumerate(tr["fields"])}
        d = np.asarray(tr["data"], float).reshape(-1, len(tr["fields"]))
        for row in d:
            x, y, z, hgt, cr, cb, trr = (row[fi[k]] for k in ("x", "y", "z", "height", "crownR", "crownBase", "trunkR"))
            cx = row[fi["crownX"]] if "crownX" in fi else x
            cy = row[fi["crownY"]] if "crownY" in fi else y
            ttop = z + max(cb + 0.5 * (hgt - cb), 3.0)
            self._disc(trunk, x, y, trr + 0.5, lambda dd: np.full(dd.shape, ttop), np.maximum)
            self._disc(tmask, x, y, trr + 1.2, lambda dd: np.ones(dd.shape), np.maximum)
            v, yy = z + cb, z + hgt
            cz, rz = (v + yy) / 2, max((yy - v) / 2, 0.5)
            half = lambda dd: rz * np.sqrt(np.clip(1 - (dd / cr) ** 2, 0, 1))  # noqa: E731
            self._disc(clo, cx, cy, cr, lambda dd: cz - half(dd), np.minimum)
            self._disc(chi, cx, cy, cr, lambda dd: cz + half(dd), np.maximum)
        gy, gx = np.gradient(ground.astype(np.float64))
        steep = np.hypot(gx, gy) > math.tan(math.radians(max_slope_deg))
        return {"ground": ground, "top": top, "slab_lo": slo, "slab_hi": shi, "trunk": trunk,
                "crown_lo": clo, "crown_hi": chi, "steep": steep, "tmask": tmask > 0}

    def _disc(self, grid, cx, cy, r, value, op):
        i0, i1 = max(0, int(cx - r - self.x0)), min(self.nx - 1, int(cx + r - self.x0) + 1)
        j0, j1 = max(0, int(cy - r - self.y0)), min(self.ny - 1, int(cy + r - self.y0) + 1)
        if i1 < i0 or j1 < j0:
            return
        X = self.x0 + np.arange(i0, i1 + 1)
        Y = self.y0 + np.arange(j0, j1 + 1)
        dd = np.hypot(X[None, :] - cx, Y[:, None] - cy)
        m = dd <= r
        sub = grid[j0:j1 + 1, i0:i1 + 1]
        sub[m] = op(sub[m], value(dd[m]).astype(np.float32))

    def _box(self, ring):
        xs, ys = ring[0::2], ring[1::2]
        i0 = max(0, int(math.floor(min(xs) - self.x0)))
        i1 = min(self.nx - 1, int(math.ceil(max(xs) - self.x0)))
        j0 = max(0, int(math.floor(min(ys) - self.y0)))
        j1 = min(self.ny - 1, int(math.ceil(max(ys) - self.y0)))
        return None if i1 < i0 or j1 < j0 else (j0, j1, i0, i1)

    def _fill(self, ring, box):
        """Even-odd point-in-polygon for the grid vertices inside box."""
        j0, j1, i0, i1 = box
        X = self.x0 + np.arange(i0, i1 + 1, dtype=np.float64)
        Y = self.y0 + np.arange(j0, j1 + 1, dtype=np.float64)
        xa = np.asarray(ring[0::2], float)
        ya = np.asarray(ring[1::2], float)
        xb, yb = np.roll(xa, -1), np.roll(ya, -1)
        inside = np.zeros((len(Y), len(X)), bool)
        for k in range(len(xa)):
            if ya[k] == yb[k]:
                continue
            rows = (ya[k] > Y) != (yb[k] > Y)
            if not rows.any():
                continue
            xi = xa[k] + (Y[rows] - ya[k]) * (xb[k] - xa[k]) / (yb[k] - ya[k])
            inside[rows] ^= X[None, :] < xi[:, None]
        return inside

    # ------------------------------------------------------------------ scalar queries
    def idx(self, x, y):
        i = int(x - self.x0 + 0.5)
        j = int(y - self.y0 + 0.5)
        i = 0 if i < 0 else self.nx - 1 if i >= self.nx else i
        j = 0 if j < 0 else self.ny - 1 if j >= self.ny else j
        return j * self.nx + i

    def h(self, x, y):
        """Top of the ground-attached solid (decks excluded)."""
        return self._h[self.idx(x, y)]

    def floor(self, x, y, z, step_up):
        """Surface a mech with feet at z stands on: a deck counts once the feet are near or above it."""
        k = self.idx(x, y)
        f = self._h[k]
        s = self._shi[k]
        if s > f and z >= s - step_up - 0.05:
            f = s
        return f

    def blocked(self, x, y, z, height, step_up):
        """A mech body (feet z) cannot occupy this column."""
        k = self.idx(x, y)
        if self._h[k] > z + step_up:
            return True
        lo = self._slo[k]
        return lo < z + height and self._shi[k] > z + step_up

    def ceiling(self, x, y, z):
        """Underside of a deck above z, or 1e9."""
        k = self.idx(x, y)
        lo = self._slo[k]
        return lo if lo >= z else NONE_LO

    def solid(self, x, y, z):
        k = self.idx(x, y)
        return z < self._h[k] or self._slo[k] <= z <= self._shi[k]

    def steep_at(self, x, y):
        return self._steep[self.idx(x, y)] == 1

    def is_open(self, x, y):
        k = self.idx(x, y)
        return self._h[k] - float(self.ground.flat[k]) < 0.5 and self._slo[k] > 1e8

    def los(self, ax, ay, az, bx, by, bz, spacing=2.0):
        """Shot line of fire: terrain, buildings and decks block; trees do not."""
        dx, dy, dz = bx - ax, by - ay, bz - az
        n = int(math.hypot(dx, dy) / spacing) + 1
        h, slo, shi, nx, ny, x0, y0 = self._h, self._slo, self._shi, self.nx, self.ny, self.x0, self.y0
        for k in range(1, n):
            f = k / n
            i = int(ax + dx * f - x0 + 0.5)
            j = int(ay + dy * f - y0 + 0.5)
            if 0 <= i < nx and 0 <= j < ny:
                c = j * nx + i
                zz = az + dz * f
                if zz < h[c] or slo[c] <= zz <= shi[c]:
                    return False
        return True

    def sees(self, ax, ay, az, bx, by, bz, spacing=1.0):
        """Sight line: solids and trunks block; crowns fade it (official canopy rule)."""
        dx, dy, dz = bx - ax, by - ay, bz - az
        L = math.sqrt(dx * dx + dy * dy + dz * dz)
        n = int(math.hypot(dx, dy) / spacing) + 1
        st, slo, shi, clo, chi = self._st, self._slo, self._shi, self._clo, self._chi
        nx, ny, x0, y0 = self.nx, self.ny, self.x0, self.y0
        inside = 0
        lim = self.max_canopy_m * n / max(L, 1e-6)
        for k in range(1, n):
            f = k / n
            i = int(ax + dx * f - x0 + 0.5)
            j = int(ay + dy * f - y0 + 0.5)
            if 0 <= i < nx and 0 <= j < ny:
                c = j * nx + i
                zz = az + dz * f
                if zz < st[c] or slo[c] <= zz <= shi[c]:
                    return False
                if clo[c] <= zz <= chi[c]:
                    inside += 1
                    if inside > lim:
                        return False
        return True

    def clear(self, x, y, z, ux, uy, dist, step_up, height=9.0):
        """True if walking dist metres along unit (ux, uy) meets no wall, ledge > step_up, low deck or trunk."""
        cur = z
        d = 1.5
        while d <= dist:
            k = self.idx(x + ux * d, y + uy * d)
            if self._tmask[k]:
                return False
            hh = self._h[k]
            s = self._shi[k]
            if s > hh and cur >= s - step_up - 0.05:
                hh = s
            elif self._slo[k] < cur + height and s > cur + step_up:
                return False
            if hh > cur + step_up + 0.2:
                return False
            cur = hh if hh > cur - 3 else cur - 3
            d += 1.5
        return True

    # ------------------------------------------------------------------ vectorised (simulator)
    def _samples(self, A, B, spacing):
        D = B - A
        L = np.hypot(D[:, 0], D[:, 1])
        for lo, hi in ((0, 40), (40, 100), (100, 1e9)):
            sel = np.flatnonzero((L >= lo) & (L < hi))
            if not len(sel):
                continue
            n = int(L[sel].max() / spacing) + 2
            f = np.linspace(0.0, 1.0, n)[1:-1]
            P = A[sel, None, :] + D[sel, None, :] * f[None, :, None]
            i = np.clip((P[..., 0] - self.x0 + 0.5).astype(np.int32), 0, self.nx - 1)
            j = np.clip((P[..., 1] - self.y0 + 0.5).astype(np.int32), 0, self.ny - 1)
            yield sel, P[..., 2], j, i, n

    def los_many(self, A, B, spacing=1.5):
        """Shot line of fire for rows of A -> B (n x 3). Endpoints are not tested."""
        out = np.ones(len(A), bool)
        for sel, Z, j, i, _ in self._samples(A, B, spacing):
            out[sel] = ~((Z < self.top[j, i]) | ((Z >= self.slab_lo[j, i]) & (Z <= self.slab_hi[j, i]))).any(1)
        return out

    def sees_many(self, A, B, spacing=1.0):
        """Sight lines for rows of A -> B, with trunks and the canopy transmittance rule."""
        out = np.ones(len(A), bool)
        if not len(A):
            return out
        L3 = np.linalg.norm(B - A, axis=1)
        for sel, Z, j, i, n in self._samples(A, B, spacing):
            hard = ((Z < self.sight_top[j, i]) | ((Z >= self.slab_lo[j, i]) & (Z <= self.slab_hi[j, i]))).any(1)
            canopy = ((Z >= self.crown_lo[j, i]) & (Z <= self.crown_hi[j, i])).sum(1) * (L3[sel] / (n - 1))
            out[sel] = ~hard & (canopy <= self.max_canopy_m)
        return out
